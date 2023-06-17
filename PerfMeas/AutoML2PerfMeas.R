# Load the libraries

suppressMessages(library(tidyverse))
suppressMessages(library(GGally))
suppressMessages(library(h2o))
suppressMessages(library(recipes))
suppressMessages(library(rsample))
suppressMessages(library(knitr))
suppressMessages(library(cowplot))
suppressMessages(library(glue))


# load the dataset
product_backorders_tbl <- read_csv("product_backorders.csv")
# product_backorders_tbl %>% glimpse()

# Create data pre-processing recipe and convert certain variables to factors
recipe_obj <- recipe(went_on_backorder ~ ., data = product_backorders_tbl) %>%
  step_zv(all_predictors()) %>%
  step_mutate_at(potential_issue, deck_risk, oe_constraint, ppap_risk, stop_auto_buy, rev_stop, fn = as.factor) %>%
  prep()

# recipe_obj


# Split the dataset into train and test
set.seed(1234)

split_obj <- initial_split(product_backorders_tbl, prop = 0.85)
train_readable_tbl <- training(split_obj)
test_readable_tbl <- testing(split_obj)

# Pre-process it using the recipe
train_tbl <- bake(recipe_obj, new_data = train_readable_tbl)
test_tbl  <- bake(recipe_obj, new_data = test_readable_tbl)

# Initialize the H2O package
h2o.init()

# Use H2O to split train set into train and val set
split_h2o <- h2o.splitFrame(as.h2o(train_tbl), ratios = c(0.85), seed = 1234)
train_h2o <- split_h2o[[1]] 
valid_h2o <- split_h2o[[2]]
test_h2o  <- as.h2o(test_tbl)

y <- "went_on_backorder" # Target variable
x <- setdiff(names(train_h2o), y) # Assign predictor variables

# Initialize the automl process and specify the cross validation folds and 
# the leaderboard dataset
automl_models_h2o <- h2o.automl(
  x = x,
  y = y,
  training_frame    = train_h2o,
  validation_frame  = valid_h2o,
  leaderboard_frame = test_h2o,
  max_runtime_secs  = 120,
  nfolds            = 5
)

# Predict using H2O and save output as table
predictions <- h2o.predict(automl_models_h2o@leader, newdata = as.h2o(test_tbl))

# Create the predictions table
predictions_tbl <- predictions %>% as_tibble()

# Save the resulting model in the current working directory
# h2o.saveModel(automl_models_h2o@leader, path="saved_model/")

# End of AutoML2


# Start of Performance Measures
# Tasks:
# 1. Leaderboard visualization
# 2. Tune a model with grid search
# 3. Visualize the trade of between the precision and the recall and the optimal threshold
# 4. ROC Plot
# 5. Precision vs Recall Plot
# 6. Gain Plot
# 7. Lift Plot
# 8. Dashboard with cowplot

# Leaderboard visualization
# View the predictions table
predictions_tbl %>% glimpse()

# Exclude certain metrics from leaderboard
automl_models_h2o@leaderboard %>% 
  as_tibble() %>% 
  select(-c(mean_per_class_error, rmse, mse))

# Create the plotting function for the models based on AUC and LogLoss (taken from Business Case)
plot_h2o_leaderboard <- function(h2o_leaderboard, order_by = c("auc", "logloss"), 
                                 n_max = 20, size = 4, include_lbl = TRUE) {
  
  
  order_by <- tolower(order_by[[1]])
  
  leaderboard_tbl <- h2o_leaderboard %>%
    as_tibble() %>%
    select(-c(aucpr, mean_per_class_error, rmse, mse)) %>% 
    mutate(model_type = str_extract(model_id, "[^_]+")) %>%
    rownames_to_column(var = "rowname") %>%
    mutate(model_id = paste0(rowname, ". ", model_id) %>% as.factor())
  
  # Transformation
  if (order_by == "auc") {
    
    data_transformed_tbl <- leaderboard_tbl %>%
      slice(1:n_max) %>%
      mutate(
        model_id   = as_factor(model_id) %>% reorder(auc),
        model_type = as.factor(model_type)
      ) %>%
      pivot_longer(cols = -c(model_id, model_type, rowname), 
                   names_to = "key", 
                   values_to = "value", 
                   names_transform = list(key = forcats::fct_inorder)
      )
    
  } else if (order_by == "logloss") {
    
    data_transformed_tbl <- leaderboard_tbl %>%
      slice(1:n_max) %>%
      mutate(
        model_id   = as_factor(model_id) %>% reorder(logloss) %>% fct_rev(),
        model_type = as.factor(model_type)
      ) %>%
      pivot_longer(cols = -c(model_id, model_type, rowname), 
                   names_to = "key", 
                   values_to = "value", 
                   names_transform = list(key = forcats::fct_inorder)
      )
    
  } else {
    # If nothing is supplied
    stop(paste0("order_by = '", order_by, "' is not a permitted option."))
  }
  
  # Visualization
  g <- data_transformed_tbl %>%
    ggplot(aes(value, model_id, color = model_type)) +
    geom_point(size = size) +
    facet_wrap(~ key, scales = "free_x") +
    labs(title = "Leaderboard Metrics",
         subtitle = paste0("Ordered by: ", toupper(order_by)),
         y = "Model Postion, Model ID", x = "")
  
  if (include_lbl) g <- g + geom_label(aes(label = round(value, 2), 
                                           hjust = "inward"))
  
  return(g)
  
}

# Use our function to create the plot
h2o_plot <- automl_models_h2o@leaderboard %>% plot_h2o_leaderboard()

# Save the plot
ggsave("h2o_plot.png", h2o_plot, width=15, height=15)

h2o_plot

# Grid Search

# Load the model we saved
deeplearning_h2o <- h2o.loadModel("saved_model/StackedEnsemble_AllModels_3_AutoML_1")

# View all of its parameters
deeplearning_h2o@allparameters

Deeplearning_grid_01 <- h2o.grid( algorithm = "deeplearning",grid_id = "Deaplearning_grid_01",
  
     # Predictor and response variables
     x = x,
     y = y,

     # Traind and Validation sets + the number of folds for CV
     training_frame   = train_h2o,
     validation_frame = valid_h2o,
     nfolds = 5,

     # Hyperparamters
     hyper_params = list(
         # Use some combinations (the first one was the original)
         hidden = list(c(10, 10, 10), c(50, 20, 10), c(20, 20, 20)),
         epochs = c(10, 50, 100)
     )
 )


# Get the 3rd model from H2O's Deep Learning Grid Models, Save it, and then load it
# Deeplearning_grid_01_model_3 <- h2o.getModel("Deaplearning_grid_01_model_3")
# Deeplearning_grid_01_model_3 %>% h2o.saveModel(path = "save_model/Deaplearning_grid_01_model_3")
Deeplearning_grid_01_model_3 <- h2o.loadModel("save_model/Deaplearning_grid_01_model_3")

# Evaluate and save its performance and then transform it into a table through h2o metric extraction
performance_h2o <- h2o.performance(Deeplearning_grid_01_model_3, newdata = as.h2o(test_tbl))
performance_tbl <- performance_h2o %>% h2o.metric() %>% as.tibble()

# specify a new theme 
theme_new <- theme(
  legend.position  = "bottom",
  panel.background = element_rect(fill   = "transparent"),
  panel.border     = element_rect(color = "black", fill = NA, size = 0.5),
  panel.grid.major = element_line(color = "grey", size = 0.333)
  ) 

# Save the performance table
saveRDS(performance_tbl, file = "performance_tbl.rds")

# Load the performance table
performance_tbl <- readRDS("performance_tbl.rds")


# Visualize the trade off between the precision and the recall and the optimal threshold
performance_tbl %>%
  filter(f1 == max(f1))

precision_vs_recall_plot_optim <- performance_tbl %>%
  ggplot(aes(x = threshold)) +
  geom_line(aes(y = precision), color = "blue", size = 1) +
  geom_line(aes(y = recall), color = "red", size = 1) +
  
  # Insert line where precision and recall are harmonically optimized
  geom_vline(xintercept = h2o.find_threshold_by_max_metric(performance_h2o, "f1")) +
  labs(title = "Precision vs Recall", y = "value") +
  theme_new

ggsave("precision_vs_recall_plot_optim.png", precision_vs_recall_plot_optim)

precision_vs_recall_plot_optim

# Visualize the ROC Plot
ROC_plot <- performance_tbl %>%
  ggplot(aes(fpr, tpr)) +
  geom_line(size = 1) +
  
  # just for demonstration purposes
  geom_abline(color = "red", linetype = "dotted") +
  
  theme_new +
  theme(
    legend.direction = "vertical",
  ) +
  labs(
    title = "ROC Plot",
    subtitle = "Performance of 3 Top Performing Models"
  )

ggsave("ROC_plot.png", ROC_plot)

ROC_plot

# Precision vs Recall Plot
prec_vs_recall <- performance_tbl %>%
  ggplot(aes(recall, precision)) +
  geom_line(size = 1) +
  theme_new + 
  theme(
    legend.direction = "vertical",
  ) +
  labs(
    title = "Precision vs Recall Plot",
    subtitle = "Performance of 3 Top Performing Models"
  )

ggsave("Prec_Vs_Recall.png", prec_vs_recall)

prec_vs_recall

# Creating the ranked predictions table
ranked_predictions_tbl <- predictions_tbl %>%
  bind_cols(test_tbl) %>%
  select(predict:Yes, went_on_backorder) %>%
  # Sorting from highest to lowest class probability
  arrange(desc(Yes))

# Creating the gain and lift table from the ranked predictions table
calculated_gain_lift_tbl <- ranked_predictions_tbl %>%
  mutate(ntile = ntile(Yes, n = 10)) %>%
  group_by(ntile) %>%
  summarise(
    cases = n(),
    responses = sum(went_on_backorder == "Yes")
  ) %>%
  arrange(desc(ntile)) %>%
  
  # Add group numbers (opposite of ntile)
  mutate(group = row_number()) %>%
  select(group, cases, responses) %>%
  
  # Calculations
  mutate(
    cumulative_responses = cumsum(responses),
    pct_responses        = responses / sum(responses),
    gain                 = cumsum(pct_responses),
    cumulative_pct_cases = cumsum(cases) / sum(cases),
    lift                 = gain / cumulative_pct_cases,
    gain_baseline        = cumulative_pct_cases,
    lift_baseline        = gain_baseline / cumulative_pct_cases
  )


# Obtaining performance metrics for the gain_lift_table
gain_lift_tbl <- performance_h2o %>%
  h2o.gainsLift() %>%
  as.tibble()

# Transform the gain_lift_table
gain_transformed_tbl <- gain_lift_tbl %>% 
  select(group, cumulative_data_fraction, cumulative_capture_rate, cumulative_lift) %>%
  select(-contains("lift")) %>%
  mutate(baseline = cumulative_data_fraction) %>%
  rename(gain     = cumulative_capture_rate) %>%
  # prepare the data for the plotting (for the color and group aesthetics)
  pivot_longer(cols = c(gain, baseline), values_to = "value", names_to = "key")

gain_transformed_plot <- gain_transformed_tbl %>%
  ggplot(aes(x = cumulative_data_fraction, y = value, color = key)) +
  geom_line(size = 1.5) +
  labs(
    title = "Gain Chart",
    x = "Cumulative Data Fraction",
    y = "Gain"
  ) +
  theme_new

ggsave("gain_transformed_plot.png", gain_transformed_plot)

gain_transformed_plot


# Creating the lift_transformed_table
lift_transformed_tbl <- gain_lift_tbl %>% 
  select(group, cumulative_data_fraction, cumulative_capture_rate, cumulative_lift) %>%
  select(-contains("capture")) %>%
  mutate(baseline = 1) %>%
  rename(lift = cumulative_lift) %>%
  pivot_longer(cols = c(lift, baseline), values_to = "value", names_to = "key")

lift_transformed_plot <- lift_transformed_tbl %>%
  ggplot(aes(x = cumulative_data_fraction, y = value, color = key)) +
  geom_line(size = 1.5) +
  labs(
    title = "Lift Chart",
    x = "Cumulative Data Fraction",
    y = "Lift"
  ) +
  theme_new

ggsave("lift_transformed_plot.png", lift_transformed_plot)

lift_transformed_plot


# Creating a dashboard using cowplot
# cowplot::get_legend extracts a legend from a ggplot object
p_legend <- get_legend(ROC_plot)
# Remove legend from p1
ROC_plot <- ROC_plot + theme(legend.position = "none")

# cowplot::plt_grid() combines multiple ggplots into a single cowplot object
cowplotgrid <- cowplot::plot_grid(ROC_plot, prec_vs_recall, gain_transformed_plot, lift_transformed_plot,  ncol = 2)
ggsave("cowplot.png", cowplotgrid)

# Display cowplot
cowplotgrid