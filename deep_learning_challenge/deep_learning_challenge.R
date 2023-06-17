# Load the libraries

suppressMessages(library(tidyverse))
suppressMessages(library(keras))
suppressMessages(library(lime))
suppressMessages(library(rsample))
suppressMessages(library(recipes))
suppressMessages(library(yardstick))
suppressMessages(library(corrr))

# Load the data
churn_data_raw <- read.csv("Customer-Churn.csv")

# View the data
churn_data_raw %>% glimpse()

# Filter the data
churn_data_tbl <- churn_data_raw %>%
  select(Churn, everything(), -customerID) %>%
  tidyr::drop_na()

# Split test/training sets
set.seed(100)
train_test_split <- rsample::initial_split(churn_data_tbl, prop =0.8)
train_test_split

## <Analysis/Assess/Total>
## <5626/1406/7032>

# Retrieve train and test sets
train_tbl <- training(train_test_split)
test_tbl  <- testing(train_test_split)

# Create two plots of tenure counts, one with and the other without binning
counts_no_bins <- churn_data_tbl %>% ggplot(aes(x = tenure)) + 
  geom_histogram(binwidth = 0.5, fill =  "#2DC6D6") +
  labs(
    title = "Tenure Counts Without Binning",
    x     = "tenure (month)"
  )

ggsave("counts_no_bins.png", counts_no_bins)

counts_no_bins

counts_6_bins <- churn_data_tbl %>% ggplot(aes(x = tenure)) + 
  geom_histogram(bins = 6, color = "white", fill =  "#2DC6D6") +
  labs(
    title = "Tenure Counts With Six Bins",
    x     = "tenure (month)"
  )

ggsave("counts_6_bins.png", counts_6_bins)

counts_6_bins

# Create a plot of total charges
total_chg_plot <- churn_data_tbl %>% ggplot(aes(x = TotalCharges)) + 
  geom_histogram(bins = 100, fill =  "#2DC6D6") +
  labs(
    title = "TotalCharges Histogram, 100 bins",
    x     = "TotalCharges"
  )

ggsave("total_chg_plot.png", total_chg_plot)
total_chg_plot

churn_data_tbl_mod <- churn_data_tbl %>% 
  mutate(TotalCharges = log10(TotalCharges))
churn_data_tbl_mod %>% ggplot(aes(x = TotalCharges)) + 
  geom_histogram(bins = 100, fill =  "#2DC6D6") +
  labs(
    title = "TotalCharges Histogram, 100 bins",
    x     = "TotalCharges"
  )


# Determine if log transformation improves correlation 
# between TotalCharges and Churn

train_tbl %>%
  select(Churn, TotalCharges) %>%
  mutate(
    Churn = Churn %>% as.factor() %>% as.numeric(),
    LogTotalCharges = log(TotalCharges)
  ) %>%
  correlate() %>%
  focus(Churn) %>%
  fashion()


churn_data_tbl %>% 
  pivot_longer(cols      = c(Contract, InternetService, MultipleLines, PaymentMethod), 
               names_to  = "feature", 
               values_to = "category") %>% 
  ggplot(aes(category)) +
  geom_bar(fill = "#2DC6D6") +
  facet_wrap(~ feature, scales = "free") +
  labs(
    title = "Features with multiple categories: Need to be one-hot encoded"
  ) +
  theme(axis.text.x = element_text(angle = 25, 
                                   hjust = 1))


# Create recipe to transform our data
rec_obj <- recipe(Churn ~ ., data = train_tbl) %>%
  step_rm(Churn) %>% 
  step_discretize(tenure, options = list(cuts = 6)) %>%
  step_log(TotalCharges) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = T) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  prep(data = train_tbl)

# Apply the recipe
x_train_tbl <- bake( rec_obj , new_data =  train_tbl)
x_test_tbl  <- bake( rec_obj , new_data =  test_tbl)


y_train_vec <- ifelse( train_tbl$Churn == "Yes", TRUE, FALSE )
y_test_vec  <- ifelse( test_tbl$Churn  == "Yes", TRUE, FALSE)


# Building our Artificial Neural Network
model_keras <- keras_model_sequential()

model_keras %>% 
  # First hidden layer
  layer_dense(
    units              = 16, 
    kernel_initializer = "uniform", 
    activation         = "relu",
    input_shape        = ncol(x_train_tbl))%>% 
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  # Second hidden layer
  layer_dense(
    units              = 16, 
    kernel_initializer = "uniform", 
    activation         = "relu") %>% 
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  # Output layer
  layer_dense(
    units              = 1, 
    kernel_initializer = "uniform", 
    activation         = "sigmoid") %>% 
  # Compile ANN
  compile(
    optimizer = 'adam',
    loss      = 'binary_crossentropy',
    metrics   = c('accuracy')
  )

model_keras

# I managed to get to this point, but ran into an error that concerns dense_2_input.
# The rest of the code may be right or wrong, but it's hard to tell without being able to execute it.
# In theory, it should give the results we need. In view of not being able to train the model, I have
# decided to stop at this point. s

x_train_mrx = as.matrix(x_train_tbl)

ncol(x_train_tbl)

# Fit the model
fit_keras <- keras::fit(
  object = model_keras,
  x = x_train_tbl,
  y = y_train_vec ,
  epochs = 35 ,
  batch_size = 50 ,
  validation_split = 0.3
)

# View fit data
fit_keras

plot(fit_keras) +
  labs(title = "Deep Learning Training Results") +
  theme(legend.position  = "bottom",
        strip.placement  = "inside",
        strip.background = element_rect(fill = "#grey"))

# Predicted Class
yhat_keras_class_vec <- predict_classes(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()

# Predicted Class Probability
yhat_keras_prob_vec  <- predict_proba(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()

# Format test data and predictions for yardstick metrics
estimates_keras_tbl <- tibble(
  truth      = as.factor(y_test_vec) %>% fct_recode(yes = "1", no = "0"),
  estimate   = as.factor(yhat_keras_class_vec) %>% fct_recode(yes = "1", no = "0"),
  class_prob = yhat_keras_prob_vec
)

estimates_keras_tbl

# Confusion Table
estimates_keras_tbl %>% conf_mat(
  truth,
  estimate)

# Accuracy
estimates_keras_tbl %>% accuracy(truth, estimate)

# AUC
estimates_keras_tbl %>% roc_auc(
  data,
  truth,
  event_level = "second")

# Precision
tibble(
  precision = precision(
    data,
    truth),
  recall    = recall(
    data,
    truth)
)

# F1-Statistic
estimates_keras_tbl %>% f_meas(truth, estimate, beta = 1)

class(model_keras)

# Setup lime::model_type() function for keras
model_type.keras.engine.sequential.Sequential  <- function(x, ...) {
  return("classification")
}

# Setup lime::predict_model() function for keras
predict_model.keras.engine.sequential.Sequential <- function(x, newdata, type, ...) {
  pred <- predict_proba(object = x, x = as.matrix(newdata))
  return(data.frame(Yes = pred, No = 1 - pred))
}

library(lime)
# Test our predict_model() function
predict_model(x = model_keras, newdata = x_test_tbl, type = 'raw') %>%
  tibble::as_tibble()

# Run lime() on training set
explainer <- lime::lime(
  x_train_tbl,
  y_train_vec ,
  bin_continuous = FALSE)

explanation <- lime::explain(
  x_test_tbl[1:10,],
  explainer = explainer,
  n_labels   = 1,
  n_features = 51,
  kernel_width   = 1)

# Feature correlations to Churn
corrr_analysis <- x_train_tbl %>%
  mutate(Churn = y_train_vec) %>%
  correlate() %>%
  focus(Churn) %>%
  rename(feature = rowname) %>%
  arrange(abs(Churn)) %>%
  mutate(feature = as_factor(feature))
corrr_analysis

# Correlation visualization
corrr_plot <- corrr_analysis %>%
  ggplot(aes(x = ..., y = fct_reorder(..., desc(...)))) +
  geom_point() +
  
  # Positive Correlations - Contribute to churn
  geom_segment(aes(xend = ..., yend = ...),
               color = "red",
               data = corrr_analysis %>% filter(... > ...)) +
  geom_point(color = "red",
             data = corrr_analysis %>% filter(... > ...)) +
  
  # Negative Correlations - Prevent churn
  geom_segment(aes(xend = 0, yend = feature),
               color = "#2DC6D6",
               data = ...) +
  geom_point(color = "#2DC6D6",
             data = ...) +
  
  # Vertical lines
  geom_vline(xintercept = 0, color = "#f1fa8c", size = 1, linetype = 2) +
  geom_vline( ... ) +
  geom_vline( ... ) +
  
  # Aesthetics
  labs( ... )

# Save the plot
ggsave("corrr_plot.png", corrr_plot)

# View the plot
corrr_plot