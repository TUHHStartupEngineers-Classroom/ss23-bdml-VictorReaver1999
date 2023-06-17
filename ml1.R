library(tidymodels)  # for the parsnip package, along with the rest of tidymodels

# Helper packages
library(broom.mixed) # for converting bayesian models to tidy tibbles

# Data set
bike_data_tbl <- readRDS("ds_data/bike_orderlines.rds")

bike_plot <- ggplot(bike_data_tbl,
                    aes(x = price, 
                        y = weight, 
                        group = category_1, 
                        col = category_1)) +
  geom_point() +
  geom_smooth(method = lm, se = FALSE) +
  scale_color_manual(values=c("black", "green", "skyblue", "purple", "brown"))

# ggsave("bike_plot.png", bike_plot)

# create model
lm_mod <- linear_reg() %>% 
  set_engine("lm")

lm_mod

lm_fit <- lm_mod %>% 
  fit(weight ~ price * category_1, 
      data = bike_data_tbl)

tidy(lm_fit)

# expand/create graph
new_points <- expand.grid(price = 2000, 
                          category_1 = c("E-Bikes", "Hybrid / City", "Mountain", "Road"))

# predict new weights
mean_pred <- predict(lm_fit, new_data = new_points)
mean_pred


conf_int_pred <- predict(lm_fit, 
                         new_data = new_points, 
                         type = "conf_int")
conf_int_pred


# Now combine: 
plot_data <- new_points %>% 
  bind_cols(mean_pred) %>% 
  bind_cols(conf_int_pred)

# and plot:
pred_plot <- ggplot(plot_data, aes(x = category_1)) + 
  geom_point(aes(y = .pred)) + 
  geom_errorbar(aes(ymin = .pred_lower, 
                    ymax = .pred_upper),
                width = .2) + 
  labs(y = "Bike weight", x = "Category") 

ggsave("pred_plot.png", pred_plot)