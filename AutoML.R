# Load libraries
suppressMessages(library(h2o))
suppressMessages(library(GGally))
suppressMessages(library(tidyverse))

# Load and glimpse attrition data
employee_attrition_table <- read.csv("AutoML/HR-Employee-Attrition.txt")
employee_attrition_table %>% glimpse()

# Create function given in Business Data Case
plot_ggpairs <- function(data, color = NULL, density_alpha = 0.5) {
  
  color_expr <- enquo(color)
  
  if (rlang::quo_is_null(color_expr)) {
    
    g <- data %>%
      ggpairs(lower = "blank") 
    
  } else {
    
    color_name <- quo_name(color_expr)
    
    g <- data %>%
      ggpairs(mapping = aes_string(color = color_name), 
              lower = "blank", legend = 1,
              diag = list(continuous = wrap("densityDiag", 
                                            alpha = density_alpha))) +
      theme(legend.position = "bottom")
  }
  
  return(g)
  
}

# Create the plot
emp_att_plot <- employee_attrition_table %>%
  select(Attrition, MonthlyIncome, PercentSalaryHike, StockOptionLevel, EnvironmentSatisfaction, WorkLifeBalance, JobInvolvement, OverTime, TrainingTimesLastYear, YearsAtCompany, YearsSinceLastPromotion) %>%
  plot_ggpairs(color = Attrition)

# Save the plot
ggsave("emp_att_plot.png", emp_att_plot, width = 15, height = 15)

# Display the plot
emp_att_plot
