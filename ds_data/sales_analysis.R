# Data Science at TUHH ------------------------------------------------------
# SALES ANALYSIS ----

# 1.0 Load libraries ----
library(tidyverse)
library(lubridate)


bike_orderlines_tbl <- readRDS("C:\\Users\\fvsha\\Documents\\GitHub\\ss23-bdsb-VictorReaver1999\\Business_Data_Science\\ds_data\\01_bike_sales\\02_wrangled_data\\bike_orderlines.rds")

glimpse(bike_orderlines_tbl)

n <- 10
# Data Manipulation
top_customers_tbl <- bike_orderlines_tbl %>%
  
  # Select relevant columns
  select(bikeshop, total_price) %>%
  
  # Collapse the least frequent values into “other”
  mutate(bikeshop = as_factor(bikeshop) %>% fct_lump(n = n, w = total_price)) %>%
  
  # Group and summarize
  group_by(bikeshop) %>%
  summarize(revenue = sum(total_price)) %>%
  ungroup() %>%
  
  # Reorder the column customer_city by revenue
  mutate(bikeshop = bikeshop %>% fct_reorder(revenue)) %>%
  # Place "Other" at the beginning
  mutate(bikeshop = bikeshop %>% fct_relevel("Other", after = 0)) %>%
  # Sort by this column
  arrange(desc(bikeshop)) %>%
  
  # Add Revenue Text
  mutate(revenue_text = scales::dollar(revenue, 
                                       scale  = 1e-6, 
                                       prefix = "", 
                                       suffix = "M €")) %>%
  
  # Add Cumulative Percent
  mutate(cum_pct = cumsum(revenue) / sum(revenue)) %>%
  mutate(cum_pct_text = scales::percent(cum_pct)) %>%
  
  # Add Rank
  mutate(rank = row_number()) %>%
  mutate(rank = case_when(
    rank == max(rank) ~ NA_integer_,
    TRUE ~ rank
  )) %>%
  
  # Add Label text
  mutate(label_text = str_glue("Rank: {rank}\nRev: {revenue_text}\nCumPct: {cum_pct_text}"))

rich_customers <- top_customers_tbl %>%
  
  # Canvas
  ggplot(aes(revenue, bikeshop)) +
  
  # Geometries
  geom_segment(aes(xend = 0, yend = bikeshop), 
               color = RColorBrewer::brewer.pal(n = 11, name = "RdBu")[11],
               size  = 1) +
  
  geom_point(aes(size = revenue),
             color = RColorBrewer::brewer.pal(n = 11, name = "RdBu")[11]) +
  
  geom_label(aes(label = label_text), 
             hjust = "inward",
             size  = 3,
             color = RColorBrewer::brewer.pal(n = 11, name = "RdBu")[11]) +
  
  # Formatting
  scale_x_continuous(labels = scales::dollar_format(scale = 1e-6, 
                                                    prefix = "",
                                                    suffix = "M €")) +
  labs(
    title = str_glue("Top {n} Customers"),
    subtitle = str_glue(
      "Start: {year(min(bike_orderlines_tbl$order_date))}
               End:  {year(max(bike_orderlines_tbl$order_date))}"),
    x = "Revenue (M €)",
    y = "Customer",
    caption = str_glue("Top 6 customers contribute
                           52% of purchasing power.")
  ) +
  
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold"),
    plot.caption = element_text(face = "bold.italic")
  )

ggsave("rich_customers.png", rich_customers)
