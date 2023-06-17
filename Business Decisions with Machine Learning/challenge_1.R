# Load the libraries
suppressMessages(library(tidyverse))
suppressMessages(library(tidyquant))
suppressMessages(library(broom))
suppressMessages(library(umap))
suppressMessages(library(ggrepel))
suppressMessages(library(RColorBrewer))

# Part 1
sp_500_prices_tbl <- read_rds("sp_500_prices_tbl.rds")
sp_500_index_tbl <- read_rds("sp_500_index_tbl.rds")

# Select specific columns
sp_500_daily_returns_tbl <- sp_500_prices_tbl[, c("symbol", "date", "adjusted")]
sp_500_index_tbl <- sp_500_index_tbl[, c("symbol", "company", "sector")]

# Filtering dates starting from 2018
sp_500_daily_returns_tbl <- sp_500_daily_returns_tbl %>%
  filter(date >= as.Date("2018-01-01"))

# Computing lagged adjusted prices within each symbol
sp_500_daily_returns_tbl <- sp_500_daily_returns_tbl %>%
  group_by(symbol) %>%
  mutate(lagged_adjusted = lag(adjusted)) %>%
  na.omit()  # Removing NA values resulting from the lag operation

# Computing the difference and percentage return
sp_500_daily_returns_tbl <- sp_500_daily_returns_tbl %>%
  mutate(diff_adjusted = adjusted - lagged_adjusted,
         pct_return = diff_adjusted / lagged_adjusted)

# Selecting final columns
sp_500_daily_returns_tbl <- sp_500_daily_returns_tbl %>%
  select(symbol, date, pct_return)

# Saving as a variable named sp_500_daily_returns_tbl
sp_500_daily_returns_tbl <- as_tibble(sp_500_daily_returns_tbl)


# Part 2
# Spreading the date column and filling NAs with zeros
stock_date_matrix_tbl <- sp_500_daily_returns_tbl %>%
  pivot_wider(names_from = date, values_from = pct_return, values_fill = 0)


# Part 3
# Dropping the non-numeric column 'symbol'
numeric_data <- stock_date_matrix_tbl %>%
  select(-symbol)

# Performing K-Means clustering
kmeans_obj <- kmeans(numeric_data, centers = 4, nstart = 20)

# Extracting tot.withinss using glance()
tot_withinss <- glance(kmeans_obj)$tot.withinss

# Printing the tot.withinss value
print("total within-cluster sum of squares")
print(tot_withinss)


# Part 4
# 4.1
# Define the kmeans_mapper function
kmeans_mapper <- function(center = 3) {
  stock_date_matrix_tbl %>%
    select(-symbol) %>%
    kmeans(centers = center, nstart = 20)
}

# Create a tibble with column 'centers'
k_means_mapped_tbl <- tibble(centers = 1:30) %>%
  # Add 'k_means' column using map() and kmeans_mapper()
  mutate(k_means = map(centers, ~kmeans_mapper(.x))) %>%
  # Add 'glance' column using map() and glance() function
  mutate(glance = map(k_means, ~glance(.x)))

# 4.2
# Unnest the glance column
unnested_tbl <- k_means_mapped_tbl %>%
  unnest(glance)

# Create a Scree Plot
scree_plot <- ggplot(unnested_tbl, aes(x = centers, y = tot.withinss)) +
  geom_point() +
  geom_line() +
  labs(title = "Scree Plot") +
  theme_minimal()

# Save the Scree plot
ggsave("my_scree.png", scree_plot)

scree_plot

# Part 5
# 5.1
# Apply UMAP to stock_date_matrix_tbl
umap_results <- stock_date_matrix_tbl %>%
  select(-symbol) %>%
  umap()


# 5.2 (Creat UMAP Results TBL)
umap_results_tbl <- umap_results$layout %>%
  as_tibble() %>%
  bind_cols(stock_date_matrix_tbl %>% select(symbol))

# 5.3 (Visualize UMAP)
umap_plot <-
  ggplot(umap_results_tbl, aes(x = V1, y = V2)) +
  geom_point(alpha = 0.5) +
  theme_tq() +
  labs(title = "UMAP Projection")

# Save the image
# ggsave("umap_projection.png", umap_plot)

umap_plot

# Part 6
# 6.1 (Pull out K-Means for 10 centers)
k_means_obj <- kmeans(numeric_data, centers=10, nstart = 20)


# 6.2 (Combine k_means_obj with umap_results_tbl)
kmeans_stock_date_tbl <- k_means_obj %>%
  augment(stock_date_matrix_tbl) %>%
  select(symbol, .cluster)

umap_kmeans_results_tbl <- kmeans_stock_date_tbl %>% left_join(umap_results_tbl, by = "symbol") %>%
left_join(sp_500_index_tbl %>% select(symbol, company, sector), by = "symbol")

head(umap_kmeans_results_tbl, n = 10)


# 6.3 (Plot the k_means and umap results)
# Get the number of clusters
num_clusters <- length(unique(umap_kmeans_results_tbl$.cluster))

# Define a color palette with enough colors for the clusters
base_palette <- brewer.pal(min(num_clusters, 12), "Set3")  

# Create the plot
k_means_umap_plot <- umap_kmeans_results_tbl %>%
  ggplot(aes(x = V1, y = V2, color = factor(.cluster))) +
  geom_point(alpha = 0.5) +
  scale_color_manual(values = base_palette) +
  theme_tq() +
  labs(title = "K-Means Clustering with UMAP Projection")

# Save the plot
ggsave('k_means_umap_plot.png', k_means_umap_plot)

k_means_umap_plot