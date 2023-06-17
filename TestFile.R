library(tidyverse)
library(modelr)

plot1 <- ggplot(sim1, aes(x, y)) + 
  geom_point(size = 2, colour = "#2DC6D6")

ggsave("plot1.png", plot1)

models <- tibble(
  a1 = runif(250, -20, 40),
  a2 = runif(250, -5, 5)
)

plot2 <- ggplot(sim1, aes(x, y)) + 
  geom_abline(aes(intercept = a1, slope = a2), data = models, alpha = 1/4) +
  geom_point()

ggsave("plot2.png", plot2)
