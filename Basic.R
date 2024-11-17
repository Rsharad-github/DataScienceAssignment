# Ye ek single-line comment hai
print("Hello, R!")  # Is line ke baad bhi comment likha jaa sakta hai

# R Console
# source("your_script.R")

library(ggplot2) #import packages and library
library(dplyr)

# Getting Data into R
library(readr)
data <- read_csv("data.csv")
print(head(data))

# Saving Output in R 
write_csv(data, "output.csv")
print("Data saved in R!")

# Accessing Records and Variables in R
print(data$column_name)  # Ek column access karna
print(data[1, ])  # Pehla row access karna
