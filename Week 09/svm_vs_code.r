install.packages('e1071', dependencies = TRUE)
install.packages('dummies', dependencies = TRUE)
library(e1071)
library(dummies)

##################################################
# GOAL:
# Classify the datasets
# Experiment with different C and gamma values.
##################################################

# 1. import datasets.
vowel <- read.csv("vowel.csv")

# 2. train test split
rows <- 1:nrow(vowel)
data <- sample(nrow(vowel), .3 * nrow(vowel))
train <- vowel[data,]
test <- vowel[-data,]

# 3. Encode categorical data
train <- dummy.data.frame(train, names = c("Class"), sep="_")
print(typeof(train))


# 4. Linear Scaling
# train = scale(train)

