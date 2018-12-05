# Install these only if you're running this script for the first time.
# install.packages('e1071', dependencies = TRUE)
# install.packages('dummies', dependencies = TRUE)
library(e1071)
library(dummies)

##################################################
# GOAL:
# Classify the datasets
# Experiment with different C and gamma values.
##################################################

# 1. import datasets.
letters <- read.csv("letters.csv")

# 2. train test split
rows <- 1:nrow(letters)
data <- sample(nrow(letters), .3 * nrow(letters))
train <- letters[data,]
test <- letters[-data,]

# 3. Encode categorical data

# 4. Linear Scaling
# keeps <- c("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9")
# train[keeps] = scale(train[keeps])



# 5. Make model
model <- svm(letter~., data = train, kernel = "radial", gamma = 0.032, cost = 11)

# 6. Predict
prediction <- predict(model, test[,-1])

# 7. Confusion matrix
confusionMatrix <- table(pred = prediction, true = test$letter)

# 8. Calculate the accuracy, by checking the cases that the targets agreed
agreement <- prediction == test$letter
accuracy <- prop.table(table(agreement))

# 9. Print our results to the screen
print(confusionMatrix)
print(accuracy)

