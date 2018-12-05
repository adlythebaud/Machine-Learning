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
vowel <- read.csv("vowel.csv")

# 2. train test split
rows <- 1:nrow(vowel)
data <- sample(nrow(vowel), .3 * nrow(vowel))
train <- vowel[data,]
test <- vowel[-data,]

# 3. Encode categorical data
# train <- dummy.data.frame(train, names = c("Class"), sep="_")

# Concat encoded columns into a new column
# train$target <- paste(train$Class_had,train$Class_hAd, train$Class_hed, 
#                       train$Class_hEd, train$Class_hid, train$Class_hId, 
#                       train$Class_hod, train$Class_hOd, train$Class_hud,  
#                       train$Class_hUd, train$Class_hYd )
# drop columns used for concat.
# keeps <- c("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "target")
# train <- train[keeps]


# 4. Linear Scaling
keeps <- c("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9")
train[keeps] = scale(train[keeps])




# 5. Make model
model <- svm(Class~., data = train, kernel = "radial", gamma = 0.034, cost = 21)

# 6. Predict
prediction <- predict(model, test[,-13])

# 7. Confusion matrix
confusionMatrix <- table(pred = prediction, true = test$Class)

# 8. Calculate the accuracy, by checking the cases that the targets agreed
agreement <- prediction == test$Class
accuracy <- prop.table(table(agreement))

# 9. Print our results to the screen
print(confusionMatrix)
print(accuracy)

