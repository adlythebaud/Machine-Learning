install.packages("arules")
library(arules)
groceries <- read.transactions("groceries.csv", sep = ",")

rules <- apriori(data = groceries, parameter = list(support = (20 / length(groceries)), confidence = .003, minlen = 2))

inspect(sort(rules, by = "support"))

inspect(sort(rules, by = "confidence")[1:5])

inspect(sort(rules, by = "lift")[1:5])
