library(datasets)
library(stats)
library(cluster)
myData = state.x77
str(myData)

# 1. Write to CSV, read CSV into dataframe
write.csv(myData,file="states.csv")
data <- read.csv("states.csv")

# 2. Save data
tempdata <- data

# 3. Remove non numeric column (states)
data <- data[2:9]

# first compute a distance matrix
distance = dist(as.matrix(data))

# now perform the clustering
hc = hclust(distance)

# finally, plot the dendrogram
plot(hc)


# 4. Feature Scale and Normalize
# lapply returns a matrix, so convert it back into a dataframe
data <- as.data.frame(lapply(data, scale))



# 5. make clusters
myclusters <- kmeans(data,3)
print(myclusters$size)
centers <- myclusters$centers
print(myclusters$centers)
print(myclusters$withinss)
print(myclusters$tot.withinss)

# 6. find elbow point
x <- list()
for (i in 1:25)
{
  myclusters <- kmeans(data,i)
  x[i] <- myclusters$tot.withinss
  
}

plot(c(1:25),x, xlab = "k", ylab = "myclusters$tot.withinss")

myclusters <- kmeans(data,5)
# 7. list the states in each cluster

clusters <- list("1", "2", "3", "4", "5")
for (i in 1:50)
{
  
}

cluster_id <- list(myclusters$cluster)
print(cluster_id)
write.csv(cluster_id,file="clusters.csv")
clusplot(data, myclusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)













