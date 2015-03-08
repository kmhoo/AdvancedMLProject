library(ggplot2)
library(stats)

incAxisLabelSpace = function(plot){
  # Input: ggplot object
  # Output: same ggplot object with increased space between axis
  #         and axis label
  new_plot = plot + theme(axis.title.x = element_text(vjust=-.1),
                          axis.title.y = element_text(vjust=1),
                          plot.title = element_text(vjust=.5))
  return(new_plot)
}

#import data
training <- read.csv("yelp_training.csv")
missing <- read.csv("missingUserInfo.csv")

#Subset for just use columns
users <- training[, c('user_id', 'u_votes_funny', 'u_votes_cool', 'u_votes_useful', 'u_review_count')]
missing_users <- missing[, c('user_id', 'u_votes_funny', 'u_votes_cool', 'u_votes_useful', 'u_review_count')]

#get unique users with no missing data
user_unique <- users[!duplicated(users$user_id),]
user_unique <- user_unique[!is.na(user_unique$u_votes_funny),]

#combine missing and unique
all_users <- rbind(user_unique, missing_users)

#Scale the columns
users_scaled <- scale(all_users[, 2:5])

#List of number of clusters
clusters <- c(1:15)

#Determine number of clusters from sum of squares
wss <- (nrow(users_scaled)-1)*sum(apply(users_scaled,2,var))
for (i in 2:15) wss[i] <- kmeans(users_scaled, centers=i, iter.max=50)$tot.withinss

#create dataframe with clusters and wss
data <- data.frame(Clusters=clusters, Sum.Of.Squares=wss)

#Plot the scree plot
k <- ggplot(data, aes(x=Clusters, y=Sum.Of.Squares))
k <- k + geom_line(color="blue", alpha=0.8)
k <- k + geom_point(color="blue", shape=15, size=3)
k <- k + ggtitle("Scree Plot of KMeans Clustering (Users)")
k <- k + xlab("Number of Clusters") + ylab("Within Groups Sum of Squares")
k <- incAxisLabelSpace(k)
k
ggsave(k, filename="FeatureEngineering/UserKMeansScreePlot.png", width=7, height=5, units="in")

#Calculating our kmean for 5 clusters
final_cluster <- kmeans(users_scaled, centers=5, iter.max=50)$cluster

with_cluster <- cbind(all_users, final_cluster)

full_dataset <- merge(users, with_cluster, by='user_id')
full_dataset <- full_dataset[,c('user_id', 'final_cluster')]

write.csv(full_dataset, '../KMeansClusters.csv', row.names=FALSE)
