library(ggplot2)
library(stats)
setwd('/Users/kaileyhoo/Documents/MSAN/Module 3/MSAN630/Project/AdvancedMLProject/')

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
missing <- read.csv("missingUserInfoTraining.csv")

#Subset for just use columns
users <- training[, c('user_id', 'u_votes_useful', 'u_review_count', 'u_average_stars')]
missing_users <- missing[, c('user_id', 'u_votes_useful', 'u_review_count', 'u_average_stars')]

#get unique users with no missing data
user_unique <- users[!duplicated(users$user_id),]
user_unique <- user_unique[!is.na(user_unique$u_votes_useful),]

#combine missing and users
all_users <- rbind(user_unique, missing_users)

#Scale the columns
users_scaled <- scale(all_users[, 2:4])

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
ggsave(k, filename="FeatureEngineering/UserKMeansScreePlotInit.png", width=7, height=5, units="in")

#########

#RUN WITH SECOND INTERATION OF PROCESSING

#import data
training2 <- read.csv("training_2.csv")

#subset for correct columns
users2 <- training2[, c('user_id', 'u_votes_useful_update', 'u_review_count_update', 'u_stars_update')]

#get unique users with no missing data
user_unique2 <- users2[!duplicated(users2$user_id),]
user_unique2 <- user_unique2[!is.na(user_unique2$u_votes_useful),]

#Scale the columns
users_scaled2 <- scale(user_unique2[, 2:4])

#List of number of clusters
clusters <- c(1:15)

#Determine number of clusters from sum of squares
wss2 <- (nrow(users_scaled2)-1)*sum(apply(users_scaled2,2,var))
for (i in 2:15) wss2[i] <- kmeans(users_scaled2, centers=i, iter.max=50)$tot.withinss

#create dataframe with clusters and wss
data2 <- data.frame(Clusters=clusters, Sum.Of.Squares=wss2)

#Plot the scree plot
k2 <- ggplot(data2, aes(x=Clusters, y=Sum.Of.Squares))
k2 <- k2 + geom_line(color="blue", alpha=0.8)
k2 <- k2 + geom_point(color="blue", shape=15, size=3)
k2 <- k2 + ggtitle("Scree Plot of KMeans Clustering (Users)")
k2 <- k2 + xlab("Number of Clusters") + ylab("Within Groups Sum of Squares")
k2 <- incAxisLabelSpace(k2)
k2
ggsave(k2, filename="FeatureEngineering/UserKMeansScreePlot2.png", width=7, height=5, units="in")

# #Calculating our kmeans using second process for 5 clusters
# final_cluster <- kmeans(users_scaled, centers=5, iter.max=50)$cluster
# with_cluster <- cbind(all_users, final_cluster)
# 
# #Subset to just users and clusters
# user_cluster <- with_cluster[,c('user_id', 'final_cluster')]
# write.csv(user_cluster, '../KMeansClustersInit.csv', row.names=FALSE)