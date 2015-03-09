library(reshape2)
#library(amap)

#import training data
training <- read.csv('yelp_training.csv')

ratings = subset(training, select=c(business_id, user_id, r_stars))
ratings = ratings[!duplicated(ratings),]

# Use just a subset!
ratings = ratings[1:100,]
n_train = 90

ratings_wide = dcast(ratings, user_id ~ business_id, value.var="r_stars")
ratings_mat = as.matrix(ratings_wide)

dist_obj = dist(ratings_mat, "euclidean", diag=TRUE, upper=TRUE)
dist_mat = as.matrix(dist_obj)

dist_mat = distance()

neighbors = sapply(1:nrow(dist_mat), function(i){
  excluded_users = c(i, (n_train+1):nrow(dist_mat))
  distances = dist_mat[i, ]
  closest_users = order(distances, decreasing=FALSE)
  closest_users = closest_users[!(closest_users %in% excluded_users)]
  closest_users = closest_users[1:3]
  return(closest_users)
})
neighbors=t(neighbors)


