#Import data
training <- read.csv("../yelp_training.csv")
# testing <- read.csv("../yelp_test.csv")

#Fill in N/A's with review votes we have in dataset
votes_train <- training[is.na(training$u_votes_funny),]
votes_train <- votes_train[,c('user_id', 'r_votes_funny', 'r_votes_useful', 'r_votes_cool')]
votes_update_train <- aggregate(.~user_id, votes_train, FUN=sum)
names(votes_update_train) <- c('user_id', 'u_votes_funny', 'u_votes_useful', 'u_votes_cool')

# votes_test <- testing[is.na(testing$u_votes_funny),]
# votes_test <- votes_test[,c('user_id', 'r_votes_funny', 'r_votes_useful', 'r_votes_cool')]
# votes_update_test <- aggregate(.~user_id, votes_test, FUN=sum)
# names(votes_update_test) <- c('user_id', 'u_votes_funny', 'u_votes_useful', 'u_votes_cool')


#Fill in N/A's with review counts we have in dataset
count_train <- training[is.na(training$u_review_count),]
count_train <- count_train[,c('user_id', 'business_id')]
count_update_train <- aggregate(.~user_id, count_train, FUN=length)
names(count_update_train) <- c('user_id', 'u_review_count')

# count_test <- testing[is.na(testing$u_review_count),]
# count_test <- count_test[,c('user_id', 'business_id')]
# count_update_test <- aggregate(.~user_id, count_test, FUN=length)
# names(count_update_test) <- c('user_id', 'u_review_count')


#Fill in N/A's by averaging stars for each review
stars_train <- training[is.na(training$u_average_stars),]
stars_train <- stars_train[,c('user_id', 'r_stars')]
stars_update_train <- aggregate(.~user_id, stars_train, FUN=mean)
names(stars_update_train) <- c('user_id', 'u_average_stars')

# stars_test <- testing[is.na(testing$u_average_stars),]
# stars_test <- stars_test[,c('user_id', 'r_stars')]
# stars_update_test <- aggregate(.~user_id, stars_test, FUN=mean)
# names(stars_update_test) <- c('user_id', 'u_average_stars')


#Merge all of the columns together
merged_train <- merge(votes_update_train, count_update_train, by='user_id')
merged_train <- merge(stars_update_train, merged_train, by='user_id')

# merged_test <- merge(votes_update_test, count_update_test, by='user_id')
# merged_test <- merge(stars_update_test, merged_test, by='user_id')


#Write missing user info to csv
write.csv(merged_train, file="../missingUserInfoTraining.csv", row.names=FALSE)
# write.csv(merged_test, file="../missingUserInfoTesting.csv", row.names=FALSE)