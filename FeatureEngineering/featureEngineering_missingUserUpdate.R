#Import data
training <- read.csv("yelp_training.csv")

#Fill in N/A's with review votes we have in dataset
votes <- training[is.na(training$u_votes_funny),]
votes <- votes[,c('user_id', 'r_votes_funny', 'r_votes_useful', 'r_votes_cool')]
votes_update <- aggregate(.~user_id, votes, FUN=sum)
names(votes_update) <- c('user_id', 'u_votes_funny', 'u_votes_useful', 'u_votes_cool')

#Fill in N/A's with review counts we have in dataset
count <- training[is.na(training$u_review_count),]
count <- count[,c('user_id', 'business_id')]
count_update <- aggregate(.~user_id, count, FUN=length)
names(count_update) <- c('user_id', 'u_review_count')

#Fill in N/A's by averaging stars for each review
stars <- training[is.na(training$u_average_stars),]
stars <- stars[,c('user_id', 'r_stars')]
stars_update <- aggregate(.~user_id, stars, FUN=mean)
names(stars_update) <- c('user_id', 'u_average_stars')

merged <- merge(votes_update, count_update, by='user_id')
merged <- merge(stars_update, merged, by='user_id')

write.csv(merged, file="../missingUserInfo.csv", row.names=FALSE)