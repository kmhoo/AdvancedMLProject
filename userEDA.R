library(ggplot2)
library(reshape2)
library(GGally)

incAxisLabelSpace = function(plot){
  # Input: ggplot object
  # Output: same ggplot object with increased space between axis
  #         and axis label
  new_plot = plot + theme(axis.title.x = element_text(vjust=-.1),
                          axis.title.y = element_text(vjust=1),
                          plot.title = element_text(vjust=.5))
  return(new_plot)
}

## Import data

#setwd("/Users/griffin/Google Drive/USF/Spring1/AdvancedMachineLearning/AdvancedMLProject")
#training = read.csv("yelp_training.csv")

## Separate out individual data sets

users = training[,grepl("u_|user_id", names(training))]
users = users[!duplicated(users),]
# Number of users in review+user dataset:
nrow(users)
# Number of users in review set with no matches in user set:
sum(is.na(users$u_average_stars))


## Plots

#setwd("/Users/griffin/Google Drive/USF/Spring1/AdvancedMachineLearning/AdvancedMLProject/Plots")

# Create a melted data frame of users votes
# Each row has the vote type (cool, funny, useful) and a number of votes
# for one particular user
userVotesMelt = melt(subset(users, 
                            select=c(u_votes_cool, u_votes_funny, u_votes_useful)))
names(userVotesMelt) = c('vote_type', 'num')
userVotesMelt = userVotesMelt[!is.na(userVotesMelt$num),]
userVotesMelt$vote_type = as.factor(gsub("u_votes_", "", 
                                         as.character(userVotesMelt$vote_type)))

# Boxplots of user votes by vote type/category
p1 = ggplot(userVotesMelt, aes(y=num, x=vote_type))
p1 = p1 + geom_boxplot(varwidth=TRUE, fill='lightblue') 
p1 = p1 + ggtitle("Boxplots of Number of User Votes by Vote Type")
p1 = p1 + xlab("Vote Type") + ylab("Number of Votes Across Reviews per User")
p1 = p1 + coord_flip()
p1 = incAxisLabelSpace(p1)
p1
ggsave(p1, filename="userVotesBoxplots.png", height=6, width=6, units="in")

# Limit the range displayed because distributions are heavily right-skewed
p1 = p1 + ylim(0,quantile(userVotesMelt$num, 0.75))
p1 = p1 + ggtitle("Boxplots of Number of User Votes by Vote Type
                  Limited Range")
ggsave(p1, filename="userVotesBoxplotsLim.png", height=6, width=6, units="in")


# Barchart of number of total votes for each vote category
p2 = ggplot(userVotesMelt, aes(y=num, x=vote_type)) 
p2 = p2 + geom_bar(stat="identity", fill='red')
p2 = p2 + ggtitle("Total User Votes for Each Vote Type")
p2 = p2 + xlab("Vote Type") + ylab("Number of Votes")
p2 = incAxisLabelSpace(p2)
p2
ggsave(p2, filename="userVotesBarplot.png", height=5, width=5, units="in")


# Density Plot of user average ratings
p3 = ggplot(users, aes(x=u_average_stars)) 
p3 = p3 + geom_density(na.rm=TRUE, colour="DeepSkyBlue", 
                       fill="lightblue", alpha=0.2)
p3 = p3 + ylim(0,1.2)
meanRating = mean(users$u_average_stars, na.rm=TRUE)
p3 = p3 + geom_vline(xintercept=meanRating)
p3 = p3 + geom_text(label=paste("Mean:", signif(meanRating,5)), 
                    x=4.3, y=1.2, size=3)
p3 = p3 + ggtitle("User Average Stars for Reviews")
p3 = p3 + xlab("Average Stars") + ylab('Density')
p3 = incAxisLabelSpace(p3)
ggsave(p3, filename="userAvgRatingDensity.png", width=6, height=4, units="in")


# Density of user review counts
p4 = ggplot(users, aes(x=u_review_count))
p4 = p4 + geom_density(na.rm=TRUE, colour="DeepSkyBlue",
                       fill="lightblue", alpha=0.2)
meanCnt = mean(users$u_review_count, na.rm=TRUE)
medCnt = median(users$u_review_count, na.rm=TRUE)
p4 = p4 + geom_text(x=2000, y=0.05, size=3,
                    label=paste("Mean Review Count:", 
                                signif(meanCnt, 5)))
p4 = p4 + geom_text(x=2000, y=0.04, size=3,
                    label=paste("Median Review Count:", medCnt))
p4 = p4 + ggtitle("User Review Counts")
p4 = p4 + xlab("Number of Reviews") + ylab("Density")
ggsave(p4, filename="userReviewCntDensity.png", width=6, height=4, units="in")


# Barchart: Total number of reviews in dataset vs. number of
# reviews written by users
revTotals = data.frame(cbind(c("Reviews in Dataset", "Reviews by Users"),
                             c(nrow(training), sum(users$u_review_count, na.rm=TRUE))))
names(revTotals) = c("rev_source", "num")
revTotals$num = as.numeric(as.character(revTotals$num))
p5 = ggplot(revTotals, aes(x=rev_source, y=num)) 
p5 = p5 + geom_bar(stat='identity', fill="red")
p5 = p5 + geom_text(aes(label=revTotals$num), vjust=-0.3, size=3)
p5 = p5 + ggtitle("Reviews in Data Set vs. All Reviews by Users")
p5 = p5 + xlab("Review Source") + ylab("Number of Reviews")
p5 = incAxisLabelSpace(p5)
ggsave(p5, filename="reviewCountsBar.png", width=5, height=5, units="in")


# Barchart: number of users in data set vs number of users 
usrTotals = data.frame(cbind(c("Users in Review Data", "Users in User Data"),
                             c(nrow(users), sum(!is.na(users$u_average_stars)))))
names(usrTotals) = c("usr_source", "num")
usrTotals$num = as.numeric(as.character(usrTotals$num))
p6 = ggplot(usrTotals, aes(x=usr_source, y=num)) 
p6 = p6 + geom_bar(stat='identity', fill="red")
p6 = p6 + geom_text(aes(label=usrTotals$num), vjust=-0.3, size=3)
p6 = p6 + ggtitle("Users in Review Data vs. Users in User Data")
p6 = p6 + xlab("User Source") + ylab("Number of Reviews")
p6 = incAxisLabelSpace(p6)
ggsave(p6, filename="userCountsBar.png", width=5, height=5, units="in")


