library(ggplot2)
library(reshape2)

incAxisLabelSpace = function(plot){
  # Input: ggplot object
  # Output: same ggplot object with increased space between axis
  #         and axis label
  new_plot = plot + theme(axis.title.x = element_text(vjust=-.1),
                          axis.title.y = element_text(vjust=1),
                          plot.title = element_text(vjust=.5))
  return(new_plot)
}

# Import data
#setwd("/Users/griffin/Google Drive/USF/Spring1/AdvancedMachineLearning/AdvancedMLProject")
training <- read.csv("yelp_training.csv")

# Create a melted data frame of users votes
# Each row has the vote type (cool, funny, useful) and a number of votes
# for one particular user
userVotesMelt = melt(subset(training, 
                            select=c(u_votes_cool, u_votes_funny, u_votes_useful)))
userVotesMelt = userVotesMelt[!is.na(userVotesMelt$num),]
names(userVotesMelt) = c('vote_type', 'num')
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
p1 = p1 + ylim(0,150)
p1 = p1 + ggtitle("Boxplots of Number of User Votes by Vote Type
                  Limited Range")
ggsave(p1, filename="userVotesBoxplotsLim.png", height=6, width=6, units="in")


# Barchart of number of total votes for each vote category
p2 = ggplot(userVotesMelt, aes(y=num, x=vote_type)) 
p2 = p2 + geom_bar(stat="identity", fill='red')
p2

# Density Plot of user average ratings
