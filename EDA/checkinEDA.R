library(ggplot2)

incAxisLabelSpace = function(plot){
  # Input: ggplot object
  # Output: same ggplot object with increased space between axis
  #         and axis label
  new_plot = plot + theme(axis.title.x = element_text(vjust=-.1),
                          axis.title.y = element_text(vjust=1),
                          plot.title = element_text(vjust=.76))
  return(new_plot)
}

## Import data
#training = read.csv("yelp_training.csv")

# Number of checkins per business
checkins = subset(training, select=c(business_id, b_sum_checkins))
checkins = checkins[!duplicated(checkins),]
checkins = checkins[order(checkins$b_sum_checkins, decreasing=TRUE),]


# Histogram of number of checkins per business
c1 = ggplot(checkins[,], aes(x=b_sum_checkins)) 
c1 = c1 + geom_histogram(fill="LimeGreen")
c1 = c1 + ggtitle("Number of Checkins Per Business")
c1 = c1 + xlab("Number of Checkins") + ylab("Frequency")
c1
ggsave(c1, filename="Plots/checkinsHistogram.png", width=5, height=5, units="in")

# proportion of businesses with less than 1000 checkins
sum(checkins$b_sum_checkins<1000, na.rm=TRUE)/nrow(checkins)

# Limit histogram to less than 1000 checkins
c2 = ggplot(checkins[checkins$b_sum_checkins<1000,], aes(x=b_sum_checkins)) 
c2 = c2 + geom_histogram(fill="LimeGreen")
c2 = c2 + ggtitle("Number of Checkins Per Business
                  (Less than 1000)")
c2 = c2 + xlab("Number of Checkins") + ylab("Frequency")
c2
ggsave(c2, filename="Plots/checkinsHistogramLim.png", width=5, height=5, units="in")


# Barchart: how many business has any checkins
checkins$has_checkins = !is.na(checkins$b_sum_checkins)
c3 = ggplot(checkins, aes(x=has_checkins))
c3 = c3 + geom_bar(stat="bin", fill="LimeGreen") 
c3 = c3 + ggtitle("Businesses With and Without Checkins")
c3 = c3 + xlab("Business Has Checkins") + ylab("Frequency")
c3 = incAxisLabelSpace(c3)
c3
ggsave(c3, filename="Plots/checkinsBarchart.png", width=5, height=5, units="in")


# Plot of number of reviews against number of checkins (in data set)
checkins$b_sum_checkins[is.na(checkins$b_sum_checkins)] = 0
reviewsPerBus = aggregate(. ~ business_id, training[,1:2], FUN=length)
names(reviewsPerBus)[2] = "num_reviews"
checkins = merge(checkins, reviewsPerBus)
max(checkins$b_sum_checkins)

c4 = ggplot(checkins[checkins$b_sum_checkins<20000,], 
            aes(x=b_sum_checkins, y=num_reviews))
c4 = c4 + geom_point()
c4 = c4 + ggtitle("Business Check-Ins vs. Reviews (in data set)")
c4 = c4 + xlab("Number of Check-Ins") + ylab("Number of Reviews")
corr = cor(checkins$b_sum_checkins, checkins$num_reviews)
c4 = c4 + geom_text(x=11000, y=800, size=3,
                    label=paste("Correlation:", signif(corr, 4)))
ggsave(c4, filename="Plots/checkinsVsReviewsScatter.png", 
       width=5, height=5, units="in")

