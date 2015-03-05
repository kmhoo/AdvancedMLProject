library(ggplot2)
library(zoo)

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

training$r_date2 = strptime(as.character(training$r_date), "%Y-%m-%d")
training$r_monthYear = as.Date(as.yearmon(training$r_date2))

# Plot of reviews over time
revPerDay = aggregate(r_review_id ~ r_date, data=training, length)
names(revPerDay)[names(revPerDay)=="r_review_id"] = "num"
revPerDay$r_date2 = strptime(as.character(revPerDay$r_date), "%Y-%m-%d")

t1 = ggplot(revPerDay, aes(x=r_date2, y=num)) + geom_line()
t1 = t1 + ggtitle("Number of Reviews per Day")
t1 = t1 + xlab("Time") + ylab("Number of Reviews")
t1 = incAxisLabelSpace(t1)
t1
ggsave(t1, filename="Plots/reviewsOverTimeDays.png", width=6, height=4, units="in")

revPerMonth = aggregate(r_review_id ~ r_monthYear, data=training, length)
names(revPerMonth)[names(revPerMonth)=="r_review_id"] = "num"
t1a = ggplot(revPerMonth[1:(nrow(revPerMonth)-1),], 
            aes(x=r_monthYear, y=num)) + geom_line()
t1a = t1a + ggtitle("Number of Reviews per Month")
t1a = t1a + xlab("Time") + ylab("Number of Reviews")
t1a = incAxisLabelSpace(t1a)
t1a
ggsave(t1a, filename="Plots/reviewsOverTimeMonths.png", width=6, height=4, units="in")

# Plot of number of users posting reviews over time

usersPerMonth = aggregate(user_id ~ r_monthYear, data=training, 
                          function(x) length(unique(x)))
names(usersPerMonth)[names(usersPerMonth)=="user_id"] = "num"

t2 = ggplot(usersPerMonth[1:(nrow(usersPerMonth)-1),], 
            aes(x=r_monthYear, y=num)) + geom_line()
t2 = t2 + ggtitle("Number of Users Posting per Month")
t2 = t2 + xlab("Time") + ylab("Number of Users")
t2 = incAxisLabelSpace(t2)
t2
ggsave(t2, filename="Plots/usersOverTimeMonths.png", width=6, height=4, units="in")


# Plot on same plot
byMonth = rbind(cbind(revPerMonth[1:(nrow(revPerMonth)-1),], type="reviews"),
                cbind(usersPerMonth[1:(nrow(usersPerMonth)-1),], type="users"))
t3 = ggplot(byMonth, aes(x=r_monthYear, y=num, col=type)) + geom_line()
t3 = t3 + theme(legend.title=element_blank(), legend.position="top")
t3 = t3 + ggtitle("Number of Reviews and Users Per Month")
t3 = t3 + xlab("Time") + ylab("Number")
t3 = incAxisLabelSpace(t3)
t3
ggsave(t3, filename="Plots/revUsersOverTime.png", width=5, height=5, units="in")
