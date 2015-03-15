library(ggplot2)

incAxisLabelSpace = function(plot){
  # Input: ggplot object
  # Output: same ggplot object with increased space between axis
  #         and axis label
  new_plot = plot + theme(axis.title.x = element_text(vjust=-.1),
                          axis.title.y = element_text(vjust=1),
                          plot.title = element_text(vjust=.5))
  return(new_plot)
}

training = read.csv("yelp_training.csv")

r1 = ggplot(training, aes(x=as.factor(r_stars))) + geom_bar(fill='red')
r1 = r1 + ggtitle("Number of Reviews by Star Rating")
r1 = r1 + xlab("Star Rating") + ylab("Number of Reviews")
r1 = incAxisLabelSpace(r1)
r1
ggsave(r1, filename="Plots/reviewsByStarsBarplot.png", width=5, height=5, units="in")
