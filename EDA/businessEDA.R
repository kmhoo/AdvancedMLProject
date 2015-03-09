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


#import training data
training <- read.csv('../yelp_training.csv')

#subset for only business columns
business <- training[,grepl("b_|business_id", names(training))]

# number of reviews per business in dataset
reviewsPerBus <- aggregate(. ~ business_id, business[,1:2], FUN=length)
names(reviewsPerBus)[2] <- 'num_of_reviews'
b1 <- ggplot(reviewsPerBus, aes(x=num_of_reviews))
b1 <- b1 + geom_density(colour="purple4", fill="purple2", alpha=0.2)
b1 <- b1 + ggtitle("Number of Reviews Per Business")
b1 <- b1 + xlab("Number of Reviews") + ylab("Density")
b1 <- incAxisLabelSpace(b1)
b1
ggsave(b1, filename="Plots/businessTrainReviewCount.png", width=5, height=5, units="in")

# calculate median and mean
medRev <- median(reviewsPerBus$num_of_reviews)
meanRev <- mean(reviewsPerBus$num_of_reviews)

# subset the reviews to exclude large outliers
reviewsPerBusSub <- reviewsPerBus[reviewsPerBus$num_of_reviews<50, ]
b2 <- ggplot(reviewsPerBusSub, aes(x=num_of_reviews))
b2 <- b2 + geom_density(colour="purple4", fill="purple2", alpha=0.2)
b2 <- b2 + ggtitle("Number of Reviews Per Business with Less Than 50 Reviews")
b2 <- b2 + xlab("Number of Reviews") + ylab("Density")
b2 <- incAxisLabelSpace(b2)
b2
ggsave(b2, filename="Plots/businessTrainReviewCountLess50.png", width=7, height=5, units="in")

# subset to get one line per business
uniqueBusiness <- business[!duplicated(business$business_id),]

# number of reviews per business (review count)
b3 <- ggplot(uniqueBusiness, aes(x=b_review_count))
b3 <- b3 + geom_histogram(colour = "white", fill='purple4', binwidth=50)
b3 <- b3 + xlim(0,850)
b3 <- b3 + ggtitle("Number Reviews per Business")
b3 <- b3 + xlab("Number of Reviews") + ylab("Number of Businesses")
b3 <- incAxisLabelSpace(b3)
b3
ggsave(b3, filename="Plots/businessReviewCount.png", width=5, height=5, units="in")

# number of stars by business
b4 <- ggplot(uniqueBusiness, aes(x=b_stars))
b4 <- b4 + geom_histogram(colour = "white", fill='purple4', binwidth = 0.5)
b4 <- b4 + xlim(1,5.5)
b4 <- b4 + ggtitle("Number Businesses per Star Rating")
b4 <- b4 + xlab("Star Ratings") + ylab("Number of Businesses")
b4 <- incAxisLabelSpace(b4)
b4
ggsave(b4, filename="Plots/businessStarRatings.png", width=5, height=5, units="in")

# density plot of stars
b5 <- ggplot(uniqueBusiness, aes(x=b_stars))
b5 <- b5 + geom_density(colour="purple4", fill="purple2", alpha=0.2)
b5 <- b5 + ggtitle("Number Businesses per Star Rating")
b5 <- b5 + xlab("Star Ratings") + ylab("Density")
b5 <- incAxisLabelSpace(b5)
b5
ggsave(b5, filename="Plots/businessStarRatingsDensity.png", width=7, height=5, units="in")

# number of businesses in each city
busPerCity <- aggregate(. ~ b_city, uniqueBusiness[,c(1,3)], FUN=length)
names(busPerCity)[2] <- "num_of_businesses"
busPerCity <- busPerCity[order(-busPerCity$num_of_businesses),]
b6 <- ggplot(busPerCity, aes(y=num_of_businesses, x=b_city)) 
b6 <- b6 + geom_bar(stat="identity", fill='purple4')
b6 <- b6 + ggtitle("Number Businesses per City")
b6 <- b6 + xlab("Cities") + ylab("Number of Businesses")
b6 <- b6 + theme(axis.text.x=element_text(angle = 45, hjust=1))
b6 <- incAxisLabelSpace(b6)
b6
ggsave(b6, filename="Plots/businessCities.png", width=7, height=5, units="in")

# calculate the mean and median
medCity <- median(busPerCity$num_of_businesses)
meanCity <- mean(busPerCity$num_of_businesses)

# top 15 cities with most businesses
busPerCity15 <- busPerCity[1:15,]
b7 <- ggplot(busPerCity15, aes(y=num_of_businesses, x=b_city)) 
b7 <- b7 + geom_bar(stat="identity", fill='purple4')
b7 <- b7 + ggtitle("Number Businesses per City (Top 15)")
b7 <- b7 + xlab("Cities") + ylab("Number of Businesses")
b7 <- b7 + theme(axis.text.x=element_text(angle = 45, hjust=1))
b7 <- incAxisLabelSpace(b7)
b7
ggsave(b7, filename="Plots/businessCitiesTop15.png", width=7, height=5, units="in")

# subset to just category columns
buscategories <- uniqueBusiness[,-c(2,4,5,6,7,8,519)]

# sum each of the categories
justcategories <- buscategories[,-c(1,2,3,4,5)]
catSum <- colSums(justcategories)
catSumMelt <- melt(catSum)
catSumMelt$category = row.names(catSumMelt)
row.names(catSumMelt) = NULL
names(catSumMelt) <- c('num_of_businesses', 'category')
catSumMelt <- catSumMelt[order(-catSumMelt$num_of_businesses),]
catSumMelt$category <- gsub('b_categories_', "", catSumMelt$category)

# plot the density of businesses per category
b8 <- ggplot(catSumMelt, aes(x=num_of_businesses))
b8 <- b8 + geom_density(colour="purple4", fill="purple2", alpha=0.2)
b8 <- b8 + ggtitle("Number Businesses for Each Category")
b8 <- b8 + xlab("Number of Businesses") + ylab("Density")
b8 <- incAxisLabelSpace(b8)
b8
ggsave(b8, filename="Plots/businessCatDensity.png", width=7, height=5, units="in")

# plot each category and the number of businesses
b9 <- ggplot(catSumMelt, aes(y=num_of_businesses, x=category)) 
b9 <- b9 + geom_bar(stat="identity", fill='purple4')
b9 <- b9 + ggtitle("Number Businesses per Category")
b9 <- b9 + xlab("Categories") + ylab("Number of Businesses")
b9 <- b9 + theme(axis.text.x=element_text(angle = 45, hjust=1))
b9 <- incAxisLabelSpace(b9)
b9
ggsave(b9, filename="Plots/businessCat.png", width=15, height=5, units="in")

# mean and median on businesses per category
meanCat <- mean(catSumMelt$num_of_businesses)
medianCat <- median(catSumMelt$num_of_businesses)

# top 50 categories with the most businesses
catSum50 <- catSumMelt[1:50,]
b10 <- ggplot(catSum50, aes(y=num_of_businesses, x=category)) 
b10 <- b10 + geom_bar(stat="identity", fill='purple4')
b10 <- b10 + ggtitle("Number Businesses per Category (Top 50)")
b10 <- b10 + xlab("Categories") + ylab("Number of Businesses")
b10 <- b10 + theme(axis.text.x=element_text(angle = 45, hjust=1))
b10 <- incAxisLabelSpace(b10)
b10
ggsave(b10, filename="Plots/businessCatTop50.png", width=7, height=5, units="in")

# top 10 categories with the most businesses
catSum10 <- catSumMelt[1:10,]
b11 <- ggplot(catSum10, aes(y=num_of_businesses, x=category)) 
b11 <- b11 + geom_bar(stat="identity", fill='purple4')
b11 <- b11 + ggtitle("Number Businesses per Category (Top 10)")
b11 <- b11 + xlab("Categories") + ylab("Number of Businesses")
b11 <- b11 + theme(axis.text.x=element_text(angle = 45, hjust=1))
b11 <- incAxisLabelSpace(b11)
b11
ggsave(b11, filename="Plots/businessCatTop10.png", width=7, height=5, units="in")

# number of categories listed for each business
categoriesbybus <- buscategories[,-c(2,3,4,5)]
busByCategory <- rowSums(categoriesbybus[2:508])
numCatByBus <- cbind(categoriesbybus, busByCategory)
numCatByBus <- numCatByBus[,c(1,509)]
names(numCatByBus)[2] <- "num_of_categories"
b12 <- ggplot(numCatByBus, aes(x=num_of_categories))
b12 <- b12 + geom_density(colour="purple4", fill="purple2", alpha=0.2)
b12 <- b12 + ggtitle("Number Categories for Each Business")
b12 <- b12 + xlab("Number of Categories") + ylab("Density")
b12 <- incAxisLabelSpace(b12)
b12
ggsave(b12, filename="Plots/catBusinessCountDensity.png", width=7, height=5, units="in")

# number of categories listed for each business (bar chart)
b13 <- ggplot(numCatByBus, aes(x=num_of_categories))
b13 <- b13 + geom_histogram(colour='white', fill='purple4', binwidth=1)
b13 <- b13 + ggtitle("Number Categories for Each Business")
b13 <- b13 + xlab("Number of Categories") + ylab("Density")
b13 <- incAxisLabelSpace(b13)
b13
ggsave(b13, filename="Plots/catBusinessCountBar.png", width=7, height=5, units="in")

# number of businesses open or closed
busOpenSubset <- uniqueBusiness[,c('business_id', 'b_open')]
openCloseBusiness <- aggregate(.~b_open, busOpenSubset, FUN=length)
names(openCloseBusiness)[2] <- "num_of_businesses"
b14 <- ggplot(openCloseBusiness, aes(y=num_of_businesses, x=b_open)) 
b14 <- b14 + geom_bar(stat="identity", fill='purple4')
b14 <- b14 + ggtitle("Number Businesses Open or Close")
b14 <- b14 + xlab("Open = True, Closed = False") + ylab("Number of Businesses")
b14 <- b14 + theme(axis.text.x=element_text(angle = 45, hjust=1))
b14 <- incAxisLabelSpace(b14)
b14
ggsave(b14, filename="Plots/openBusinesses.png", width=5, height=5, units="in")

# subset data for open or closed businesses for separate analysis
openBusiness <- business[business$b_open=="True",]
closeBusiness <- business[business$b_open=="False",]

# open business star ratings
b15 <- ggplot(openBusiness, aes(x=b_stars))
b15 <- b15 + geom_histogram(colour = "white", fill='purple4', binwidth = 0.5)
b15 <- b15 + xlim(1,5.5)
b15 <- b15 + ggtitle("Number Businesses per Star Rating")
b15 <- b15 + xlab("Star Ratings") + ylab("Number of Businesses")
b15 <- incAxisLabelSpace(b15)
b15
ggsave(b15, filename="Plots/openBusinessStarRatings.png", width=7, height=5, units="in")

# close business star ratings
b16 <- ggplot(closeBusiness, aes(x=b_stars))
b16 <- b16 + geom_histogram(colour = "white", fill='purple4', binwidth = 0.5)
b16 <- b16 + xlim(1,5.5)
b16 <- b16 + ggtitle("Number Businesses per Star Rating")
b16 <- b16 + xlab("Star Ratings") + ylab("Number of Businesses")
b16 <- incAxisLabelSpace(b16)
b16
ggsave(b16, filename="Plots/closeBusinessStarRatings.png", width=7, height=5, units="in")

# Business star rating (b_stars) across all reviews 
# vs. avg star rating in dataset
busAvgStars = aggregate(r_stars ~ business_id, data=training, mean, na.rm=TRUE)
names(busAvgStars)[names(busAvgStars)=="r_stars"] = "r_stars_avg"
busAvgStars = merge(busAvgStars, training[,c('business_id', 'b_stars')], all=FALSE)

b17 = ggplot(busAvgStars, aes(x=b_stars, y=r_stars_avg))
b17 = b17 + geom_point()
corr = cor(busAvgStars$b_stars, busAvgStars$r_stars_avg)
b17 = b17 + geom_text(x=1.5, y=4.5, size=3,
                    label=paste("Correlation:", signif(corr, 4)))
b17 = b17 + ggtitle("Business Average Stars
                  All Reviews vs. Reviews in Data Set")
b17 = b17 + xlab("Average Stars Across All Reviews")
b17 = b17 + ylab("Average Stars for Reviews in Data Set")
b17 = incAxisLabelSpace(b17)
ggsave(b17, filename="Plots/busAvgStarsScatter.png", width=5, height=5, units="in")

