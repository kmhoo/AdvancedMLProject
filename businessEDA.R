setwd('/Users/kaileyhoo/Documents/MSAN/Module 3/MSAN630/Project/AdvancedMLProject/')
library(ggplot2)

#import training data
training <- read.csv('yelp_training.csv')

#subset for only business columns
business <- training[,grepl("b_|business_id", names(training))]

# number of reviews per business in dataset
reviewsPerBus <- aggregate(. ~ business_id, business[,1:2], FUN=length)
names(reviewsPerBus)[2] <- 'num_of_reviews'
b1 <- ggplot(reviewsPerBus, aes(x=num_of_reviews))
b1 <- b1 + geom_density(colour="purple4", fill="purple2", alpha=0.2)
b1 <- b1 + ggtitle("Number of Reviews Per Business")
b1 <- b1 + xlab("Number of Reviews") + ylab("Density")
b1

# calculate median and mean
medRev <- median(reviewsPerBus$num_of_reviews)
meanRev <- mean(reviewsPerBus$num_of_reviews)

# subset the reviews to exclude large outliers
reviewsPerBusSub <- reviewsPerBus[reviewsPerBus$num_of_reviews<50, ]
b2 <- ggplot(reviewsPerBusSub, aes(x=num_of_reviews))
b2 <- b2 + geom_density(colour="purple4", fill="purple2", alpha=0.2)
b2 <- b2 + ggtitle("Number of Reviews Per Business with Less Than 50 Reviews")
b2 <- b2 + xlab("Number of Reviews") + ylab("Density")
b2

# subset to get one line per business
uniqueBusiness <- business[!duplicated(business$business_id),]

# number of reviews per business (review count)
b3 <- ggplot(uniqueBusiness, aes(x=b_review_count))
b3 <- b3 + geom_histogram(colour = "white", fill='purple4', binwidth=50)
b3 <- b3 + xlim(0,850)
b3 <- b3 + ggtitle("Number Reviews per Business")
b3 <- b3 + xlab("Number of Reviews") + ylab("Number of Businesses")
b3

# number of stars by business
b4 <- ggplot(uniqueBusiness, aes(x=b_stars))
b4 <- b4 + geom_histogram(colour = "white", fill='purple4', binwidth = 0.5)
b4 <- b4 + xlim(1,5.5)
b4 <- b4 + ggtitle("Number Businesses per Star Rating")
b4 <- b4 + xlab("Star Ratings") + ylab("Number of Businesses")
b4

# density plot of stars
b5 <- ggplot(uniqueBusiness, aes(x=b_stars))
b5 <- b5 + geom_density(colour="purple4", fill="purple2", alpha=0.2)
b5 <- b5 + ggtitle("Number Businesses per Star Rating")
b5 <- b5 + xlab("Star Ratings") + ylab("Density")
b5

# number of businesses in each city
busPerCity <- aggregate(. ~ b_city, uniqueBusiness[,c(1,3)], FUN=length)
names(busPerCity)[2] <- "num_of_businesses"
busPerCity <- busPerCity[order(-busPerCity$num_of_businesses),]
b6 <- ggplot(busPerCity, aes(y=num_of_businesses, x=b_city)) 
b6 <- b6 + geom_bar(stat="identity", fill='purple4')
b6 <- b6 + ggtitle("Number Businesses per City")
b6 <- b6 + xlab("Cities") + ylab("Number of Businesses")
b6 <- b6 + theme(axis.text.x=element_text(angle = 45))
b6

# calculate the mean and median
medCity <- median(busPerCity$num_of_businesses)
meanCity <- mean(busPerCity$num_of_businesses)

# top 15 cities with most businesses
busPerCity15 <- busPerCity[1:15,]
b7 <- ggplot(busPerCity15, aes(y=num_of_businesses, x=b_city)) 
b7 <- b7 + geom_bar(stat="identity", fill='purple4')
b7 <- b7 + ggtitle("Number Businesses per City (Top 15)")
b7 <- b7 + xlab("Cities") + ylab("Number of Businesses")
b7 <- b7 + theme(axis.text.x=element_text(angle = 45))
b7

# subset to just category columns
buscategories <- uniqueBusiness[,-c(2,4,5,6,7,8,519)]

