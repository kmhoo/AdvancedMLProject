# Packages
#library(ggplot2)
library(RgoogleMaps)
library(RColorBrewer)
library(MASS)

# Import data
#setwd("/Users/griffin/Google Drive/USF/Spring1/AdvancedMachineLearning/AdvancedMLProject")
training <- read.csv("yelp_training.csv")


## Maps of data, not zoomed

# Business locations
businessLoc = subset(training, !duplicated(training$business_id),
                     select=c(business_id, b_longitude, b_latitude))

# Map of businesses
map = MapBackground(businessLoc$b_latitude, businessLoc$b_longitude)
redAlpha = rgb(255, 0, 0, as.integer(255*0.15), maxColorValue=255)
png("mapBusinesses.png", height=1200, width=1200)
PlotOnStaticMap(map, businessLoc$b_latitude, businessLoc$b_longitude, 
                pch=19, cex=1.5, col=redAlpha)
dev.off()

# # Map of businesses as heatmap
# cols <- rev(colorRampPalette(brewer.pal(8, 'RdYlGn'))(100))
# alpha <- seq.int(0.5, 0.95, length.out=100)
# alpha <- exp(alpha^6-1)
# cols2 <- AddAlpha(cols, alpha)
# k2 <- kde2d(training$b_longitude, training$b_latitude, n=500)
# PlotOnStaticMap(map)
# image(k2, col=cols2, add=TRUE)

# Map of reviews
blueAlpha = rgb(0, 0, 255, as.integer(255*0.025), maxColorValue=255)
png("mapReviews.png", height=1200, width=1200)
PlotOnStaticMap(map, training$b_latitude, training$b_longitude, 
                pch=19, cex=1.5, col=blueAlpha)
dev.off()

# Checkins data - locations of business that have checkins
checkins = subset(training, !is.na(training$b_sum_checkins),
                  select=c(business_id, b_latitude, b_longitude, b_sum_checkins))

# Map of checkins
purpAlpha = rgb(150, 0, 205, 
                as.integer(checkins$b_sum_checkins/max(checkins$b_sum_checkins)*255), 
                maxColorValue=255)
png("mapCheckins.png", height=1200, width=1200)
PlotOnStaticMap(map, checkins$b_latitude, checkins$b_longitude, 
                pch=19, cex=1.5, col=purpAlpha)
dev.off()


## Zoomed-In Maps - limit lat/lon coordinates

# Zoomed-In Map of businesses
zoomBusiness = businessLoc[33.2<businessLoc$b_latitude & businessLoc$b_latitude<33.8
                        & -112.4<businessLoc$b_longitude & businessLoc$b_longitude< -111.75,]
png("mapBusinessesZoom.png", height=1200, width=1200)
PlotOnStaticMap(lat=zoomBusiness$b_latitude, lon=zoomBusiness$b_longitude, 
                pch=19, cex=3, col=redAlpha, size=c(640,640))
dev.off()

# Zoomed-In Map of reviews
zoomReviews = subset(training, 
                     33.2<training$b_latitude & training$b_latitude<33.8
                     & -112.4<training$b_longitude & training$b_longitude< -111.75,
                     select=c(r_review_id, b_latitude, b_longitude))
png("mapReviewsZoom.png", height=1200, width=1200)
PlotOnStaticMap(lat=zoomReviews$b_latitude, lon=zoomReviews$b_longitude, 
                pch=19, cex=3, col=blueAlpha, size=c(640,640))
dev.off()

# Zoomed-In Map of checkins
zoomCheckins = checkins[33.2<checkins$b_latitude & checkins$b_latitude<33.8
                        & -112.4<checkins$b_longitude & checkins$b_longitude< -111.75,]
purpAlphaZoom = rgb(150, 0, 205, 
                as.integer(zoomCheckins$b_sum_checkins/max(zoomCheckins$b_sum_checkins)*255), 
                maxColorValue=255)
png("mapCheckinsZoom.png", height=1200, width=1200)
PlotOnStaticMap(lat=zoomCheckins$b_latitude, lon=zoomCheckins$b_longitude, 
                pch=19, cex=3, col=purpAlphaZoom, size=c(640,640))
dev.off()