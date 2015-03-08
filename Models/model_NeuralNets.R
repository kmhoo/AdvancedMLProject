library(nnet)
library(devtools)

#Import in the data
data <- read.csv('training_init.csv')

#Apply neural nets formula
n_net <- nnet(target ~ ., data=data, size=1)
#cannot do more that size=1 because there are too many weights
# RESULTS
# initial  value 2473888.976275 
# final  value 1680836.000000 
# converged

#plot the neural net
# source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')
# plot.nnet(n_net)

#get the weights for the system
n_net$wts

# because we can't increase the size of the hidden units, we are not going to
# continue using neural nets as a model for our recommendation system
