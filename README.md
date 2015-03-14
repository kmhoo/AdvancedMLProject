# Yelp Recommendation Model
This project contains all the files used to create the best model to predict user star ratings of a certain business.

# The Data:
The 2013 RecSys Yelp Kaggle competition is where you will find the initial data. Only the training data was used during this project. There are four files you should find:
- yelp_training_set_business.json
- yelp_training_set_user.json
- yelp_training_set_review.json
- yelp_train_set_checkin.json

# First Step
data_cleaning.py: aggregates all of the files, fixes any misspelled cities, creates dummy variables for the categories, and splits the data into a training and testing set
- the most important part of this script, is the removable of data leakage: excluding the test set information from the overall columns in the training set

# Data Preprocessing
data_processing.py and/or data_processing_update.py: each of these scripts will run our different models 
- data_processing: keeps only the columns applicable to predicting reviews and imputes all missing values with the mean of the column; returns a file to be fed into the models
- data_processing_update: includes all from data_processing as well as more efficient ways of filling in missing data and more removable of data leakage; returns a new file for the models

# Exploratory Data Analysis
<feature>EDA.R: will look at specific variables in the data set as well as their correlations with other variables (including the target, review stars

text_analysis: generates all tokens, unigrams, bigrams and trigrams of the reviews

# Feature Engineering
applyNewFeatures.py: script to implement each of the feature engineering techniques into our models
- featureEngineering_CategoryReduction.py: runs Principle Component Analysis on the business categories to decrease the number of categories as features
- featureEngineering_UserClustering: the R script will generate a Scree Plot to determine how many clusters we should use; the py script will implement KMeans on the data, and create dummy variables for every cluster
- featureEngineering_textFeatures.py: creates new features of our bigrams and trigrams based on TF-IDF scores (limited the bigrams/trigrams to only those that appear in more than 1000 documents)
- featureEngineering_CollaborativeFiltering: the R script; the py script
missingUserUpdate.R: updates information about users with missing values to be used (instead of replacing them with the mean)

# Models
model_<MODEL>.py: runs each algorithm with 5 fold cross validation and any hyper-parameter tuning (using file from data_processing)

model2_<MODEL>.py: runs the second iteration of the top algorithms with feature engineering included (using file from data_processing_update)

finalModel_RandomForest.py: runs RandomForest with User Clusters and Category Principle Components; determines best hyper-parameters and uses best model to run again actual test data

# Plots
These are all the plots that the EDA scripts will produce

