# Predicting testing times of Mercedes-Benz automobiles

Introduction
============

This [dataset](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing) contains an anonymized set of variables, each representing
a custom feature in a Mercedes vehicle. For example, a variable could be
4WD, added air suspension, or a head-up display. The ground truth is
labeled ‘y’ and represents the time that the car needed to pass testing.
The goal is to predict that time, with *R*<sup>2</sup> as the error
metric.

Exploratory Data Analysis and Feature Engineering
=================================================

``` r
# Requirements
require(dplyr)
require(ggplot2)
require(diagrammeR)
require(Matrix)
require(xgboost)
require(gridExtra)
require(reshape2)
require(ggcorrplot)
require(hclust)
require(doParallel)
require(mlr)
require(caret)
require(parallelMap)
require(ParamHelpers)

df <- read.csv("train.csv")
```

Preprocessing
-------------

Wrap all the pre-processing steps (discussed in the EDA notebook) in a function to create the training/validation data:

``` r
# Gather the reduced dataset
cols_to_use <- append(colnames(bin.df), # The reduced binary predictor subset identified during EDA
                      values = c("X0", "X2"))
                              

run_preprocessing <- function(new_data, cols_keep = cols_to_use, x0_categories = x0_lookup, x2_categories = x2_lookup)
{
  # Keep necessary columns only
  df <- new_data[, cols_keep]
  
  # Append the new categories for X0
  df <- plyr::join(df, x0_categories, by = "X0") %>% 
    mutate(X0_new = as.factor(X0_new))
  
  # Append the new categories for X2
  df <- plyr::join(df, x2_categories, by = "X2") %>%
    mutate(X2_new = as.factor(X2_new)) 
  
  df$X2 <- as.character(df$X2)
  df$X2_new <- as.character(df$X2_new)
  df[df$X2 %in% c("ae", "m", "ai", "f"), "X2_new"] <- 1
  df[df$X2 %in% c("ak", "e", "as", "other"), "X2_new"] <- 2
  df[df$X2 %in% c("aq", "r"), "X2_new"] <- 3
  df[df$X2 == "s", "X2_new"] <- 4
  df[df$X2 == "n", "X2_new"] <- 5
  
  # Drop the non-important categorical predictors
  df <- df %>% 
    select(- c(X0, X2)) 
  
  # Add interaction terms
  df <- df %>%
    mutate(X0_189 = X0_new:as.factor(X189)) %>%
    mutate(X0_115 = X0_new:as.factor(X115)) 
  
  # One-hot
  dmy <-dummyVars(" ~ .", df) 
  
  df <- data.frame(predict(dmy, newdata = df))
  
  return(df)
}
```

Let’s create the train and validation sets:

``` r
# Reproducibility
set.seed(150)

# Clean read train/test
data <- read.csv("train.csv")

df <- run_preprocessing(data) %>% 
  mutate(y = data$y) %>% # Append response
  filter(y <= 200) # Drop the one high-response observation

# Remove outliers
df <- df[-outlrs, ]

# Get feat. importance again (hust for consistency w.r.t. column names)
importance_df <- get_feature_importance(df)$feats

# Train/test split
trainIndex <- createDataPartition(df$y, p = .75, list = F)

train.set <- df[trainIndex, ]
test.set <- df[-trainIndex, ]

X.train <- subset(train.set, select = -c(y)) %>%
  as.matrix

y.train <- train.set %>%
  select(y) %>%
  as.matrix

X.test <- subset(test.set, select = -c(y)) %>%
  as.matrix

y.test <- test.set %>%
  select(y) %>%
  as.matrix

# Read prediction set
X.predict <- run_preprocessing(read.csv("test.csv"))
```

XGBoost with varying number of features
---------------------------------------

Let’s try a couple of XGBoost models with a smaller number of features,
with respect to their importance. We’ll try a different approach in this
case, so as to avoid manual re-tuning everytime: Random Search. We need
to define a number of functions, by making use of the mlr package:

-   A function to return a dataset with the top N features (in terms of
    predictive power).
-   A function to perform random search given the reduced dataset,
    number of iterations and parameter space, using a 10-fold CV
    process.

``` r
# Function to return a dataset with a reduced number of features
reduce_feats <- function(data, no_feats, feat_list)
{
  # One-hot encoding
  dmy <-dummyVars(" ~ .", data) 
  
  # Get the top no_feats to use
  feats_to_use <- feat_list %>%
    select(Feature) %>%
    head(no_feats)
  
  # Generate reduced dataset
  new.df <- data.frame(predict(dmy, newdata = data)) %>%
    select(feats_to_use$Feature)
  
  return(new.df)
}


# Function to return the learning task given a number of features to use
define_learn_task <- function(no_features, X, y, feat_importance)
{
  # Gather the reduced dataset
  data <- reduce_feats(X, no_features, feat_importance) %>% 
    mutate(y = as.vector(y)) # Append target

  # Define target task
  ml_task <- makeRegrTask(data = data, target = "y")
  
  return(ml_task)
}


# Tune the model and return the results
tune_mdl <- function(no_feats, no_random_searches, parameters,  
                     X = X.train, y = y.train, feat_importance_list = importance_df)
{
  # Start a parallel pool
  parallelStartSocket(detectCores() - 1)
  
  # Define the learning task
  learning_task <- define_learn_task(no_feats, X, y, feat_importance_list)
  
  # Tune the model
  tuned_mdl <- tuneParams(learner = parameters$learner,
                         task = learning_task,
                         resampling = parameters$sampling_plan,
                         measures = rmse,
                         par.set = parameters$parameter_space,
                         control = makeTuneControlRandom(maxit = no_random_searches),
                         show.info = F)
  
  # Apply optimal parameters found earlier
  opt.mdl <- setHyperPars(learner = parameters$learner, par.vals = tuned_mdl$x)
  
  # Verify CV performance of the best model
  performance <- resample(learner = opt.mdl, 
                          task = learning_task, 
                          resampling = parameters$sampling_plan,
                          measures = list(rmse),
                          keep.pred = T)
  
  # Gather results
  results <- list("no_feats" = no_feats,
                  "opt_params" = tuned_mdl$x,
                  "cv.rmse" = performance$aggr,
                  "predictions" = performance$pred$data[, c("id", "truth", "response")])
  
  # Release cores
  parallelStop()
  
  return(results)
}
```

Time to run our iterations:

``` r
# Reproducibility
set.seed(135)

# Generate the necessary (constant) tuning parameters
param_list <- list(
  # Define search space
  "parameter_space" = ParamHelpers::makeParamSet( 
    makeIntegerParam("nrounds", lower = 10, upper = 200),
    makeIntegerParam("max_depth", lower = 1, upper = 10),
    makeNumericParam("eta", lower = .01, upper = .3),
    makeNumericParam("gamma", lower = 1, upper = 10),
    makeNumericParam("subsample", lower = 0.4, upper = 0.9),
    makeIntegerParam("min_child_weight",lower = 1,upper = 7),
    makeNumericParam("colsample_bytree",lower = 0.4,upper = 1)),
  # Define resampling plan
  "sampling_plan" = mlr::makeResampleDesc("CV", iters = 10),
  # Define the learner
  "learner" = mlr::makeLearner(cl = "regr.xgboost"))

#xgb_mdls <- sapply(c(2, 5, 10, 20, 30, 50, 75, 98), function(features) tune_mdl(features, no_random_searches = 150, param_list))
```

Test set performance
--------------------

Let’s train the best model over the entire training set and get a
realistic performance on the test set:

``` r
# Reproducibility
set.seed(18)

# Create Dmatrices
train_final <- reduce_feats(X.train, 5, importance_df) %>% as.matrix
test_final <- reduce_feats(X.test, 5, importance_df) %>% as.matrix

dtrain <- xgb.DMatrix(data = train_final, label = y.train)
dtest <- xgb.DMatrix(data = test_final, label = y.test)

# Make the final model for the test set predictions
xgb.final <- xgb.train(params = list(eta = 0.2979364,
                                   gamma = 8.6622699,
                                   max_depth = 2,
                                   colsample_bytree = 0.4182624,
                                   subsample = 0.6158664,
                                   min_child_weight = 5),
                       data = dtrain,
                       nrounds = 23,
                       watchlist = list(train = dtrain, test = dtest),
                       metric = list("r.sq", "rmse"),
                       verbose = 1,
                       print_every_n = 23)
```

    ## [20:53:02] WARNING: amalgamation/../src/learner.cc:573: 
    ## Parameters: { "metric" } might not be used.
    ## 
    ##   This may not be accurate due to some parameters are only used in language bindings but
    ##   passed down to XGBoost core.  Or some parameters are not used but slip through this
    ##   verification. Please open an issue if you find above cases.
    ## 
    ## 
    ## [1]  train-rmse:71.288414    test-rmse:71.117622 
    ## [23] train-rmse:8.217968 test-rmse:7.855091

The error difference between train and test sets is very small, which
means we don’t have an overfitting problem. Finally, we are getting an
RMSE of 8.36 on the test set, using only 5 features!

Submission
----------

We’ll train the model over the entire training set, and make the
submission file:

``` r
# Reproducibility
set.seed(18)

# Create Dmatrices
X <- rbind(train_final, test_final)

y <- rbind(y.train, y.test)

dtrain <- xgb.DMatrix(data = X, label = y)

xgb.final <- xgb.train(params = list(eta = 0.2979364,
                                   gamma = 8.6622699,
                                   max_depth = 2,
                                   colsample_bytree = 0.4182624,
                                   subsample = 0.6158664,
                                   min_child_weight = 5),
                     data = dtrain,
                     nrounds = 23,
                     metric = "rmse",
                     verbose = 0,
                     print_every_n = 23)
```

``` r
prediction.set <- reduce_feats(X.predict, 5, importance_df) %>% as.matrix

read.csv("test.csv") %>%
  select(ID) %>%
  mutate(y = predict(xgb.final, prediction.set)) %>%
  write.csv(., "submission.csv", row.names = F)
```

Results
-------

We’re getting a score (*R*<sup>2</sup>) of 0.53244 on the public
leaderboard and 0.51768 on the private leaderboard. Top
three solutions on the private leaderboard are 0.55550, 0.55498 and
0.55450 respectively.

On the whole, not bad for a model that needs only 5 out of the original
377 features…

