
# bclass_model_test.R
# Use fitted binary glmnet model to generate predictions on claims-test.RData

library(tidyverse)
library(text2vec)
library(glmnet)

source("scripts/preprocessing.R")

# Load saved binary classification model 
load("results/binaryclass_results/bclass_primary_model.RData")


# Load and preprocess test data
load("data/claims-test.RData")  

claims_test_clean <- claims_test %>%
  parse_data() %>%
  select(.id, text_clean)

# Build TF–IDF features for test set

it_test <- itoken(
  claims_test_clean$text_clean,
  ids         = claims_test_clean$.id,
  progressbar = FALSE
)

dtm_test <- create_dtm(it_test, model_bundle$vectorizer)
X_test   <- model_bundle$tfidf_model$transform(dtm_test)


# Predict binary labels
bclass_raw <- predict(
  model_bundle$glmnet_model,
  X_test,
  s    = "lambda.min",
  type = "class"
)[, 1]

# Map 0/1 to actual class labels
bclass_pred <- ifelse(
  as.numeric(as.character(bclass_raw)) == 1,
  "Relevant claim content",
  "N/A: No relevant content."
)

# Build prediction data frame 

pred_df_binary <- tibble(
  .id         = claims_test_clean$.id,
  bclass.pred = as.character(bclass_pred)
)