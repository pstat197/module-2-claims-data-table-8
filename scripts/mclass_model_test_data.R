## multiclass_model_test
library(tidyverse)
library(text2vec)
library(glmnet)

source("scripts/preprocessing.R")

# Load saved model and text pipeline
load("results/mclass_results/mclass_primary_model.RData")

# Load and preprocess test data
load("data/claims-test.RData")

clean_test <- claims_test %>%
  parse_data() %>%
  select(.id, text_clean)

it_test  <- itoken(clean_test$text_clean, ids = clean_test$.id, progressbar = FALSE)
dtm_test <- create_dtm(it_test, text_pipeline$vectorizer)
X_test   <- text_pipeline$tfidf$transform(dtm_test)

# Predict multiclass labels
mclass_pred <- predict(
  mclass_model,
  X_test,
  s    = "lambda.min",
  type = "class"
)[, 1]

pred_df_test <- tibble(
  .id         = clean_test$.id,
  mclass.pred = as.character(mclass_pred)
)