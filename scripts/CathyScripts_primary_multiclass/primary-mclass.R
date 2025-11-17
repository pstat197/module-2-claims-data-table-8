## Train a primary multiclass glmnet model and generate predictions

library(tidyverse)
library(text2vec)
library(glmnet)
library(tidymodels)

source("scripts/preprocessing.R")

set.seed(110122)

# output directory
out_dir <- "results/mclass_results"

# make sure folders exist
if (!dir.exists("results")) dir.create("results", recursive = TRUE)
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# Load raw labeled training data and preprocess

load("data/claims-raw.RData") 

claims_clean <- claims_raw %>%
  parse_data() %>%                    
  select(.id, text_clean, mclass) %>%
  mutate(mclass = factor(mclass))

cat("Training size:", nrow(claims_clean), "pages\n")
cat("Multiclass labels:", paste(levels(claims_clean$mclass), collapse = ", "), "\n")

# Optionally save cleaned training data
if (!dir.exists("data")) dir.create("data", recursive = TRUE)
save(claims_clean, file = "data/claims-clean-mclass.RData")

# 2. Build TF–IDF features for training

it_train <- itoken(
  claims_clean$text_clean,
  ids         = claims_clean$.id,
  progressbar = FALSE
)

vocab <- create_vocabulary(
  it_train,
  ngram = c(1L, 2L)        # unigrams + bigrams
) %>%
  prune_vocabulary(
    term_count_min = 5L
  )

vectorizer <- vocab_vectorizer(vocab)
dtm_train  <- create_dtm(it_train, vectorizer)

tfidf <- TfIdf$new()
X_train <- tfidf$fit_transform(dtm_train)

y_multi <- claims_clean$mclass

cat("TF–IDF matrix:", dim(X_train)[1], "docs x", dim(X_train)[2], "terms\n")

# Fit multinomial glmnet (primary multiclass model)

cv_multi <- cv.glmnet(
  x            = X_train,
  y            = y_multi,
  family       = "multinomial",
  type.measure = "class",
  nfolds       = 5
)

# Cross-validated in-sample accuracy
pred_multi_cv <- predict(
  cv_multi,
  X_train,
  s    = "lambda.min",
  type = "class"
)[, 1]

acc_multi_cv <- mean(pred_multi_cv == y_multi)
cat("Multiclass CV accuracy (lambda.min):", round(acc_multi_cv, 3), "\n")

# Pack deployable objects
mclass_model  <- cv_multi
text_pipeline <- list(
  vocab      = vocab,
  vectorizer = vectorizer,
  tfidf      = tfidf
)

# Save model
save(
  mclass_model,
  text_pipeline,
  file = file.path(out_dir, "mclass_primary_model.RData")
)

cat("Saved multiclass model + text pipeline to", 
    file.path(out_dir, "mclass_primary_model.RData"), "\n")

# Load TEST data, preprocess, and build TF–IDF features

load("data/claims-test.RData")

clean_test <- claims_test %>%
  parse_data() %>% 
  select(.id, text_clean)

cat("Test size after parsing:", nrow(clean_test), "pages\n")

it_test <- itoken(
  clean_test$text_clean,
  ids         = clean_test$.id,
  progressbar = FALSE
)

dtm_test <- create_dtm(it_test, text_pipeline$vectorizer)
X_test   <- text_pipeline$tfidf$transform(dtm_test)

# Predict multiclass labels on test set
mclass_pred <- predict(
  mclass_model,
  X_test,
  s    = "lambda.min",
  type = "class"
)[, 1]

# Construct pred_df with required columns
pred_df <- tibble(
  .id         = clean_test$.id,
  mclass.pred = as.character(mclass_pred)
)

## Change groupN to your actual group number
save(pred_df, file = file.path(out_dir, "preds-groupN.RData"))

cat("Saved predictions to", file.path(out_dir, "preds-groupN.RData"), "\n")
