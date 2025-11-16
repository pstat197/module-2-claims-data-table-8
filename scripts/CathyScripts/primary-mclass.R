## scripts/primary-mclass.R
## Train a primary MULTICLASS glmnet model and generate predictions on claims-test.RData

library(tidyverse)
library(text2vec)
library(glmnet)
library(tidymodels)

#------------------------------------------------------------
# 0. Source preprocessing functions (your HTML -> text pipeline)
#------------------------------------------------------------

source("scripts/preprocessing.R")
# parse_data() expects a column `text_tmp` with HTML and produces `text_clean`

set.seed(110122)  # for reproducibility

# output directory for this multiclass try
out_dir <- "results/mclass-results-try"

# make sure folders exist
if (!dir.exists("results")) dir.create("results", recursive = TRUE)
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

#------------------------------------------------------------
# 1. Load raw labeled training data and preprocess
#------------------------------------------------------------

load("data/claims-raw.RData")   # should create `claims_raw`

claims_clean <- claims_raw %>%
  parse_data() %>%                    # uses your parse_fn internally
  select(.id, text_clean, mclass) %>%
  mutate(mclass = factor(mclass))

cat("Training size:", nrow(claims_clean), "pages\n")
cat("Multiclass labels:", paste(levels(claims_clean$mclass), collapse = ", "), "\n")

# Optionally save cleaned training data (still in data/)
if (!dir.exists("data")) dir.create("data", recursive = TRUE)
save(claims_clean, file = "data/claims-clean-mclass.RData")

#------------------------------------------------------------
# 2. Build TF–IDF features for training (unigrams + bigrams)
#------------------------------------------------------------

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
    term_count_min = 5L    # drop rare terms
  )

vectorizer <- vocab_vectorizer(vocab)
dtm_train  <- create_dtm(it_train, vectorizer)

tfidf <- TfIdf$new()
X_train <- tfidf$fit_transform(dtm_train)

y_multi <- claims_clean$mclass

cat("TF–IDF matrix:", dim(X_train)[1], "docs x", dim(X_train)[2], "terms\n")

#------------------------------------------------------------
# 3. Fit multinomial glmnet (primary multiclass model)
#------------------------------------------------------------

cv_multi <- cv.glmnet(
  x            = X_train,
  y            = y_multi,
  family       = "multinomial",
  type.measure = "class",
  nfolds       = 5
)

# Cross-validated in-sample accuracy (for reporting in writeup)
pred_multi_cv <- predict(
  cv_multi,
  X_train,
  s    = "lambda.min",
  type = "class"
)[, 1]

acc_multi_cv <- mean(pred_multi_cv == y_multi)
cat("Multiclass CV accuracy (lambda.min):", round(acc_multi_cv, 3), "\n")

# Pack deployable objects (model + text pipeline)
mclass_model  <- cv_multi
text_pipeline <- list(
  vocab      = vocab,
  vectorizer = vectorizer,
  tfidf      = tfidf
)

# SAVE MODEL UNDER results/mclass-results-try
save(
  mclass_model,
  text_pipeline,
  file = file.path(out_dir, "mclass_primary_model.RData")
)

cat("Saved multiclass model + text pipeline to", 
    file.path(out_dir, "mclass_primary_model.RData"), "\n")

#------------------------------------------------------------
# 4. Load TEST data, preprocess, and build TF–IDF features
#------------------------------------------------------------

load("data/claims-test.RData")   # should create `claims_test`

clean_test <- claims_test %>%
  parse_data() %>%               # same HTML -> text pipeline as training
  select(.id, text_clean)

cat("Test size after parsing:", nrow(clean_test), "pages\n")

it_test <- itoken(
  clean_test$text_clean,
  ids         = clean_test$.id,
  progressbar = FALSE
)

dtm_test <- create_dtm(it_test, text_pipeline$vectorizer)
X_test   <- text_pipeline$tfidf$transform(dtm_test)

#------------------------------------------------------------
# 5. Predict multiclass labels on test set
#------------------------------------------------------------

mclass_pred <- predict(
  mclass_model,
  X_test,
  s    = "lambda.min",
  type = "class"
)[, 1]

#------------------------------------------------------------
# 6. Construct pred_df with required columns and save
#------------------------------------------------------------

pred_df <- tibble(
  .id         = clean_test$.id,
  bclass.pred = NA_character_,              # placeholder; fill later if you add a binary model
  mclass.pred = as.character(mclass_pred)
)

## 🔴 CHANGE groupN to your actual group number, e.g. preds-group5.RData
save(pred_df, file = file.path(out_dir, "preds-groupN.RData"))

cat("Saved predictions to", file.path(out_dir, "preds-groupN.RData"), "\n")
