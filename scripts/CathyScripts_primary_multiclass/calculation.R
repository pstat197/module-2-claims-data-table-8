library(yardstick)
library(dplyr)
library(tibble)

# Predicted classes on the training data using lambda.min
pred_multi_cv <- predict(
  cv_multi,
  X_train,
  s    = "lambda.min",
  type = "class"
)[, 1]

# Overall accuracy
acc_multi_cv <- mean(pred_multi_cv == y_multi)
cat("Multiclass CV accuracy (lambda.min):", round(acc_multi_cv, 3), "\n")

# Put truth and predictions into a tibble for yardstick
eval_df <- tibble(
  truth    = y_multi,
  estimate = factor(pred_multi_cv, levels = levels(y_multi))
)

# Overall accuracy via yardstick
acc_overall <- accuracy(eval_df, truth, estimate)

# Macro-averaged sensitivity
sens_macro <- sens(eval_df, truth, estimate, estimator = "macro")

# Macro-averaged specificity
spec_macro <- spec(eval_df, truth, estimate, estimator = "macro")

# Combine into one summary table
overall_metrics <- bind_rows(
  acc_overall  %>% mutate(metric = "accuracy")  %>% select(metric, .estimate),
  sens_macro   %>% mutate(metric = "sensitivity_macro") %>% select(metric, .estimate),
  spec_macro   %>% mutate(metric = "specificity_macro") %>% select(metric, .estimate)
)

overall_metrics