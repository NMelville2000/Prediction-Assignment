# Practical Machine Learning Project
# Random Forest Model

library(caret)
library(randomForest)
library(dplyr)

set.seed(123)

# Load data
training <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))
testing  <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",  na.strings = c("NA", "", "#DIV/0!"))
```

# Remove columns with more than 95% missing values
na_cols <- colSums(is.na(training)) / nrow(training)
training_clean <- training[, na_cols < 0.95]
testing_clean  <- testing[, names(training_clean)[names(training_clean) != "classe"]]

# Remove ID and timestamp variables
remove_cols <- c(
  "X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2",
  "cvtd_timestamp", "new_window", "num_window"
)
remove_cols <- remove_cols[remove_cols %in% names(training_clean)]

training_clean <- training_clean %>% select(-all_of(remove_cols))
testing_clean  <- testing_clean %>% select(-all_of(remove_cols))

# Remove near zero variance predictors based on predictors only
predictor_names <- setdiff(names(training_clean), "classe")
nzv <- nearZeroVar(training_clean[, predictor_names])
if (length(nzv) > 0) {
  keep_predictors <- predictor_names[-nzv]
  training_clean <- training_clean[, c(keep_predictors, "classe")]
  testing_clean  <- testing_clean[, keep_predictors]
}

# Make outcome a factor with fixed levels
training_clean$classe <- factor(training_clean$classe)
classe_levels <- levels(training_clean$classe)

# Force any character predictors to have matching types in both datasets
common_predictors <- intersect(names(testing_clean), setdiff(names(training_clean), "classe"))
for (col in common_predictors) {
  if (is.character(training_clean[[col]]) || is.factor(training_clean[[col]]) ||
      is.character(testing_clean[[col]]) || is.factor(testing_clean[[col]])) {
    combined_levels <- unique(c(as.character(training_clean[[col]]), as.character(testing_clean[[col]])))
    training_clean[[col]] <- factor(as.character(training_clean[[col]]), levels = combined_levels)
    testing_clean[[col]]  <- factor(as.character(testing_clean[[col]]),  levels = combined_levels)
  }
}

# Split data 70:30
in_train <- createDataPartition(training_clean$classe, p = 0.70, list = FALSE)
train_data <- training_clean[in_train, ]
valid_data <- training_clean[-in_train, ]

# Reapply fixed outcome levels after split
train_data$classe <- factor(train_data$classe, levels = classe_levels)
valid_data$classe <- factor(valid_data$classe, levels = classe_levels)

# Train Random Forest model
rf_model <- randomForest(
  classe ~ .,
  data = train_data,
  ntree = 250,
  importance = TRUE
)

print(rf_model)

# Validation
valid_pred <- predict(rf_model, newdata = valid_data)
valid_pred <- factor(valid_pred, levels = classe_levels)
reference <- factor(valid_data$classe, levels = classe_levels)

cm <- confusionMatrix(data = valid_pred, reference = reference)
print(cm)
print(paste("Validation Accuracy:", round(cm$overall["Accuracy"], 4)))
print(paste("Expected Out of Sample Error:", round(1 - cm$overall["Accuracy"], 4)))

# Final predictions for the 20 test cases
final_predictions <- predict(rf_model, newdata = testing_clean)
final_predictions <- factor(final_predictions, levels = classe_levels)
print(final_predictions)
