# Load required libraries
library(glmnet)
library(e1071)
library(caret)

load_and_process_data <- function() {
  # Load dataset
  url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
  data <- read.csv(url, sep = ";")
  data$quality <- as.numeric(data$quality)
  data$category <- ifelse(data$quality < 6, "Low", "High")
  
  # Scale and split dataset
  preprocessing <- preProcess(data, method = c("center", "scale"))
  scaled_data <- predict(preprocessing, data)
  set.seed(42)
  trainIndex <- createDataPartition(scaled_data$quality, p = 0.8, list = FALSE, times = 1)
  train <- scaled_data[trainIndex, ]
  test <- scaled_data[-trainIndex, ]
  
  # Plot all features against category
  for (feature in colnames(data)[1:11]) {
    plot_data <- data[, c(feature, "category")]
    print(ggplot(plot_data, aes(x = category, y = .data[[feature]], color = category)) +
            geom_point() +
            ggtitle(paste0("Scatter plot of ", feature, " against category")))
  }
  
  return(list(train = train, test = test))
}

# Train and evaluate SVM regression model with cross-validation
svm_regression <- function(train, test) {
  control <- trainControl(method = "cv", number = 5)
  grid <- expand.grid(C = c(0.01, 0.1, 1, 10, 100))
  
  svm_regression_tuned <- train(quality ~ . - category, data = train, method = "svmLinear", trControl = control, tuneGrid = grid)
  predictions <- predict(svm_regression_tuned, test)
  
  mse <- mean((predictions - test$quality)^2)
  r_squared <- 1 - mse / var(test$quality)
  mae <- mean(abs(predictions - test$quality))
  
  # Predict the test data
  predicted_values <- predict(svm_regression_tuned, test)
  
  # Create a scatter plot of predicted vs. actual values
  plot(predicted_values, test$quality, xlab = "Predicted Quality", ylab = "Actual Quality", main = "Predicted vs. Actual Quality")
  abline(a = 0, b = 1, col = "red") # Add a diagonal line to visualize the perfect prediction
  
  # Create a residual plot
  residuals <- predicted_values - test$quality
  plot(predicted_values, residuals, xlab = "Predicted Quality", ylab = "Residuals", main = "Residual Plot")
  abline(h = 0, col = "red") # Add a horizontal line at y = 0 to visualize the perfect prediction
  
  # Create a histogram of residuals
  hist(residuals, breaks = 30, main = "Histogram of Residuals", xlab = "Residuals", col = "lightblue", border = "black")
  
  return(list(mse = mse, r_squared = r_squared, mae = mae))
}

# Train and evaluate SVM classification model with cross-validation
svm_classification <- function(train, test) {
  control <- trainControl(method = "cv", number = 5)
  grid <- expand.grid(C = c(0.01, 0.1, 1, 10, 100))
  
  svm_classification_tuned <- train(category ~ . - quality, data = train, method = "svmLinear", trControl = control, tuneGrid = grid)
  predictions <- predict(svm_classification_tuned, test)
  svm_classification_tuned
  
  accuracy <- mean(predictions == test$category)
  
  # Compute the confusion matrix
  confusion_matrix <- confusionMatrix(table(Predicted = predict(svm_classification_tuned, test), Actual = test$category))
  
  # Plot the confusion matrix
  library(caret)
  fourfoldplot(confusion_matrix$table, color = c("red", "blue"), conf.level = 0, margin = 1, main = "Confusion Matrix")
  
  # Calculate and print the classification performance metrics (Figure 29)
  classification_performance_metrics <- confusionMatrix(table(Predicted = predict(svm_classification_tuned, test), Actual = test$category))
  print(classification_performance_metrics)
  
  return(accuracy)
}

# Main code
data <- load_and_process_data()
train <- data$train
test <- data$test

# Train and evaluate SVM regression model with cross-validation
regression_results <- svm_regression(train, test)
print(paste0("SVM regression with cross-validation - MSE: ", round(regression_results$mse, 4)))
print(paste0("SVM regression with cross-validation - R-squared: ", round(regression_results$r_squared, 4)))
print(paste0("SVM regression with cross-validation - MAE: ", round(regression_results$mae, 4)))

train$category <- as.factor(train$category)
test$category <- as.factor(test$category)

# Train and evaluate SVM classification model with cross-validation
accuracy_cv <- svm_classification(train, test)
print(paste0("SVM classification with cross-validation - Accuracy: ", round(accuracy_cv, 4)))