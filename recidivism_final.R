# Load libraries
library(tidyverse)
library(dplyr)
library(stringr)
library(explore)
library(data.table)
library(caret)
library(xgboost)
library(SHAPforxgboost)
library(DALEX)
library(pdp)
library(ggplot2)
library(verification)

# Load data
data <- read_csv("D:/recidivism/recidivism.csv")

# Convert categorical variable to numerical values
data$Supervision_Level_First <- factor(data$Supervision_Level_First, levels = c("Specialized", "High", "Standard"), labels = c(3, 2, 1))
data$Education_Level <- factor(data$Education_Level, levels = c("At least some college", "High School Diploma", "Less than HS diploma"), labels = c(3, 2, 1))

# Remove unnecessary columns
cols_to_remove <- c("ID", "Avg_Days_per_DrugTest", "DrugTests_THC_Positive", "DrugTests_Cocaine_Positive", "DrugTests_Meth_Positive", "DrugTests_Other_Positive", "Recidivism_Arrest_Year2", "Recidivism_Arrest_Year1")
data <- data[, !names(data) %in% cols_to_remove]

# Extract integers from specified text columns
text_cols <- c('Age_at_Release', 'Dependents', 'Prison_Years', 'Prior_Arrest_Episodes_Felony', 'Supervision_Risk_Score_First', 'Supervision_Level_First', 'Prior_Arrest_Episodes_Misd', 'Prior_Arrest_Episodes_Violent', 'Prior_Arrest_Episodes_Property', 'Prior_Arrest_Episodes_Drug', 'Prior_Arrest_Episodes_PPViolationCharges', 'Prior_Conviction_Episodes_Felony', 'Prior_Conviction_Episodes_Misd', 'Prior_Conviction_Episodes_Prop', 'Prior_Conviction_Episodes_Drug', 'Delinquency_Reports', 'Program_Attendances', 'Program_UnexcusedAbsences', 'Residence_Changes', 'Percent_Days_Employed', 'Jobs_Per_Year', 'Education_Level')
for (col in text_cols) {
  data[[col]] <- sapply(data[[col]], function(string) {
    match <- str_match(string, "\\b(\\d+)(?:-\\d+)?\\b")
    if (!is.na(match[2])) {
      return(as.integer(match[2]))
    } else {
      return(NA)
    }
  })
}

# Fill NA values with mode for numeric columns
get_mode <- function(x) {
  uniqx <- unique(x)
  uniqx[which.max(tabulate(match(x, uniqx)))]
}

columns_to_fill <- c("Jobs_Per_Year", "Percent_Days_Employed", "Supervision_Risk_Score_First", "Supervision_Level_First", "Education_Level")
data[columns_to_fill] <- apply(data[columns_to_fill], 2, function(x) {
  ifelse(is.na(x), get_mode(x), x)
})

# Fill NA values with mode for logical columns
get_mode_logical <- function(x) {
  ifelse(sum(x, na.rm = TRUE) > length(x) / 2, TRUE, FALSE)
}

columns_to_fill <- c("Gang_Affiliated")
data[columns_to_fill] <- lapply(data[columns_to_fill], function(x) {
  ifelse(is.na(x), get_mode_logical(x), x)
})

# Replace NA values with custom value for specified columns
custom_value <- "None"
columns_to_replace_na <- c("Prison_Offense")
for (col in columns_to_replace_na) {
  data[[col]] <- ifelse(is.na(data[[col]]), custom_value, data[[col]])
}

# Convert categorical columns to factors
categorical_cols <- c('Gender', 'Race', 'Education_Level', 'Prison_Offense')
data[, categorical_cols] <- lapply(data[, categorical_cols, with = FALSE], as.factor)

# Encode categorical variables using label encoding
label_encode <- function(x) {
  as.integer(as.factor(x))
}
data[, categorical_cols] <- lapply(data[, categorical_cols, with = FALSE], label_encode)

# Standardize numeric columns
numeric_cols <- sapply(data, is.numeric) & !sapply(data, is.logical)
numeric_data <- data[, numeric_cols]
scaled_data <- scale(numeric_data)
data <- cbind(scaled_data, data[!numeric_cols])

# Split data into X and Y
col_names <- names(data)
Y <- data[, col_names[length(col_names)]]
X <- data[, col_names[-length(col_names)]]

# Train-test splitting
set.seed(123) # For reproducibility
train_index <- sample(1:nrow(data), 0.7 * nrow(data)) # 70% training data
test_index <- setdiff(1:nrow(data), train_index) # Remaining data for testing
X_train <- data[train_index, -ncol(data)]
Y_train <- data[train_index, ncol(data)]
X_test <- data[test_index, -ncol(data)]
Y_test <- data[test_index, ncol(data)]

# Train XGBoost model
dtrain <- xgb.DMatrix(as.matrix(X_train), label = Y_train)
xgb_model <- xgboost(data = dtrain, nrounds = 100, objective = "binary:logistic")

# Make predictions
dtest <- xgb.DMatrix(as.matrix(X_test))
predictions <- predict(xgb_model, dtest, type = "response")

# Calculate performance metrics
binary_predictions <- ifelse(predictions > 0.5, 1, 0)
conf_matrix <- table(binary_predictions, Y_test)
accuracy <- mean(binary_predictions == Y_test)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * precision * recall / (precision + recall)

# SHAP values
shap_values <- shap.values(xgb_model, dtrain)
mean_shap_score <- shap_values$mean_shap_score

# SHAP summary plot
dataX <- as.matrix(X_train)
shap_long <- shap.prep(xgb_model = xgb_model, X_train = dataX)
shap_long <- shap.prep(shap_contrib = shap_values$shap_score, X_train = dataX)
shap_summary_plot <- shap.plot.summary(shap_long)

ggsave("shap_summary_plot.png", shap_summary_plot, width = 10, height = 6, units = "in")

# Print performance metrics
print("Confusion Matrix:")
print(conf_matrix)
print(paste("Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1 Score:", f1_score))
