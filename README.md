# Recidivism Prediction Project

This project focuses on predicting recidivism using a dataset and various machine learning techniques. The main tasks include data preprocessing, model training using XGBoost, and evaluating the model's performance. Additionally, SHAP (SHapley Additive exPlanations) values are used to interpret the model's predictions.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
  - [Libraries](#libraries)
  - [Data Loading](#data-loading)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [SHAP Values and Interpretation](#shap-values-and-interpretation)
- [Performance Metrics](#performance-metrics)
- [SHAP Summary Plot](#shap-summary-plot)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, you need to have R installed on your machine. Additionally, you need to install the following R libraries:

```R
install.packages(c("tidyverse", "dplyr", "stringr", "explore", "data.table", "caret", "xgboost", "SHAPforxgboost", "DALEX", "pdp", "ggplot2", "verification"))
```

## Usage

1. Clone the repository to your local machine.
2. Set the path to your recidivism dataset in the script.
3. Run the R script to perform data preprocessing, model training, evaluation, and SHAP value computation.
4. Review the generated performance metrics and SHAP summary plot.

## Code Overview

### Libraries

The following libraries are used in this project:

- `tidyverse`, `dplyr`, `stringr`, `explore`: For data manipulation and exploration.
- `data.table`: For efficient data handling.
- `caret`: For model training and evaluation.
- `xgboost`: For training the XGBoost model.
- `SHAPforxgboost`, `DALEX`, `pdp`: For model interpretation and explanation.
- `ggplot2`: For plotting.
- `verification`: For calculating performance metrics.

### Data Loading

The dataset is loaded using the `read_csv` function from the `readr` package:

```R
data <- read_csv("D:/recidivism/recidivism.csv")
```

### Data Preprocessing

- Convert categorical variables to numerical values.
- Remove unnecessary columns.
- Extract integers from specified text columns.
- Fill NA values with the mode for numeric and logical columns.
- Replace NA values with a custom value for specified columns.
- Convert categorical columns to factors and then use label encoding.
- Standardize numeric columns.

### Model Training and Evaluation

- Split the data into training and testing sets.
- Train an XGBoost model on the training data.
- Make predictions on the testing data.
- Calculate performance metrics: confusion matrix, accuracy, precision, recall, and F1 score.

### SHAP Values and Interpretation

- Compute SHAP values to interpret the model.
- Generate a SHAP summary plot to visualize the importance of features.

## Performance Metrics

The script outputs the following performance metrics:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score

## SHAP Summary Plot

The SHAP summary plot is saved as `shap_summary_plot.png` and provides insights into the importance of various features in the model's predictions.
