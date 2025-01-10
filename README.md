# Project Title: Introduction to Machine Learning (for AI) Project

https://scikit-learn.org/1.5/modules/generated/sklearn.datasets.fetch_species_distributions.html#sklearn.datasets.fetch_species_distributions

## Repository Overview
This repository contains all the necessary files, models, and scripts for the project submission. It includes:
- Source code for the model implementation and training.
- Datasets used for training and testing.
- Analysis scripts and results.
- Final written report in PDF format.

## To-Do List
1. **Complete the Report**:
   - Ensure the report is between 3-10 pages.
   - Clearly describe what is new vs. reused code (cite reused material properly).
   - Include detailed results, analysis, and explanations for any model failure.
   - Add references to figures/tables in the report.

2. **Validate Model Training**:
   - Confirm proper use of training, validation, and test datasets.
   - Ensure overfitting is detected and addressed where necessary.
   - Document any preprocessing steps taken.

3. **Organize Repository**:
   - Ensure all source code is organized and includes comments where necessary.
   - Provide clear citations for reused code or datasets in the appropriate files.
   - Attach the source code files to the report for submission.

4. **Reproducibility Check**:
   - Verify that the steps to reproduce the project results are clear in the report.
   - Test scripts to ensure they run as expected.

## Submission Instructions
- Finalize all project components by **January 21, 2025** (flexible until January 31).
- Submit the written report (PDF) and attach all source code files.

## Notes
- The project should reflect approximately 40 hours of work.
- Ensure compliance with the evaluation rubric (quality, model training, report structure).

---

# Species Distribution Modeling Project

This project focuses on predicting species presence using geographic and environmental data. Below are the detailed steps to complete the project.

---

## 1. Load and Explore the Dataset
### Steps:
- Import the `fetch_species_distributions` function from Scikit-Learn.
- Load the dataset and examine its structure.
  - Inspect environmental feature layers (coverage attributes).
  - Understand train/test data format (latitude, longitude, species presence).
- Visualize the geographic data to get an initial sense of patterns.

---

## 2. Data Preprocessing
### Steps:
- **Extract Features:**
  - Map environmental layers to species presence points.
  - Match latitude and longitude coordinates to corresponding environmental data.
- **Handle Missing Data:**
  - Check for missing values in environmental layers or occurrence data.
  - Apply imputation techniques if necessary.
- **Normalize Features:**
  - Scale numerical features to standardize their range using tools like `StandardScaler`.
- **Feature Selection:**
  - Remove irrelevant or redundant features to simplify the model.

---

## 3. Split the Dataset
### Steps:
- Separate training and testing datasets using the provided train/test split.
- Ensure that the test set is only used for final evaluation.
- Optionally, create a validation set from the training data for hyperparameter tuning.

---

## 4. Baseline Model: Logistic Regression
### Steps:
- **Model Implementation:**
  - Train a Logistic Regression model using training data.
- **Performance Evaluation:**
  - Compute metrics such as accuracy, precision, recall, and AUC on the test set.
- **Visualization:**
  - Plot the ROC curve to evaluate classification thresholds.

---

## 5. Random Forest Classifier
### Steps:
- **Model Implementation:**
  - Train a Random Forest classifier with default parameters.
  - Experiment with hyperparameter tuning (e.g., number of trees, max depth).
- **Feature Importance:**
  - Analyze feature importance scores to identify key environmental factors.
- **Performance Evaluation:**
  - Compare metrics (accuracy, precision, recall, AUC) with Logistic Regression results.

---

## 6. Neural Network Model
### Steps:
- **Data Preparation:**
  - Convert the dataset into a format suitable for neural network training.
  - Ensure inputs are normalized and categorical variables are one-hot encoded.
- **Model Design:**
  - Build a small neural network with an input layer, hidden layers, and an output layer.
  - Use activation functions like ReLU for hidden layers and sigmoid for the output layer.
- **Training:**
  - Train the network using training data and monitor performance on a validation set.
- **Evaluation:**
  - Measure accuracy, precision, recall, and AUC on the test set.

---

## 7. Model Comparison
### Steps:
- Compare the performance of Logistic Regression, Random Forest, and Neural Network models.
  - Use accuracy, AUC, and other relevant metrics.
  - Highlight differences in predictive performance and interpretability.
- Summarize results in a comparison table.

---

## 8. Feature Importance Analysis
### Steps:
- Use Random Forest feature importance scores to identify influential environmental factors.
- Apply permutation importance or SHAP (SHapley Additive exPlanations) for more insights.
- Visualize feature importance using bar charts or heatmaps.

---

## 9. Visualization
### Steps:
- Create maps or heatmaps to show species presence predictions.
- Plot feature importance and model performance metrics.
- Use scatterplots or decision boundaries to visualize predictions.

---

## 10. Conclusion and Insights
### Steps:
- Summarize key findings, including:
  - Best-performing model and its metrics.
  - Important environmental factors driving species presence.
- Discuss strengths and limitations of each model.
- Highlight the ecological relevance of results.

---

## 11. Documentation
### Steps:
- Write a report detailing the methodology, experiments, results, and conclusions.
- Include the following sections:
  - **Introduction:** Background and objectives.
  - **Methods:** Data processing, models, and evaluation techniques.
  - **Results:** Model comparison and feature importance.
  - **Discussion:** Interpretation of results.
  - **Conclusion:** Final thoughts and potential applications.
- Incorporate visualizations and tables to support the report.

---

Follow this step-by-step guide to successfully complete the project and achieve the outlined objectives.

