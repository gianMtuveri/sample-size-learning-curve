# Sample Size and Learning Curve Analysis in Clinical Risk Prediction

## Abstract

The amount of training data available for a machine learning model strongly influences predictive performance and reliability. Understanding how performance evolves with increasing dataset size is essential when designing data collection strategies, especially in clinical applications where datasets are often limited.

This project investigates how predictive performance changes as a function of training sample size using the UCI Heart Disease dataset. Logistic regression models were trained on progressively larger subsets of the training data, and model performance was evaluated using ROC AUC and Brier score.

Results show that model performance stabilizes quickly, suggesting that additional data beyond a certain threshold provides diminishing improvements in predictive accuracy.

---

# 1 Introduction

In many machine learning applications, particularly in healthcare, obtaining large datasets can be difficult and expensive. As a result, it is important to understand how much training data is necessary for a model to reach stable performance.

Learning curve analysis provides a systematic way to evaluate how model performance evolves as more training data becomes available.

This project performs a learning curve experiment by training logistic regression models on progressively larger training subsets and measuring performance on a fixed test set.

The objective is to determine:

- how model performance scales with training data size
- whether performance plateaus after a certain dataset size
- how uncertainty in performance behaves across different sample sizes

---

# 2 Dataset

The analysis uses the **UCI Heart Disease dataset (Cleveland subset)**.

The dataset contains clinical measurements used to predict the presence of heart disease.

Example variables include:

- age
- sex
- chest pain type
- resting blood pressure
- cholesterol
- fasting blood sugar
- electrocardiogram results
- exercise-induced angina
- ST depression
- number of major vessels

The original dataset encodes disease severity from 0 to 4.

For this project the outcome is converted into a binary classification task:

target = 1 if heart disease is present  
target = 0 otherwise

---

# 3 Methods

## 3.1 Data Splitting

The dataset is divided into training and test sets using a stratified split:

- 80% training data
- 20% test data

Stratification ensures that the proportion of positive and negative cases is preserved in both sets.

The test set remains fixed throughout the experiment.

---

## 3.2 Model

The predictive model used is **logistic regression**.

Logistic regression is widely used in clinical risk modeling because it is:

- interpretable
- computationally efficient
- well suited to structured tabular data

Class imbalance is handled using balanced class weights.

---

## 3.3 Preprocessing Pipeline

A scikit-learn pipeline is used to ensure reproducible preprocessing.

Feature transformations include:

Numeric features

- median imputation
- standard scaling

Boolean features

- most frequent imputation
- conversion to integer

Categorical features

- most frequent imputation
- one-hot encoding

Embedding preprocessing inside a pipeline ensures that transformations are correctly refit during bootstrap resampling.

---

## 3.4 Learning Curve Experiment

To evaluate the effect of training dataset size, models were trained using different numbers of samples.

Training subset sizes:

- 50
- 100
- 200
- 400
- full training dataset

For each training size:

1. A stratified subset of the training data was selected
2. The model pipeline was trained on this subset
3. Performance was evaluated on the same fixed test set
4. Bootstrap resampling was used to estimate performance uncertainty

---

## 3.5 Evaluation Metrics

Two metrics were used to evaluate model performance.

### ROC AUC

The area under the receiver operating characteristic curve measures the model’s ability to discriminate between positive and negative cases.

Higher values indicate better discrimination.

### Brier Score

The Brier score measures the calibration of predicted probabilities.

Lower values indicate better calibrated predictions.

---

# 4 Results

Model performance across different training dataset sizes is shown below.

| Training Samples | AUC Mean | AUC CI Low | AUC CI High | Brier Mean |
|------------------|----------|------------|-------------|------------|
| 50  | 0.896 | 0.877 | 0.910 | 0.126 |
| 100 | 0.898 | 0.882 | 0.914 | 0.125 |
| 200 | 0.897 | 0.883 | 0.913 | 0.126 |
| 400 | 0.896 | 0.876 | 0.910 | 0.126 |
| 736 | 0.896 | 0.877 | 0.912 | 0.126 |

---

# 5 Interpretation

The results show that model performance stabilizes quickly as training size increases.

Even with only **50 training samples**, the model achieves AUC values close to those obtained with the full dataset.

Increasing the training size beyond approximately **100 samples does not produce substantial improvements** in predictive performance.

This suggests that the predictive signal contained in the dataset can already be captured with relatively small sample sizes.

Such behavior is typical for simple models applied to structured clinical datasets.

---

# 6 Discussion

Learning curve analysis provides insight into the relationship between dataset size and model performance.

Several conclusions can be drawn from this experiment.

### Early performance saturation

The logistic regression model reaches stable performance quickly, indicating that the underlying relationships between predictors and outcome are relatively simple.

### Diminishing returns from additional data

After approximately 100 training samples, further increases in dataset size provide minimal improvements in AUC and Brier score.

### Stability of predictive performance

Bootstrap confidence intervals remain relatively narrow across different training sizes, suggesting that the model is not highly sensitive to sampling variability.

---

# 7 Limitations

This analysis has several limitations.

- The dataset is relatively small compared to modern clinical datasets.
- Only logistic regression was evaluated.
- The dataset may contain strong predictive features that make the learning curve appear flatter than in more complex tasks.

Future work could explore:

- non-linear models
- regularized regression
- tree-based models
- neural networks

to determine whether more complex models benefit from larger datasets.

---

# 8 Conclusion

This project demonstrates how learning curve analysis can be used to evaluate the relationship between training dataset size and predictive performance.

Key findings include:

- logistic regression achieves strong predictive performance on the heart disease dataset
- performance stabilizes quickly as training size increases
- additional training data beyond approximately 100 samples provides limited improvement

These results highlight the importance of understanding data efficiency when designing machine learning pipelines for clinical applications.

---

# 9 Reproducibility

The experiment is implemented using a fully reproducible pipeline including:

- structured preprocessing
- logistic regression training
- stratified subsampling
- bootstrap uncertainty estimation
- automatic result export

The code and results are available in the project repository.
