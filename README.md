# Sample Size Learning Curve in Clinical Risk Prediction

## Overview

This project studies how the amount of training data affects the performance of a machine learning model for clinical risk prediction.

Using the UCI Heart Disease dataset, logistic regression models are trained on progressively larger subsets of the training data. Performance is evaluated on a fixed test set using ROC AUC and Brier score.

The objective is to understand how predictive performance evolves as more training data becomes available and whether performance stabilizes beyond a certain dataset size.

This type of analysis is commonly called a **learning curve analysis** and is useful for evaluating data efficiency and planning data collection strategies.

---

## Dataset

The dataset used in this project is the **UCI Heart Disease dataset (Cleveland subset)**.

It contains clinical measurements used to predict the presence of heart disease.

Example features include:

- age  
- sex  
- chest pain type  
- resting blood pressure  
- cholesterol  
- fasting blood sugar  
- electrocardiographic results  
- exercise-induced angina  
- ST depression  
- number of major vessels  

The original dataset encodes disease severity from 0 to 4.  
In this project the problem is converted to **binary classification**:

target = 1 if disease present (num > 0)  
target = 0 otherwise

---

## Machine Learning Pipeline

A reproducible preprocessing and modeling pipeline was implemented using scikit-learn.

Preprocessing steps include:

Numeric features  
- median imputation  
- standard scaling  

Boolean features  
- most frequent imputation  
- conversion to integer  

Categorical features  
- most frequent imputation  
- one-hot encoding  

The predictive model is **logistic regression with balanced class weights**.

Using a pipeline ensures that preprocessing is refit correctly during bootstrap resampling.

---

## Learning Curve Experiment

To study the relationship between dataset size and model performance, models were trained using progressively larger training subsets.

Training subset sizes:

- 50 samples  
- 100 samples  
- 200 samples  
- 400 samples  
- full training dataset  

For each training size:

1. A stratified subset of the training data is selected  
2. The pipeline is trained on this subset  
3. Performance is evaluated on a fixed test set  
4. Bootstrap resampling estimates uncertainty in the metrics  

---

## Evaluation Metrics

Two metrics are used to evaluate model performance.

### ROC AUC

The Area Under the Receiver Operating Characteristic Curve measures the model's ability to discriminate between positive and negative cases.

Higher values indicate better discrimination.

### Brier Score

The Brier score measures the calibration of predicted probabilities.

Lower values indicate better calibrated predictions.

---

## Results

| Training Samples | AUC Mean | AUC CI Low | AUC CI High | Brier Mean |
|------------------|----------|------------|-------------|------------|
| 50  | 0.896 | 0.877 | 0.910 | 0.126 |
| 100 | 0.898 | 0.882 | 0.914 | 0.125 |
| 200 | 0.897 | 0.883 | 0.913 | 0.126 |
| 400 | 0.896 | 0.876 | 0.910 | 0.126 |
| 736 | 0.896 | 0.877 | 0.912 | 0.126 |

The results show that model performance stabilizes quickly and that additional data beyond approximately 100 samples provides minimal improvements.

---

## Generated Outputs

The pipeline automatically generates the following output file:

results/sample_size_summary.csv

This file contains the summary metrics for each training dataset size.

---

## Project Structure

project-root/
├── src/
│   └── learning_curve.py
├── data/
│   └── heart_disease_uci.csv
├── results/
│   └── sample_size_summary.csv
├── .gitignore
└── README.md

---

## Requirements

Python 3.9+

Main dependencies:

- pandas
- numpy
- scikit-learn

Install dependencies with:

pip install pandas numpy scikit-learn

---

## Key Concepts Demonstrated

This project demonstrates several important machine learning concepts:

- learning curve analysis
- bootstrap uncertainty estimation
- stratified sampling
- reproducible preprocessing pipelines
- calibration evaluation with Brier score

These techniques are widely used in applied machine learning and clinical prediction modeling.

---

## Future Improvements

Possible extensions include:

- visualization of the learning curve with confidence intervals  
- comparison with tree-based models  
- evaluation of regularized logistic regression  
- analysis of calibration curves  

---

## License

This project is intended for educational and research purposes.

If you want, the next improvement (very small but powerful) would be addin
