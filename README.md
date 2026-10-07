# Wine Quality Prediction

End-to-end machine learning project that predicts whether a red wine is GOOD (quality ≥ 7) or BAD (quality < 7) from its physicochemical properties.

## Highlights-

Full pipeline: EDA → correlation analysis → binary target → stratified split → model comparison → hyperparameter tuning → feature importance.

Handles a heavily imbalanced problem (only ~13.6% of wines are GOOD) using stratified splitting and precision / recall / F1 instead of accuracy alone.

Compares Logistic Regression, KNN and Decision Tree, including the effect of feature scaling.

Tuned Decision Tree with GridSearchCV: F1 improved from 0.67 → 0.71, accuracy 92.5%.

Explains what drives wine quality (alcohol is the strongest signal).

## RESULT
Evaluated on a held-out, stratified test set (20%, 320 samples).

<img width="1774" height="887" alt="Model Performance Results Table" src="https://github.com/user-attachments/assets/52210c2f-a6f9-4f5f-a5b1-cf54151c4ac8" />

## Takeaways

All models hit ~89% accuracy, but that is misleading: predicting "BAD" for everything already gives ~86%. 

The linear and KNN models miss most good wines (recall 0.33–0.42).


The Decision Tree is the only model that finds ~70% of good wines, so it wins on F1.


Scaling improved Logistic Regression recall and F1 but did not change accuracy.

## What Drives Wine Quality?

Correlation with quality:
<table>
  <tr>
    <th align="left">Feature</th>
    <th align="right">Correlation</th>
  </tr>
  <tr><td>Alcohol</td><td align="right">+0.48</td></tr>
  <tr><td>Sulphates</td><td align="right">+0.25</td></tr>
  <tr><td>Citric acid</td><td align="right">+0.23</td></tr>
  <tr><td>Volatile acidity</td><td align="right">−0.39</td></tr>
  <tr><td>Total sulfur dioxide</td><td align="right">−0.19</td></tr>
  <tr><td>Density</td><td align="right">−0.17</td></tr>
</table>

# 📂 Project Structure

Wine-Quality-Project-ML/


├── wine_quality_prediction.ipynb  &ensp; # Full analysis and modeling notebook


├── winequality.csv   &ensp; # Dataset (1,599 samples, 11 features)


└── README.md

# 🗂️ Dataset
Red Wine Quality dataset (Vinho Verde, from the UCI Machine Learning Repository).


1,599 samples, 11 numerical features, no missing values.


Features: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol.


Target: quality (score 3–8), converted to binary: GOOD if ≥ 7 (217 wines), BAD otherwise (1,382 wines).

# Getting Started

## Clone
git clone https://github.com/EklavyaJain1/Wine-Quality-Project-ML.git <br>
cd Wine-Quality-Project-ML

## Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

## Run
jupyter notebook wine_quality_prediction.ipynb

You can also upload the notebook and winequality.csv to Google Colab and run all cells.

# Tech Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Jupyter
