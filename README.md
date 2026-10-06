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
