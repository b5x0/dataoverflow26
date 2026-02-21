---
phase: 9
plan: 1
wave: 1
---

# Plan 9.1: Optimization Sprint

## Objective
Write an `optimize.py` script that attacks the Macro F1 score through deep feature engineering (interaction terms and K-Fold Target Encoding) and thorough Optuna hyperparameter searching (50 trials on LightGBM).

## Context
- .gsd/SPEC.md
- solution.py
- read_pdf.py (Hackathon Constraints)

## Tasks

<task type="auto">
  <name>Generate optimize.py Script</name>
  <files>optimize.py</files>
  <action>
    - Ensure `optuna` and `category_encoders` are installed.
    - Re-implement baseline preprocessing with additional deep features:
        - `Income_per_Dependent` = `Estimated_Annual_Income / (Total_Dependents + 1)`
        - `Risk_Ratio` = `Risk_Score_Proxy / (Years_Without_Claims + 1)`
    - Implement `category_encoders.TargetEncoder` for `Region_Code`, `Broker_ID`, `Employer_ID`. Ensure the encoder is exported to `target_encoder.pkl`.
    - Define Optuna objective wrapping a 5-fold Stratified K-Fold CV pipeline training a LightGBM `LGBMClassifier(class_weight='balanced')` optimizing for validation `f1_macro`.
    - Set hyperparameter search space: `num_leaves`, `max_depth`, `learning_rate`, `min_child_weight`, `subsample`, `colsample_bytree`. Wait for 50 trials.
    - Export the final model trained on the best params to `model_opt.pkl`.
  </action>
  <verify>python -c "import os; print(os.path.exists('optimize.py'))"</verify>
  <done>Script exists and executes the search space properly.</done>
</task>

## Success Criteria
- [ ] `optimize.py` performs 50 trials and exports `model_opt.pkl` and `target_encoder.pkl`
- [ ] Script successfully engineers target encoded fields and ratio interactions.
