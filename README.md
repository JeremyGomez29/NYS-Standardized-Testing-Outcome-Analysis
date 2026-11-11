# Predicting NYS Test Proficiency with LightGBM and a ReLU-Penalized Loss

**Author:** Jeremy Gomez (Kaggle: `Jerm.gome`)

This repository holds my entry for the STA 9890 Kaggle prediction competition. The task was to predict the **percent of students who scored proficient** on New York State standardized tests for each school, assessment and student subgroup. The final model is a **LightGBM regressor with a custom ReLU-penalized loss**, so predictions stay inside the 0–100% range. Its hyperparameters were tuned with **Optuna** using 6-fold cross-validation.

---

## Repository contents

| File | Description |
|---|---|
| `Project_Code.ipynb` | End-to-end notebook: data loading and joins, imputation, feature engineering, the ReLU-penalized LightGBM, Optuna tuning, held-out evaluation, diagnostics, retraining on all labeled data, and the Kaggle submission file |
| `LightGBM_with_ReLU_Loss_Function_After_Action_Report.pdf` | After-action report on pipeline decisions, an analysis of systematic errors, feature importance and what I'd change next time |
| `Project_Presentation.pdf` | Slide deck summarizing the approach |

## Data

The competition data loads directly from the course site in the notebook's first cell, so no download is needed:

```python
base_path = "https://michael-weylandt.com/STA9890/competition_data/"
```

| File | Contents |
|---|---|
| `scores_training.csv` | About 145k labeled rows, one per school × assessment × subgroup, with the `PERCENT_PROFICIENT` target |
| `scores_test.csv` | About 48k unlabeled rows used for the Kaggle submission |
| `school_covariates.csv` | School-level features: demographics, funding, class sizes, staffing, graduation outcomes |
| `district_covariates.csv` | District-level features |

---

## Pipeline

```
Data cleaning & imputation → Feature engineering → 80/20 train/test split
   → LightGBM + ReLU loss → Optuna tuning (6-fold CV) → Retrain on all labeled data
```

### 1. Joins and imputation
- The school and district covariates are left-joined onto the score data.
- Five features with more than 60% missing values are dropped: teacher turnover rate and the kindergarten, grade 1, grade 2 and history class-size averages.
- The remaining numeric features, each under 12% missing, are filled with **the median for the same school**, falling back to the **statewide median**. No categorical feature has missing values.

### 2. Feature engineering
Five engineered features are added:

| Feature | Definition |
|---|---|
| Total Funding per Pupil | Federal + local funding per pupil |
| Federal Funding Ratio | Federal funding ÷ total funding |
| Mean Class Size | Average of the ELA, math and science class sizes |
| Student–Teacher Ratio | Number of pupils ÷ number of teachers |
| Subgroup Share | Students taking the exam ÷ total pupils |

Categorical features (school, district, county, assessment, subgroup, district type and region) go to LightGBM as native categoricals, without one-hot encoding.

### 3. LightGBM with a ReLU-penalized loss
LightGBM's default L2 loss doesn't stop predictions from falling below 0% or rising above 100%. The custom objective adds a scaled ReLU hinge penalty to the MSE gradient whenever a prediction leaves that range:

```python
grad = 2 * (y_pred - y_true) \
       - LAMBDA * (y_pred < 0) \
       + LAMBDA * (y_pred > 100)
hess = 2
```

`LAMBDA` was tuned along with the other hyperparameters. In testing, this beat training with plain L2 and clipping the predictions afterward.

### 4. Hyperparameter tuning with Optuna
Optuna's TPE sampler tuned 11 LightGBM hyperparameters plus `LAMBDA`, with a median pruner and a fixed learning rate of 0.06. Each trial was scored as the average MSE across 6 folds. Gains levelled off after about 30 trials, which took roughly 3 hours on a high-RAM Google Colab CPU. The notebook runs 30 trials.

### 5. Held-out evaluation
A model with the tuned hyperparameters is fit on the 80% training split and scored on the 20% held-out split. The notebook then shows residual plots and feature importance by gain (top 10 and bottom 10).

### 6. Final fit and submission
The training and held-out splits are combined, and a new model is fit on **all** labeled rows with the same tuned hyperparameters. On the unblinded competition test set, this lowered MSE by about 10 points. The competition test rows go through the same joins, column drops, imputation (using training-set medians), feature engineering and fitted preprocessor. Predictions are clipped to 0–100 and written to `full_lgbm_kaggle_submission.csv`.

---

## Results and insights

- **Best 6-fold CV MSE:** 92.2 for the ReLU-loss LightGBM.
- **Held-out 20% split:** test MSE 83.1, test R² 0.880, test MAE 6.25. The train MSE is 6.5, so the model fits the training data much more closely than unseen data.
- **No overall bias:** the mean residual on the test set is −0.035 and the median is −0.043.
- **Bounded-target pattern:** residuals slope downward against actual proficiency. Low-proficiency subgroups are over-predicted and high-proficiency subgroups under-predicted, with wider spread at the extremes. This fits a bounded target where predictions regress toward the mean.
- **No geographic bias:** mean residuals by county range only from −2.0 to +1.4.
- **Feature importance:** the high-cardinality categoricals **School**, **Assessment** and **District** dominate gain. Subgroup Share is the only engineered feature in the top 10. Features such as % in foster care, % dropout, % GED and % migrant contribute little, likely because the school identifier already captures them.

## What I'd do differently

- **Drop redundant inputs after engineering features.** For example, remove federal and local funding per pupil once total funding per pupil exists. This would cut training time without losing information.
- **Address the bounded-target residual pattern directly.** Options include a transformed target paired with a matching objective, or a stacking ensemble with a meta-learner on top.
- **Put more effort into feature engineering** and remove features the categorical IDs already capture.

---

## Running the notebook

### Requirements
Python 3.10+ with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm optuna
```

The notebook installs Optuna itself with `!pip install optuna -q`, which works out of the box in Google Colab.

### Steps
1. Open `Project_Code.ipynb` in Jupyter or Google Colab. A high-RAM runtime is recommended.
2. Run all cells in order. The data downloads automatically from the course site.

Optuna tuning is the slow step and can take a few hours. The results aren't fully deterministic, because the trials run in parallel (`n_jobs=-1`), so the best hyperparameters and scores may differ slightly from those reported here.

### Outputs
| File | Description |
|---|---|
| `full_lgbm_new_objective.txt` | Saved LightGBM booster trained on the 80% split |
| `submission_model_full_data.txt` | Saved LightGBM booster trained on all labeled data |
| `full_lgbm_kaggle_submission.csv` | Kaggle submission with `ASSESSMENT_ID, PERCENT_PROFICIENT` columns, clipped to 0–100 |

## AI usage
Google Gemini was used as a coding assistant through its Google Colab integration. No generative AI was used for writing or idea generation in the report.
