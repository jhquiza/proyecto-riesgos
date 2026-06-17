# seminal-paper

Knowledge base for the seminal paper this thesis is based on.

---

## Reference

**Arias-Serna, M.A., Granda-Rodriguez, E., Loubes, J.M., Quiza-Montealegre, J.J., & Orozco-Duque, A.F.**
*Machine Learning Model for default risk prediction in Colombia's solidarity sector.*
Universidad de Medellín, Colombia.

---

## Core Objective

Develop a **transparent, robust, and fair** ML-based default prediction framework for the Colombian solidarity sector (savings and credit cooperatives), jointly evaluating:
- Predictive performance
- Interpretability (SHAP)
- Algorithmic fairness (DI, EO)
- Robustness (stress testing)

---

## Dataset

| Property | Value |
|---|---|
| Source | Colombian savings and credit cooperative |
| Observations | 12,910 consumer loans |
| Period | 4 years |
| Initial features | >60 columns → reduced to 26 |
| Women (Xi=1) | 5,988 |
| Men (Xi=0) | 6,869 |
| Defaults (Y=1) | 1,369 (509 men, 860 women) |
| Default rate | ~10.6% |
| Default definition | ≥90 days of delinquency within 12 months (Colombian Financial Superintendence) |

### 26 Variables (after feature selection)

| Variable | Description |
|---|---|
| Quota Value | Periodic payment amount (installment) |
| Loan Value | Original disbursed loan amount |
| Interest Balance (s_intereses) | Accrued unpaid interest at cutoff |
| Contributions | Borrower's equity contributions to cooperative |
| Guarantees | Collateral type |
| Guarantee Value | Estimated value of pledged asset |
| Savings Accounts | Total balance in borrower's savings |
| Current Total Income | Borrower's total reported income |
| Total Expenses | Borrower's total reported expenses |
| Restructured | Whether loan terms were formally modified |
| Capital Balance | Outstanding principal balance |
| Vinculation/Days (vinculacion) | Days since member joined cooperative |
| Term | Agreed repayment period |
| Collateral Value | Value of asset securing the loan |
| Data Score | Credit score (higher = better history) |
| Member Type (tipoasociado) | Legal nature / participation level |
| Update Data (actualizacion) | Whether borrower's data was periodically updated (binary: Yes=1 / No=0) |
| Sex | Borrower's gender (sensitive variable) |
| Age | Borrower's age |
| Stratum | Socioeconomic stratum |
| Department Group | Borrower's department |
| City Group | Borrower's city of residence |
| Client Status | Current affiliation status in cooperative |
| Economic Activity Group | Primary economic sector |

---

## Methodology

### Preprocessing
- skrub `TableVectorizer` applied across **all models** as unified preprocessor
- Handles numerical, categorical, and high-cardinality variables consistently
- Missing values imputed; feature scaling and encoding standardized
- Class imbalance handled with **class-weighted learning** (not resampling)
- All features strictly **ex-ante** to default event (no data leakage)

### Models trained
1. Logistic Regression (baseline)
2. MLP (Multilayer Perceptron)
3. Random Forest
4. LightGBM
5. XGBoost

Training: stratified cross-validation + hyperparameter optimization, scoring metric: F1.

---

## Results (Table 2)

| Model | Precision | Recall | F1-Score |
|---|---|---|---|
| **LightGBM** | **0.81** | **0.77** | **0.79** |
| **XGBoost** | **0.81** | **0.75** | **0.78** |
| Random Forest | 0.74 | 0.77 | 0.75 |
| MLP | 0.74 | 0.50 | 0.60 |
| Logistic Regression | 0.46 | 0.83 | 0.59 |

- **Best models**: XGBoost and LightGBM (F1 ~0.79)
- LR has high recall but very low precision → unacceptable for cooperatives (false positives have high operational cost)
- Tree-based ensemble methods outperform LR by ~0.20 F1

---

## Fairness Analysis

### Metrics used

**Disparate Impact (DI)**
```
DI = P(Ŷ=1 | S=0) / P(Ŷ=1 | S=1)
```
- DI ≈ 1 → fair; DI < 1 → adverse impact on protected group (S=0)
- Threshold: model DI should not fall below baseline DI in raw data
- 4/5 rule: DI ≥ 0.8 is generally acceptable

**Equal Opportunity (EO)**
```
EO = P(Ŷ=1 | S=0, Y=1) / P(Ŷ=1 | S=1, Y=1)
```
- EO = 1 → both groups have identical true positive rates
- Evaluates conditional fairness given the actual outcome

### Sensitive variable: Sex (S=0 = men, S=1 = women)

**Baseline DI in dataset**: 0.93

| Model | DI |
|---|---|
| General (data) | 0.93 |
| Logistic Regression | 0.89 |
| Random Forest | 0.94 |
| LightGBM | 0.93 |
| XGBoost | 0.94 |
| MLP | 0.92 |

→ No significant bias amplification. XGBoost and Random Forest **reduce** bias vs. baseline.

**EO result**: 95% CI = [0.904, 1.058] — includes 1.0, satisfies equal opportunity.

### Fairness techniques applied (on XGBoost and LightGBM)

1. **Separate Training (SepTr)**: Train distinct models per gender group
2. **Post-Processing for Disparate Impact (PosDi)**: Adjust decision threshold for minority group to reach DI target (default = 0.8)
3. **Cross-Validation Process**: Compute DI + accuracy across 6-fold CV for robust evaluation

Other sensitive variables analyzed: socioeconomic stratum (DI=0.99), economic activity group (DI=1.00), city group (DI=1.00) — all minimal bias.

---

## Interpretability — SHAP

### Global (SHAP Summary Plot — Figure 6)

Top features by importance (descending):

| Rank | Feature | Interpretation |
|---|---|---|
| 1 | **actualizacion** | Data update flag — active engagement with entity; Yes(1) → lower default risk |
| 2 | **s_intereses** | Accrued unpaid interest — low values → lower default probability |
| 3 | **vinculacion** | Days since joining — fewer days → higher default risk (unstable relationship) |
| 4 | **tipoasociado** | Member type — legal/participation level |

### Local (SHAP Waterfall Plot — Figure 7)

- Per-observation breakdown of feature contributions
- Red bars = increase default probability; blue bars = decrease
- s_intereses is the dominant feature for individual predictions

---

## Regulatory Context (Colombia)

| Regulation | Content |
|---|---|
| Colombian Financial Superintendence | Default = 90 days delinquency in 12 months |
| Constitutional Court C-282/2021 | Financial institutions must provide written explanation of credit denial reasons upon request |
| External Circular 0014/2022 | Must inform consumers of objective, verifiable reasons for negative credit decisions |
| Solidarity Economy Superintendence (SES) | Oversees cooperatives; traditional models (LR) are the current regulatory standard |
| EU GDPR (reference) | Right to explanation for automated decisions with material consequences |
| EU AI Act (reference) | Creditworthiness assessment = high-risk AI application |
| Basel Committee | Credit scoring models must be transparent, traceable, and rigorously validated |

---

## Key Takeaways for Thesis

1. **Best ML models**: XGBoost and LightGBM consistently outperform others (same finding as this thesis)
2. **skrub TableVectorizer**: Used as the unified preprocessing tool across all models — same as this thesis
3. **Class imbalance**: Handled with class weights, not resampling — thesis uses both strategies
4. **No significant gender bias**: Models preserve baseline DI, some even improve it
5. **Top predictive feature**: `actualizacion` (data engagement with entity) — behavioral signal
6. **Fairness ≠ excluding sensitive variables**: Proxy variables can still introduce bias
7. **LR limitation**: High recall but low precision makes it unsuitable in practice
8. **Regulatory pressure**: Colombian law requires explainable credit decisions → SHAP is not optional, it's regulatory compliance
