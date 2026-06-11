# CreditWise-Loan-Approval-System
CreditWise is a machine learning–based loan approval system that predicts borrower eligibility using financial and demographic data, helping automate and improve credit decision-making.

## Dataset

The dataset (`loan_approval_data.csv`) contains applicant-level records with the following types of features:

- **Demographics**: Gender, Marital Status, Education Level, Age
- **Employment**: Employment Status, Employer Category
- **Financials**: Applicant Income, Coapplicant Income, Savings, Loan Amount, DTI Ratio
- **Credit**: Credit Score
- **Loan Details**: Loan Purpose, Property Area
- **Target**: Loan_Approved (Yes/No)

## Project Workflow

### 1. Data Loading & Inspection
- Load the dataset with pandas
- Inspect structure, data types, missing values, and summary statistics

### 2. Handling Missing Values
- Numerical columns → imputed with **mean** (`SimpleImputer`)
- Categorical columns → imputed with **most frequent value**

### 3. Exploratory Data Analysis (EDA)
- Class balance check for the target variable (pie chart)
- Distribution of categorical features (bar plots)
- Distribution of Applicant Income and Coapplicant Income (histograms)
- Outlier detection via box plots (Income, Credit Score, Age, DTI Ratio, Savings, Loan Amount vs. Loan Approval)
- Credit Score and Income distributions split by approval status

### 4. Feature Cleaning
- Dropped `Applicant_ID` (no predictive value)

### 5. Feature Encoding
- **Label Encoding**: `Education_Level`, `Loan_Approved` (target)
- **One-Hot Encoding**: `Marital_Status`, `Employment_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category`

### 6. Correlation Analysis
- Correlation heatmap of numeric features
- Ranked correlations with `Loan_Approved` to identify key drivers

### 7. Train-Test Split & Feature Scaling
- 80/20 train-test split (`random_state=42`)
- Standardized features using `StandardScaler`

### 8. Model Training & Evaluation (Baseline)
Trained and evaluated three classifiers on the scaled features:
- Logistic Regression
- K-Nearest Neighbors (k=5)
- Gaussian Naive Bayes

Metrics used: **Precision, Recall, F1-score, Accuracy, Confusion Matrix**

> Best baseline model (by precision): **Naive Bayes**

### 9. Feature Engineering
- Added squared terms: `DTI_Ratio_sq`, `Credit_Score_sq`
- Dropped original `Credit_Score` and `DTI_Ratio` in favor of engineered features
- Re-split and re-scaled the data

### 10. Model Re-Evaluation
Retrained the same three models (Logistic Regression, kNN, Naive Bayes) on the engineered feature set and compared performance against the baseline.

## Tech Stack
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## How to Run
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Place `loan_approval_data.csv` in the project directory
4. Open and run `Loan_approval.ipynb`
