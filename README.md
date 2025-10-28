# Machine Learning Algorithms Project

A comprehensive collection of machine learning regression algorithms and practical applications, including hypothesis testing and real-world datasets.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Notebooks](#notebooks)
- [Datasets](#datasets)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Key Learnings](#key-learnings)
- [Models](#models)

## Overview

This project contains implementations and practical applications of various machine learning regression techniques, statistical hypothesis testing, and real-world case studies. Each notebook provides detailed explanations, visualizations, and hands-on examples.

## Project Structure

```
ml-algorithm/
├── data/                           # Dataset files
│   ├── economic_index.csv
│   ├── height-weight.csv
│   ├── Algerian_forest_fires_dataset_UPDATE.csv
│   ├── forest_fire_cleaned.csv
│   └── forest_fire_regression.csv
├── model/                          # Saved model files
│   ├── linear_regression_model.pkl
│   ├── Ridge_linear_regression_model.pkl
│   └── scaler.pkl
├── simple_linear_regression.ipynb
├── multiple_linear_regression.ipynb
├── multiple_linear2.ipynb
├── polynomial_regression.ipynb
├── linear_project.ipynb
├── forest_fire.ipynb
├── forest_fire_model_training.ipynb
├── 08-practical-lab (1).ipynb
├── requirements.txt
└── README.md
```

## Notebooks

### 1. Simple Linear Regression
**File:** [simple_linear_regression.ipynb](simple_linear_regression.ipynb)

Introduction to linear regression with a single predictor variable.
- Basic concepts and theory
- Fitting a simple linear model
- Model evaluation metrics (R², MSE, MAE)
- Visualization of regression line
- Predictions and residual analysis

### 2. Multiple Linear Regression
**Files:**
- [multiple_linear_regression.ipynb](multiple_linear_regression.ipynb)
- [multiple_linear2.ipynb](multiple_linear2.ipynb)

Regression with multiple predictor variables.
- California Housing dataset analysis
- Feature selection and engineering
- Multiple predictor variables
- Multicollinearity detection (VIF)
- Model comparison and evaluation
- Standardization and scaling
- Ridge regression implementation

### 3. Polynomial Regression
**File:** [polynomial_regression.ipynb](polynomial_regression.ipynb)

Non-linear relationships using polynomial features.
- Polynomial feature transformation
- Degree selection
- Overfitting vs underfitting
- Cross-validation
- Model complexity analysis

### 4. Linear Regression Project
**File:** [linear_project.ipynb](linear_project.ipynb)

End-to-end regression project with real-world data.
- Complete data analysis pipeline
- Feature engineering
- Model training and validation
- Performance evaluation
- Business insights

### 5. Forest Fire Prediction
**Files:**
- [forest_fire.ipynb](forest_fire.ipynb) - Data cleaning and EDA
- [forest_fire_model_training.ipynb](forest_fire_model_training.ipynb) - Model development

Predicting forest fire occurrences using weather and environmental data.
- **Dataset:** Algerian Forest Fires (2012)
- **Regions:** Bejaia and Sidi-Bel Abbes
- **Features:** Temperature, humidity, wind speed, rain, FWI indices
- Data cleaning and preprocessing
- Exploratory data analysis
- Feature correlation analysis
- Fire risk classification
- Model training and evaluation
- Saved models for deployment

### 6. Hypothesis Testing Practical Lab
**File:** [08-practical-lab (1).ipynb](08-practical-lab%20(1).ipynb)

Comprehensive statistical hypothesis testing scenarios.
- **Scenario 1:** Drug trial analysis (Welch's t-test)
- **Scenario 2:** A/B testing for website conversion
- **Scenario 3:** Chi-squared test for categorical associations
- **Scenario 4:** Non-parametric testing (Mann-Whitney U)
- Effect size calculations (Cohen's d)
- Confidence intervals
- Assumption checking (normality, equal variance)
- Business impact analysis

## Datasets

### 1. Economic Index
**File:** `data/economic_index.csv`
- Economic indicators and trends

### 2. Height-Weight
**File:** `data/height-weight.csv`
- Anthropometric data for regression analysis

### 3. Algerian Forest Fires
**Files:**
- `data/Algerian_forest_fires_dataset_UPDATE.csv` (raw)
- `data/forest_fire_cleaned.csv` (processed)
- `data/forest_fire_regression.csv` (modeling-ready)

**Features:**
- Temperature, RH (Relative Humidity), Ws (Wind Speed)
- Rain, FFMC, DMC, DC (Fire Weather Indices)
- ISI, BUI, FWI (Fire Behavior Indices)
- Classes (fire/not fire), Region (0: Bejaia, 1: Sidi-Bel Abbes)

### 4. California Housing
**Source:** `sklearn.datasets.fetch_california_housing()`
- 20,640 instances with 8 features
- Median house values in California districts

## Requirements

### Python Packages
```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
scipy
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-algorithm
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Notebooks

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Navigate to the desired notebook and open it

3. Run cells sequentially (Shift + Enter)

### Loading Saved Models

```python
import pickle

# Load trained model
with open('model/linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Make predictions
predictions = model.predict(scaler.transform(new_data))
```

## Key Learnings

### Statistical Concepts
- **Hypothesis Testing:** Always check assumptions before selecting tests
- **Effect Sizes:** P-values alone don't tell the full story; include confidence intervals and effect sizes
- **Visualizations:** Essential for revealing outliers, skewness, and patterns
- **Robust Methods:** Use nonparametric tests when assumptions are violated
- **Context Matters:** Interpret statistical significance alongside business impact

### Machine Learning Best Practices
- Data preprocessing and cleaning are critical
- Feature engineering can significantly improve model performance
- Always split data into training and testing sets
- Use cross-validation for robust model evaluation
- Regularization (Ridge/Lasso) helps prevent overfitting
- Model interpretability is important for real-world applications

### Forest Fire Insights
- Most fires occur in July and August
- Strong correlation between temperature and fire occurrence
- FFMC (Fine Fuel Moisture Code) is a strong predictor
- Regional differences exist between Bejaia and Sidi-Bel Abbes

## Models

### Saved Models (in `model/` directory)

1. **Linear Regression Model** (`linear_regression_model.pkl`)
   - Standard linear regression for baseline predictions

2. **Ridge Regression Model** (`Ridge_linear_regression_model.pkl`)
   - Regularized model to handle multicollinearity

3. **Standard Scaler** (`scaler.pkl`)
   - Feature scaling for consistent predictions

## Contributing

Feel free to fork this repository and submit pull requests for improvements.

## License

This project is for educational purposes.

## Contact

For questions or feedback, please open an issue in the repository.

---

**Last Updated:** October 2025
