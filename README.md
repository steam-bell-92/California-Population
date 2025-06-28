# 🏡 California Population - EDA + Linear Regression Modeling

This project explores and models the **California Housing dataset** using both **Exploratory Data Analysis (EDA)** and **Supervised Machine Learning (Linear Regression)**.

## 📂 Dataset

- **Source:** `california_housing_train.csv` (Google Colab sample data)
- Contains key housing features like:
  - `median_house_value`, `median_income`
  - `total_rooms`, `total_bedrooms`
  - `population`, `households`
  - `housing_median_age`, `latitude`, `longitude`

## 🔍 Project Highlights

### 🔹 Exploratory Data Analysis (EDA)
- Summary statistics and data structure
- Missing values and distribution plots
- Correlation heatmaps

### 🔹 Machine Learning
- **Target:** `median_house_value`
- **Model:** `LinearRegression` from `sklearn`
- **Preprocessing:**
  - Feature scaling with `RobustScaler`
  - Train/test split
- **Evaluation:**
  - Predictions on test set
  - R² score for performance

## 🛠️ Technologies Used

- `Python`
- `Numpy`
- `Pandas`
- `Matplotlib.pyplot`
- `Seaborn`
- `scikit-learn`
- `Gardio`
- `Joblib`
- `Hugging Face Space`

