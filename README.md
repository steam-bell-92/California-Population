# 🏡 California Housing: Population & Price Prediction

This project explores California housing data through **EDA** and builds a **Linear Regression model** to predict population based on 6 major factors.

- 📈 **`R² Score (Test Set)` ≈ 0.8698**
- 🔁 **`Mean R² (10-Fold CV)` ≈ 0.8503**

⭐ If you find this useful, consider giving it a star!

> The web app is deployed using **Gradio** and hosted on **Hugging Face Spaces**<br>
> Note: This dataset is generally used for predicting house prices

---

## 🧰 Tech Stack

| Tool / Library          | Purpose                                 |
|-------------------------|-----------------------------------------|
| **`NumPy`**             | Numerical operations                    |
| **`Pandas`**            | Data loading, wrangling, and analysis   |
| **`Matplotlib`**        | Data visualization                      |
| **`Seaborn`**           | Statistical plots and heatmaps          |
| **`scikit-learn`**      | Model training, evaluation, scaling     |
| **`RobustScaler`**      | Scaling features resistant to outliers  |
| **`Joblib`**            | Saving/loading trained model            |
| **`Gradio`**            | Web-based interface                     |
| **`Hugging Face Spaces`** | Free model hosting                   |

---

## 🚀 Try the App Live

👉 [**Click here to use the live app on Hugging Face**](https://huggingface.co/spaces/steam-bell-92/California_population)  

---

## 📈 Features Used in Prediction

- 🧍‍♂️ Population  
- 🛏️ Total bedrooms  
- 💰 Median income  
- 🧱 Housing median age  
- 🌎 Latitude & Longitude  

---

## 🛠️ How It Works

1. Dataset loaded from Colab sample files.
2. EDA performed to understand distribution, correlation, and geography.
3. Data scaled using `RobustScaler` for better regression stability.
4. A `LinearRegression` model is trained and evaluated.
5. Final model deployed as an interactive app via Gradio.

---

## 📁 Project Structure

```
California-Population/
├── app.py                                         🔹 Gradio interface
├── model.joblib                                   🔹 Trained regression model
├── scaler.joblib                                  🔹 Store fitted scaler object 
├── California_housing_train.ipynb                 🔹 EDA + model notebook
├── requirements.txt                               🔹 Python dependencies
└── README.md                                      🔹 This file!
```
---

## 👤 Author

Anuj Kulkarni aka steam-bell-92
