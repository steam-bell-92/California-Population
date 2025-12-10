import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
import joblib
import zipfile

class CaliforniaHousingPipeline:
    def __init__(self, model_path='model.joblib', scaler_path='scaler.joblib'):
        self.data = None
        self.df_clean = None
        self.model = LinearRegression()
        self.scaler = RobustScaler()
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, filepath):
        """Loads the dataset and adjusts the index."""
        print(f"Loading data from {filepath}...")
        self.data = pd.read_csv(filepath)
        self.data.index = range(1, len(self.data) + 1)
        print(f"Data loaded. Shape: {self.data.shape}")

    def feature_engineering(self):
        """Creates new features like room/household."""
        if self.data is None: raise ValueError("Data not loaded")
        
        self.data['room/household'] = self.data['total_rooms'] / self.data['households']
        print("Feature engineering complete: 'room/household' created.")

    def remove_outliers(self, columns):
        """
        Applies IQR filtering sequentially. 
        FIX: Ensures filters accumulate instead of overwriting.
        """
        

[Image of box plot anatomy]

        self.df_clean = self.data.copy()
        original_len = len(self.df_clean)

        for col in columns:
            Q1 = self.df_clean[col].quantile(0.25)
            Q3 = self.df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Apply filter
            self.df_clean = self.df_clean[
                (self.df_clean[col] >= lower_bound) & 
                (self.df_clean[col] <= upper_bound)
            ]

        print(f"Outliers removed. Rows dropped: {original_len - len(self.df_clean)}")
        print(f"New Shape: {self.df_clean.shape}")

    def visualize_eda(self):
        """Encapsulates all EDA plots."""
        df = self.df_clean if self.df_clean is not None else self.data
        
        # 1. KDE Plot
        plt.figure(figsize=(10, 6))
        sns.kdeplot(x='room/household', y='total_rooms', data=self.data, fill=True, thresh=0.0005, label='Original')
        sns.kdeplot(x='room/household', y='total_rooms', data=df, fill=True, thresh=0.0005, alpha=0.5, label='Cleaned')
        plt.title('Effect of Outlier Removal')
        plt.legend()
        plt.show()

        # 2. Mapbox Visualization
        fig = px.scatter_mapbox(
            df, lat="latitude", lon="longitude", 
            color="median_house_value", 
            color_continuous_scale="Portland", 
            size_max=15, zoom=5, height=600, 
            title="California Housing Prices", opacity=0.5
        )
        fig.update_layout(mapbox_style="open-street-map")
        fig.show()

        # 3. Correlation Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), cmap='coolwarm', cbar=True, annot=True)
        plt.title('Correlation Matrix')
        plt.show()

    def prepare_model_data(self, target_col='population', drop_cols=None):
        """
        Splits and scales the data.
        Allows flexibility in choosing the target variable.
        """
        if drop_cols is None:
            drop_cols = ['population', 'total_bedrooms', 'room/household']

        X = self.df_clean.drop(drop_cols, axis=1)
        y = self.df_clean[target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 
        # Scaling
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print(f"Data prepared. Target variable: {target_col}")

    def train_and_evaluate(self):
        """Trains the model and calculates R2 score."""
        print("Training Linear Regression...")
        self.model.fit(self.X_train_scaled, self.y_train)
        
        y_pred = self.model.predict(self.X_test_scaled)
        
        r2 = r2_score(self.y_test, y_pred)
        cv_score = cross_val_score(self.model, self.X_train_scaled, self.y_train, cv=10).mean()
        
        print(f"R² Score: {r2:.4f}")
        print(f"CV Score (10-fold): {cv_score:.4f}")
        
        return r2

    def save_artifacts(self, zip_name="California_app.zip"):
        """Saves model, scaler, and creates the deployment zip."""
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        with open("requirements.txt", "w") as f:
            f.write("gradio\nscikit-learn\nnumpy\njoblib\npandas")
            
        with zipfile.ZipFile(zip_name, "w") as zipf:
            zipf.write(self.model_path)
            zipf.write(self.scaler_path)
            zipf.write("requirements.txt")
            
        print(f"Artifacts saved and zipped to {zip_name}")

# --- Usage ---
if __name__ == "__main__":
    # Initialize
    pipeline = CaliforniaHousingPipeline()
    
    # 1. Load Data
    # Note: Replace path with your actual file location
    try:
        pipeline.load_data('/content/sample_data/california_housing_train.csv')
    except:
        print("File not found. Please ensure the path is correct.")

    if pipeline.data is not None:
        # 2. Feature Engineering
        pipeline.feature_engineering()
        
        # 3. Outlier Removal (using the list from your script)
        check_cols = ['median_income', 'median_house_value', 'households', 
                      'population', 'total_bedrooms', 'total_rooms', 
                      'housing_median_age', 'room/household']
        pipeline.remove_outliers(check_cols)
        
        # 4. Visualization (Optional)
        # pipeline.visualize_eda()
        
        # 5. Modeling
        # Note: I kept your logic of predicting 'population'. 
        # If you meant to predict house value, change target_col to 'median_house_value'
        pipeline.prepare_model_data(target_col='population')
        pipeline.train_and_evaluate()
        
        # 6. Save
        pipeline.save_artifacts()