# California Housing Population Prediction

## Overview

This project analyzes the California housing dataset, performs exploratory data analysis, and trains a linear regression model to predict population from housing-related features.

## Problem Statement

The goal is to understand how housing attributes relate to population and to build a regression model that estimates population from the available features.

## Dataset

Dataset source: placeholder to be documented by the project owner.

The standardized data input path is `data/raw/california_housing_train.csv`.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- Joblib
- Gradio

## Project Structure

- `data/raw/` stores the source dataset.
- `data/processed/` is reserved for cleaned or derived datasets.
- `notebooks/` contains the analysis notebook.
- `src/` contains the reusable training script.
- `models/` stores the trained model and scaler artifacts.
- `images/` stores project visuals used in documentation.
- `WEBSITE/` contains the Gradio application entrypoint.

## Workflow

1. Load the housing dataset from a configurable local path.
2. Perform exploratory data analysis and feature inspection.
3. Create the engineered `room/household` feature.
4. Remove outliers using the existing IQR-based workflow.
5. Split the data, scale features with `RobustScaler`, and train a `LinearRegression` model.
6. Evaluate the model with test-set R² and 10-fold cross-validation.
7. Save the trained artifacts for the Gradio app.

## Results

- Test R² score: approximately 0.8698
- Mean 10-fold cross-validation R²: approximately 0.8503

## Future Improvements

- Document the dataset source in this README.
- Add a reproducible data download or preparation step.
- Expand evaluation with residual analysis and error diagnostics.
- Add automated checks for the training and inference pipeline.