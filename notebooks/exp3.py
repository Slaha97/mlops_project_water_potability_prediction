# Import necessary libraries for data handling, machine learning, tracking, and visualization
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import os
from datetime import datetime

# Initialize DagsHub and set up MLflow experiment tracking
dagshub.init(repo_owner='Slaha97', repo_name='mlops_project_water_potability_prediction', mlflow=True)
mlflow.set_experiment("Experiment 3")  # Updated experiment name
mlflow.set_tracking_uri("https://dagshub.com/Slaha97/mlops_project_water_potability_prediction.mlflow")  # URL to track the experiment

# Load the dataset from a CSV file
data = pd.read_csv("C:\\Users\\subha\\OneDrive\\Desktop\\Water_potability_prediction\\water_potability.csv")

# Split the dataset into training and test sets (80% training, 20% testing)
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)

# Function to fill missing values with the median value of each column
def fill_missing_with_mean(df):
    for column in df.columns:
        if df[column].isnull().any():
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)
    return df

# Fill missing values in both the training and test datasets using the median
train_processed_data = fill_missing_with_mean(train_data)
test_processed_data = fill_missing_with_mean(test_data)

# Separate features (X) and target (y) for training
X_train = train_processed_data.drop(columns=["Potability"], axis=1)  # Features
y_train = train_processed_data["Potability"]  # Target variable
X_test = test_processed_data.drop(columns=["Potability"], axis=1)  # Features for testing
y_test = test_processed_data["Potability"]  # Target variable for testing

# Define the classifiers to be used
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Classifier": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "XG Boost": XGBClassifier()
}

# Ensure the directory for saving models and plots exists
artifact_dir = "artifacts"
os.makedirs(artifact_dir, exist_ok=True)

# Start a new MLflow run for tracking the experiment
for model_name, model in classifiers.items():
    # Start the MLflow run
    with mlflow.start_run(run_name=model_name):
        
        # Train the model
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Calculate performance metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics to MLflow for tracking
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1-score", f1)

        # Log the classifier name as a parameter
        mlflow.log_param("classifier", model_name)

        # Log the trained model to MLflow
        mlflow.sklearn.log_model(model, f"model_{model_name.replace(' ', '_').lower()}")

        # Generate and log the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {model_name}")

        # Save the confusion matrix plot as a PNG file
        cm_image_path = os.path.join(artifact_dir, f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png")
        plt.savefig(cm_image_path)
        plt.close()  # Close the plot to free memory
        mlflow.log_artifact(cm_image_path)  # Log the confusion matrix image as an artifact

        # Save the trained model using pickle
        model_pickle_path = os.path.join(artifact_dir, f"{model_name.replace(' ', '_').lower()}.pkl")
        with open(model_pickle_path, 'wb') as f:
            pickle.dump(model, f)
        mlflow.log_artifact(model_pickle_path)  # Log the pickle file as an artifact

        # Log source code (optional)
        mlflow.log_artifact(__file__)

        # Set tags in MLflow to store additional metadata
        mlflow.set_tag("author", "datathinkers")
        mlflow.set_tag("model", model_name)

        # Print out the performance metrics for reference
        print(f"Results for {model_name}:")
        print(f"Accuracy: {acc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
        print("-" * 50)

print("All models have been trained, and metrics/artifacts have been logged.")