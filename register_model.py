import mlflow
import mlflow.sklearn
import joblib 
import os

# ----------------------------------------------------------------------------------
# CRITICAL STEP 1: Set the Tracking URI to the Tenant-Level MLflow
# You must replace this placeholder with your actual URI.
mlflow.set_tracking_uri("http://asa-532836a6-18bc-4f86-8be4-dc52ae067fcb.kt-wast-app.svc.cluster.local:80")
# ----------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------
# CRITICAL STEP 2: Set the Experiment Name
# This ensures the runs show up under a specific experiment, not "Default".
mlflow.set_experiment("Meralco_Model_Registration")
# ----------------------------------------------------------------------------------


# Assume your models are in the 'model/' subdirectory
CLASSIFICATION_MODEL_PATH = "model/classification.pkl"
CLUSTERING_MODEL_PATH = "model/clustering.pkl"

# --- MLflow Setup ---
# This run will now be logged under the "Meralco_Model_Registration" experiment
with mlflow.start_run(run_name="Initial_Registration_v2") as run: # Changed run_name slightly for clarity

    # 1. Load the classification model from your local .pkl file using joblib
    try:
        classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
    except Exception as e:
        print(f"Error loading classification model with joblib: {e}")
        raise e

    # 2. Log the model artifact and register it
    mlflow.sklearn.log_model(
        sk_model=classification_model,
        artifact_path="classification_model_artifact",
        registered_model_name="Meralco_Classification_Model"
    )

    # 3. Repeat for the second model (Clustering)
    try:
        clustering_model = joblib.load(CLUSTERING_MODEL_PATH)
    except Exception as e:
        print(f"Error loading clustering model with joblib: {e}")
        raise e

    mlflow.sklearn.log_model(
        sk_model=clustering_model,
        artifact_path="clustering_model_artifact",
        registered_model_name="Meralco_Clustering_Model"
    )

print("Models registered in MLflow!")