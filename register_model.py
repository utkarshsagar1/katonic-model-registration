import mlflow
import mlflow.sklearn
import pickle
import os

# Assume your models are in the 'model/' subdirectory
CLASSIFICATION_MODEL_PATH = "model/classification.pkl"
CLUSTERING_MODEL_PATH = "model/clustering.pkl"

# --- MLflow Setup ---
# Katonic usually sets the TRACKING_URI automatically, but we start a run.
with mlflow.start_run(run_name="Meralco_Initial_Registration") as run:

    # 1. Load the model from your local .pkl file
    with open(CLASSIFICATION_MODEL_PATH, 'rb') as f:
        classification_model = pickle.load(f)

    # 2. Log the model artifact and register it
    # This logs it as a scikit-learn model 'flavor'
    mlflow.sklearn.log_model(
        sk_model=classification_model,
        artifact_path="classification_model_artifact",
        registered_model_name="Meralco_Classification_Model" # This is the name in the registry!
    )

    # 3. Repeat for the second model (Clustering)
    with open(CLUSTERING_MODEL_PATH, 'rb') as f:
        clustering_model = pickle.load(f)

    mlflow.sklearn.log_model(
        sk_model=clustering_model,
        artifact_path="clustering_model_artifact",
        registered_model_name="Meralco_Clustering_Model"
    )

print("Models registered in MLflow!")