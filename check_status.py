from mlflow import MlflowClient
import mlflow

# Client sets their MLflow URI
mlflow.set_tracking_uri("http://asa-c25d20a2-8350-4950-8064-1d8a819e702c.kt-wast-app.svc.cluster.local:80")

client = MlflowClient()

models_to_check = [
    "Meralco_Classification_Model",
    "Meralco_Clustering_Model"
]

print("Checking model details...\n")

for model_name in models_to_check:
    try:
        model = client.get_registered_model(model_name)
        print(f"Model: {model_name}")

        for version in model.latest_versions:
            print(f"  Version {version.version}:")
            print(f"    Stage: {version.current_stage}")  # ← THIS IS KEY!
            print(f"    Status: {version.status}")
            print()

    except Exception as e:
        print(f"Error checking {model_name}: {e}\n")