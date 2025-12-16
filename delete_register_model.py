from mlflow import MlflowClient
import mlflow
import sys


# STEP 1: Set your MLflow Tracking Server URI
MLFLOW_TRACKING_URI = "http://asa-c25d20a2-8350-4950-8064-1d8a819e702c.kt-wast-app.svc.cluster.local:80"

# STEP 2: List models to delete
MODELS_TO_DELETE = [
    "Meralco_Classification_Model",
    "Meralco_Clustering_Model",
]


# ============================================
# VALIDATION
# ============================================

def validate_configuration():

    pass


# ============================================
# MAIN DELETION FUNCTION
# ============================================

def delete_registered_model_completely(client, model_name):
    """
    Delete a registered model and ALL its versions.

    Args:
        client: MlflowClient instance
        model_name (str): Name of the model to delete

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "=" * 70)
    print(f"DELETING MODEL: {model_name}")
    print("=" * 70)

    try:
        # Step 1: Get model details
        print("\n📋 Step 1: Fetching model details...")
        model = client.get_registered_model(model_name)
        print(f"✅ Found model: {model.name}")

        # Step 2: List all versions
        if len(model.latest_versions) == 0:
            print("   ℹ️  No versions found")
        else:
            print(f"   Found {len(model.latest_versions)} version(s):")
            for version in model.latest_versions:
                print(f"      - Version {version.version} (Stage: {version.current_stage})")

        # Step 3: Delete all versions
        if model.latest_versions:
            print("\n🗑️  Step 2: Deleting all versions...")
            for version in model.latest_versions:
                version_num = version.version
                print(f"   Deleting Version {version_num}...", end=" ")

                try:
                    client.delete_model_version(
                        name=model_name,
                        version=version_num
                    )
                    print("✅ Deleted")
                except Exception as e:
                    print(f"❌ Failed: {e}")
                    return False

        # Step 4: Delete the registered model itself
        print("\n🗑️  Step 3: Deleting registered model...")
        client.delete_registered_model(name=model_name)
        print(f"✅ Model '{model_name}' completely removed from registry!")

        return True

    except Exception as e:
        print(f"\n❌ ERROR deleting model '{model_name}':")
        print(f"   {type(e).__name__}: {e}")
        return False


# ============================================
# VERIFICATION FUNCTION
# ============================================

def verify_deletion(client, model_name):
    """
    Verify that a model has been deleted.

    Args:
        client: MlflowClient instance
        model_name (str): Name of the model to verify
    """
    try:
        model = client.get_registered_model(model_name)
        print(f"   ⚠️  WARNING: Model '{model_name}' still exists!")
        return False
    except Exception:
        print(f"   ✅ Confirmed: '{model_name}' no longer exists")
        return True


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution function."""

    print("\n" + "=" * 70)
    print("MLflow MODEL DELETION SCRIPT")
    print("=" * 70)

    # Step 1: Validate configuration
    print("\n🔍 Validating configuration...")
    validate_configuration()

    # Step 2: Connect to MLflow
    print(f"\n🔌 Connecting to MLflow server...")
    print(f"   URI: {MLFLOW_TRACKING_URI}")

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        # Test connection
        experiments = client.search_experiments(max_results=1)
        print(f"✅ Connected successfully!")

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check if MLflow server URL is correct")
        print("   2. Verify server is running")
        print("   3. Check network connectivity")
        sys.exit(1)

    # Step 3: Delete each model
    print(f"\n🗑️  Deleting {len(MODELS_TO_DELETE)} model(s)...")

    results = {}

    for model_name in MODELS_TO_DELETE:
        success = delete_registered_model_completely(client, model_name)
        results[model_name] = success

    # Step 4: Verify deletions
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    print("\n🔍 Verifying deletions...")
    for model_name in MODELS_TO_DELETE:
        print(f"\n   Checking '{model_name}'...")
        verify_deletion(client, model_name)

    # Step 5: Summary
    print("\n" + "=" * 70)
    print("DELETION SUMMARY")
    print("=" * 70)

    successful = sum(1 for success in results.values() if success)
    failed = len(results) - successful

    print(f"\n✅ Successfully deleted: {successful}/{len(MODELS_TO_DELETE)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(MODELS_TO_DELETE)}")
        print("\nFailed models:")
        for model_name, success in results.items():
            if not success:
                print(f"   - {model_name}")

    # Step 6: Next steps
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\n1. Go to MLflow UI → Models tab")
    print("2. Refresh the page")
    print("3. Verify the following models are gone:")
    for model_name in MODELS_TO_DELETE:
        print(f"   - {model_name}")
    print("\n✅ Deletion process completed!")
    print("=" * 70 + "\n")


# ============================================
# RUN THE SCRIPT
# ============================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)