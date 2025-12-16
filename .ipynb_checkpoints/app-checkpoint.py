import joblib
import pandas as pd
from flask import Flask, request, jsonify
# --- FIX: Import WSGIMiddleware for Uvicorn/ASGI compatibility ---
from uvicorn.middleware.wsgi import WSGIMiddleware 

# --- 1. GLOBAL MODEL OBJECTS ---
# These global variables will hold the loaded models and be accessible by the /predict route
CLASSIFICATION_MODEL = None
CLUSTERING_MODEL = None

# --- 2. MODEL LOADING FUNCTION (Runs once at startup) ---
def load_models():
    """
    Loads both classification and clustering models into memory using joblib.
    """
    global CLASSIFICATION_MODEL, CLUSTERING_MODEL
    
    # NOTE: Set the correct path to your .pkl files
    model_path = 'model/' 
    
    try:
        print("--- Loading Meralco Classification Model ---")
        CLASSIFICATION_MODEL = joblib.load(f'{model_path}classification.pkl')
        
        print("--- Loading Meralco Clustering Model ---")
        CLUSTERING_MODEL = joblib.load(f'{model_path}clustering.pkl')
        
        print("Models loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load models. Check file paths and dependencies in requirements.txt. Error: {e}")
        # In a real production system, you might choose to exit here.
        
# Call the function to load the models immediately when the service starts
load_models() 

# --- 3. FLASK APPLICATION SETUP ---
# Initialize the Flask application instance
app = Flask(__name__)

# --- 4. PREDICTION ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint that receives JSON data and returns predictions.
    """
    if CLASSIFICATION_MODEL is None or CLUSTERING_MODEL is None:
        return jsonify({"error": "Models are not loaded. Check server logs."}), 503 # Service Unavailable

    try:
        # A. Get Input Data
        json_data = request.json
        if not json_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Convert JSON data (list of dictionaries) into a pandas DataFrame
        # This assumes the input data matches the features the model was trained on.
        input_df = pd.DataFrame(json_data)
        
        # B. PREPROCESSING (CRITICAL STEP - YOU MUST CUSTOMIZE THIS!)
        # *** WARNING: THIS IS A PLACEHOLDER. YOUR MODEL WILL FAIL WITHOUT THE CLIENT'S ***
        # *** EXACT PRE-PROCESSING STEPS (scaling, encoding, feature engineering).     ***
        processed_data = input_df # <-- REPLACE THIS WITH MERALCO'S LOGIC
        
        # C. Run Inference in Sequence
        # Step 1: Classification
        classification_result = CLASSIFICATION_MODEL.predict(processed_data).tolist()
        
        # Step 2: Clustering (Assumes it runs on the same input data for simplicity)
        clustering_result = CLUSTERING_MODEL.predict(processed_data).tolist()
        
        # D. Post-processing and Response
        output = {
            "status": "success",
            "classification_label": classification_result,
            "clustering_group_id": clustering_result,
            "message": "Prediction made using Meralco proprietary models."
        }
        
        return jsonify(output)
        
    except Exception as e:
        # Log the detailed error for debugging purposes in the Katonic logs
        print(f"An exception occurred during /predict: {e}")
        return jsonify({"error": f"Internal Prediction Error. Check logs for details. Exception: {str(e)}"}), 500

# --- 5. RUNNING THE APPLICATION ---

# ----------------------------------------------------------------------------------
# 🚀 THE FIX: EXPOSE THE APPLICATION AS AN ASGI-COMPATIBLE OBJECT 🚀
# ----------------------------------------------------------------------------------
# The 'app' object is a WSGI callable (Flask).
# The 'asgi_app' object is an ASGI callable (Wrapped Flask app).
# Uvicorn needs the ASGI callable.
asgi_app = WSGIMiddleware(app)

if __name__ == '__main__':
    # For local testing, you can continue to use the Flask development server,
    # but for production/Katonic, the server will use 'asgi_app'.
    app.run(host='0.0.0.0', port=5000, debug=True)