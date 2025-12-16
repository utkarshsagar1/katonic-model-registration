import pickle
import numpy as np

def loadmodel(logger):
    """Load the model - called once during deployment startup"""
    try:
        logger.info("Loading model...")
        with open('model/model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def preprocessing(data, logger):
    """Optional: preprocess input data"""
    try:
        # If no preprocessing needed, return False
        if not data:
            return False
        
        # Convert to numpy array if needed
        processed_data = np.array(data['data'])
        logger.info(f"Data preprocessed: shape {processed_data.shape}")
        return processed_data
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        return False

def predict(data, model, logger):
    """Make predictions"""
    try:
        logger.info("Making predictions...")
        predictions = model.predict(data)
        logger.info(f"Predictions generated: {len(predictions)}")
        
        # Return in expected format
        return {
            "predictions": predictions.tolist(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return {"error": str(e), "status": "failed"}