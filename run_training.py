import os
from src.data_processing import build_recommender_model

if __name__ == "__main__":
    # Ensure the 'models' directory exists before training
    os.makedirs('models', exist_ok=True)
    
    print("--- Starting Model Training ---")
    
    # Run the main processing and training function
    success = build_recommender_model()
    
    if success:
        print("--- Model Training Successful ---")
    else:
        print("--- Model Training Failed ---")