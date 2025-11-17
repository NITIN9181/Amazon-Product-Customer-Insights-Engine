import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Define the file paths
RAW_DATA_PATH = 'data/00_raw/amazon.csv'
MODEL_DIR = 'models'
PRODUCTS_DF_PATH = os.path.join(MODEL_DIR, 'products_df.pkl')
COSINE_SIM_PATH = os.path.join(MODEL_DIR, 'cosine_sim.pkl')
INDICES_PATH = os.path.join(MODEL_DIR, 'indices.pkl')

def build_recommender_model():
    """
    Loads raw data, processes it, builds the model,
    and saves the artifacts to the 'models/' folder.
    """
    print("Starting model building process...")
    
    # --- All the logic from our notebook ---
    
    # 1. Load Data
    try:
        df = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {RAW_DATA_PATH}")
        return False
    
    print("Data loaded successfully.")

    # 2. Create 'parent_category'
    df['parent_category'] = df['category'].str.split('|').str[0]
    
    # 3. Create 'product_dna' (handle potential NaNs)
    df['about_product'] = df['about_product'].fillna('')
    df['product_name'] = df['product_name'].fillna('')
    df['parent_category'] = df['parent_category'].fillna('')
    
    df['product_dna'] = df['product_name'] + ' ' + \
                          df['parent_category'] + ' ' + \
                          df['about_product']
    
    # 4. Create Unique Product List
    product_cols = ['product_id', 'product_name', 'product_dna']
    products_df = df[product_cols].drop_duplicates(subset=['product_id'])
    products_df = products_df.reset_index(drop=True)
    print(f"Created unique product list: {len(products_df)} products.")

    # 5. Build TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products_df['product_dna'])
    print(f"TF-IDF matrix built with shape: {tfidf_matrix.shape}")

    # 6. Build Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_matrix)
    print(f"Cosine Similarity matrix built with shape: {cosine_sim.shape}")

    # 7. Create Indices
    indices = pd.Series(products_df.index, index=products_df['product_name'])
    
    # 8. Ensure 'models' directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 9. Save the 3 Artifacts
    pickle.dump(products_df, open(PRODUCTS_DF_PATH, 'wb'))
    pickle.dump(cosine_sim, open(COSINE_SIM_PATH, 'wb'))
    pickle.dump(indices, open(INDICES_PATH, 'wb'))

    print(f"Model building complete. 3 artifacts saved to '{MODEL_DIR}/'.")
    return True

if __name__ == "__main__":
    # This part allows you to run this file directly if needed
    # e.g., python src/data_processing.py
    build_recommender_model()