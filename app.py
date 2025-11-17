import streamlit as st
import pickle
import os
from src.recommender import get_recommendations # Import your function

# --- File Paths ---
PRODUCTS_DF_PATH = os.path.join('models', 'products_df.pkl')
COSINE_SIM_PATH = os.path.join('models', 'cosine_sim.pkl')
INDICES_PATH = os.path.join('models', 'indices.pkl')

# --- Load the Model Artifacts ---
# We use st.cache_resource to load these only once
@st.cache_resource
def load_model_artifacts():
    try:
        products_df = pickle.load(open(PRODUCTS_DF_PATH, 'rb'))
        cosine_sim = pickle.load(open(COSINE_SIM_PATH, 'rb'))
        indices = pickle.load(open(INDICES_PATH, 'rb'))
        return products_df, cosine_sim, indices
    except FileNotFoundError:
        st.error("Model files not found. Please run `run_training.py` first.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.stop()

# Load the data
products_df, cosine_sim, indices = load_model_artifacts()


# --- Streamlit App Layout ---

st.title('Amazon Product Recommendation Engine')
st.markdown("Find products similar to the one you like!")

# Create a dropdown menu to select a product
if products_df is not None:
    product_list = products_df['product_name'].values
    selected_product = st.selectbox(
        "Select a product to get recommendations:",
        product_list
    )

    # Create a button to get recommendations
    if st.button('Get Recommendations'):
        if selected_product:
            try:
                # Call our imported function!
                recommendations = get_recommendations(
                    product_name=selected_product,
                    cosine_sim_matrix=cosine_sim,
                    products_dataframe=products_df,
                    indices_lookup=indices
                )
                
                # Display the recommendations
                st.subheader('Here are your top 5 recommendations:')
                for product in recommendations:
                    st.write(product)
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")