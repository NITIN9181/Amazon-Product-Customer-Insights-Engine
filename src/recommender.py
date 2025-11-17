import pandas as pd

def get_recommendations(product_name, cosine_sim_matrix, products_dataframe, indices_lookup):
    """
    Finds the top 5 similar products given a product name.
    """
    try:
        # Get the index of the product that matches the name
        product_index = indices_lookup[product_name]
    except KeyError:
        return [f"Error: Product '{product_name}' not found in index."]
    except TypeError:
        # This handles a rare case where the product name might be duplicated
        # We'll just take the first index if it's a list
        product_index = indices_lookup[product_name].iloc[0]


    # Get the row of similarity scores for that product
    similarity_scores = list(enumerate(cosine_sim_matrix[product_index]))

    # Sort the products based on the similarity scores
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 6 most similar products (index 0 is the product itself)
    top_6_scores = sorted_scores[1:6] # Get top 5 recommendations

    # Get the product indices
    recommendation_indices = [i[0] for i in top_6_scores]

    # Return the names of the recommended products
    return products_dataframe['product_name'].iloc[recommendation_indices]