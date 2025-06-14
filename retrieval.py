import math

def cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two vectors.
    The return value is the cosine similarity score (float) between 0 and 1.
    """
    # --- Calculate Dot Product ---
    # We can efficiently calculate the dot product by iterating through the
    # terms in the smaller vector
    if len(vec1) > len(vec2):
        vec1, vec2 = vec2, vec1
        
    dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in vec1)

    # --- Calculate Euclidean Norms ---
    norm_vec1 = math.sqrt(sum(val**2 for val in vec1.values()))
    norm_vec2 = math.sqrt(sum(val**2 for val in vec2.values()))

    # --- Calculate Cosine Similarity ---
    if norm_vec1 > 0 and norm_vec2 > 0:
        return dot_product / (norm_vec1 * norm_vec2)
    else:
        return 0.0
