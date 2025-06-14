import math
from collections import Counter


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


def rank_documents(query_tokens, document_vectors, idf_scores):
    """
    Ranks documents against a pre-processed query using cosine similarity.
    We have a list of document vectors and a dictionary of IDF scores in artifacrt.
    Returns a list of tuples, where each tuple contains (document_id, similarity_score) in descending order of score.
    """
    
    # --- Step 1: Create the TF-IDF vector for the query ---
    query_vector = {}
    tf_counts = Counter(query_tokens)
    
    for term, tf in tf_counts.items():
        if term in idf_scores:
            # The query vector's weight for a term is its tf * the term's idf.
            query_vector[term] = tf * idf_scores[term]

    # --- Step 2: Calculate similarity scores for all documents ---
    scores = []
    # Enumerate through the document vectors to get both the doc_id and vector.
    for doc_id, doc_vector in enumerate(document_vectors):
        # Calculate the similarity between the query and the current document.
        similarity = cosine_similarity(query_vector, doc_vector)
        
        # We only need to store documents that have a non-zero similarity.
        if similarity > 0:
            # Store the document ID and its similarity score.
            scores.append((doc_id + 1, similarity))
    
    # --- Step 3: Sort the documents by similarity score ---
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores

