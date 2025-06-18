import math
import nltk
from collections import Counter
import numpy as np

# --- NLTK Data Downloads ---
def download_nltk_data():
    """
    Checks for and downloads required NLTK data packages.
    
    This function ensures that the necessary data for tokenization, stop word
    removal, and lemmatization is available on the system before the main
    functions are called.
    """
    # A dictionary mapping the NLTK package ID to its path for checking.
    required_packages = {
        'stopwords': 'corpora/stopwords',
        'punkt': 'tokenizers/punkt',
        'wordnet': 'corpora/wordnet.zip',
        'omw-1.4': 'corpora/omw-1.4.zip' 
    }
    # Loop through the required packages.
    for pkg_id, path in required_packages.items():
        try:
            # Check if the data is already present.
            nltk.data.find(path)
        except LookupError:
            # If the data is not found, download it.
            nltk.download(pkg_id)

# Call the function once when this module is first imported.
download_nltk_data()

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def preprocess_text(raw_docs):
    """
    Takes a list of raw document strings and applies all preprocessing steps.
    """
    # Initialize a list to hold the final lists of processed tokens.
    processed_docs = []
    # Use a regular expression tokenizer to extract words.
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    # Load the set of English stop words for fast lookup.
    stop_words = set(stopwords.words('english'))
    # Initialize the stemmer inside the function to avoid caching issues with Streamlit.
    stemmer = PorterStemmer()

    # Process each raw document string in the input list.
    for doc in raw_docs:
        # Convert the entire document to lowercase.
        doc_lower = doc.lower()
        # Split the document into a list of word tokens.
        tokens = tokenizer.tokenize(doc_lower)
        # Remove any token that is a stop word.
        filtered_tokens = [token for token in tokens if token not in stop_words]
        # use the Porter Stemmer to stem each token.
        stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
        # Add the final list of tokens to our main list.
        processed_docs.append(stemmed_tokens)
        
    return processed_docs


def cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two sparse vectors (dictionaries).
    """
    # For efficiency, iterate through the smaller vector's keys.
    if len(vec1) > len(vec2):
        vec1, vec2 = vec2, vec1
    # Calculate the dot product
    dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in vec1)
    
    # Calculate the magnitude (Euclidean norm) of each vector.
    norm_vec1 = math.sqrt(sum(val**2 for val in vec1.values()))
    norm_vec2 = math.sqrt(sum(val**2 for val in vec2.values()))
    
    # Handle the case of zero-magnitude vectors to avoid division by zero.
    if norm_vec1 > 0 and norm_vec2 > 0:
        return dot_product / (norm_vec1 * norm_vec2)
    else:
        return 0.0

def create_query_vector(query_tokens, idf_scores):
    """
    Creates a TF-IDF vector for a given list of query tokens.
    """
    # The query vector is a dictionary mapping terms to their TF-IDF scores.
    query_vector = {}
    # Use Counter to get the term frequency (TF) for each token in the query.
    tf_counts = Counter(query_tokens)
    # Calculate the TF-IDF weight for each term in the query.
    for term, tf in tf_counts.items():
        if term in idf_scores: 
            query_vector[term] = tf * idf_scores[term]
    return query_vector


def rank_documents(query_vector, document_vectors):
    """
    Ranks documents against a query vector using cosine similarity.
    """
    # A list to store tuples of (doc_id, similarity_score).
    scores = []
    # Iterate through all document vectors, getting their ID and vector.
    for doc_id, doc_vector in enumerate(document_vectors):
        # Calculate the similarity between the query and the current document.
        similarity = cosine_similarity(query_vector, doc_vector)
        # Only store documents with a similarity score greater than 0.
        if similarity > 0:
            # Append the 1-based doc ID and its score.
            scores.append((doc_id + 1, similarity))
    # Sort the list of documents by score in descending order.
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def apply_rocchio_feedback(original_query_vector, relevant_doc_ids, non_relevant_doc_ids, document_vectors, idf_scores, alpha=1.0, beta=0.75, gamma=0.15):
    """
    Modifies a query vector using the Rocchio algorithm.
    """
    # Create the full vocabulary from the keys of the idf_scores dictionary.
    # This ensures we have every term that exists in the entire collection.
    vocab = list(idf_scores.keys())

    # Create a mapping from each term to a unique index (0, 1, 2, ...).
    # This gives each term a specific position in our new dense vectors.
    vocab_map = {term: i for i, term in enumerate(vocab)}

    # Convert the sparse original query vector (q0) to a dense numpy array.
    # First, create a giant vector of all zeros.
    q0 = np.zeros(len(vocab))
    # Then, loop through the original sparse vector (dictionary) and place its
    # weights into the correct positions in the new dense vector.
    for term, weight in original_query_vector.items():
        if term in vocab_map:
            q0[vocab_map[term]] = weight

    if relevant_doc_ids:
        # Create another giant vector of zeros to act as an accumulator.
        sum_relevant = np.zeros(len(vocab))
        
        # Loop through each provided relevant document ID.
        for doc_id in relevant_doc_ids:
            # For each relevant document, loop through its terms and weights.
            for term, weight in document_vectors[doc_id].items():
                if term in vocab_map:
                    # Add the weight of the term to the correct position in our accumulator.
                    sum_relevant[vocab_map[term]] += weight
                    
        # Divide the sum of all vectors by the number of documents to get the average.
        centroid_relevant = sum_relevant / len(relevant_doc_ids)
    else:
        # If no relevant documents were given, the centroid is just a vector of zeros.
        centroid_relevant = np.zeros(len(vocab))


    # This part only runs if the user provided a list of non-relevant document IDs.
    if non_relevant_doc_ids:
        # Create a vector of zeros to accumulate the weights.
        sum_non_relevant = np.zeros(len(vocab))
        
        # Loop through each provided non-relevant document ID.
        for doc_id in non_relevant_doc_ids:
            # For each non-relevant document, loop through its terms and weights.
            for term, weight in document_vectors[doc_id].items():
                if term in vocab_map:
                    # Add the weight of the term to our accumulator.
                    sum_non_relevant[vocab_map[term]] += weight
                    
        # Divide the sum by the number of non-relevant documents to get the average vector.
        centroid_non_relevant = sum_non_relevant / len(non_relevant_doc_ids)
    else:
        # If no non-relevant documents were given, the centroid is a vector of zeros.
        centroid_non_relevant = np.zeros(len(vocab))


    # Apply the formula using the dense numpy vectors we created.
    qm = alpha * q0 + beta * centroid_relevant - gamma * centroid_non_relevant

    # This line finds any negative values in the new vector and sets them to 0.
    qm[qm < 0] = 0

    # This creates a new dictionary containing only the terms that have a
    # final weight greater than 0, making it efficient again.
    modified_query_vector = {term: qm[i] for term, i in vocab_map.items() if qm[i] > 0}
        
    return modified_query_vector

