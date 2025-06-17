import math
import nltk
from collections import Counter, defaultdict
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
from nltk.stem import WordNetLemmatizer


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
    # Initialize the lemmatizer inside the function to avoid caching issues with Streamlit.
    lemmatizer = WordNetLemmatizer()

    # Process each raw document string in the input list.
    for doc in raw_docs:
        # Convert the entire document to lowercase.
        doc_lower = doc.lower()
        # Split the document into a list of word tokens.
        tokens = tokenizer.tokenize(doc_lower)
        # Remove any token that is a stop word.
        filtered_tokens = [token for token in tokens if token not in stop_words]
        # Reduce each token to its dictionary root form (lemma).
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        # Add the final list of tokens to our main list.
        processed_docs.append(lemmatized_tokens)
        
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
    # Create the vocabulary from the keys of the idf_scores dictionary,
    # which contains every term in the entire collection.
    vocab = list(idf_scores.keys())
    vocab_map = {term: i for i, term in enumerate(vocab)}
    
    # Convert the sparse original query vector (q0) to a dense numpy array.
    q0 = np.zeros(len(vocab))
    for term, weight in original_query_vector.items():
        if term in vocab_map:
            q0[vocab_map[term]] = weight

    # Calculate the centroid of all relevant documents.
    if relevant_doc_ids:
        sum_relevant = np.zeros(len(vocab))
        for doc_id in relevant_doc_ids:
            for term, weight in document_vectors[doc_id].items():
                 if term in vocab_map:
                    sum_relevant[vocab_map[term]] += weight
        centroid_relevant = sum_relevant / len(relevant_doc_ids)
    else:
        centroid_relevant = np.zeros(len(vocab))

    # Calculate the centroid of all non-relevant documents.
    if non_relevant_doc_ids:
        sum_non_relevant = np.zeros(len(vocab))
        for doc_id in non_relevant_doc_ids:
            for term, weight in document_vectors[doc_id].items():
                if term in vocab_map:
                    sum_non_relevant[vocab_map[term]] += weight
        centroid_non_relevant = sum_non_relevant / len(non_relevant_doc_ids)
    else:
        centroid_non_relevant = np.zeros(len(vocab))

    # Apply the Rocchio formula to get the modified query vector (qm).
    qm = alpha * q0 + beta * centroid_relevant - gamma * centroid_non_relevant
    qm[qm < 0] = 0
    
    # Convert the dense numpy vector back to a sparse dictionary.
    modified_query_vector = {term: qm[i] for term, i in vocab_map.items() if qm[i] > 0}
    
    return modified_query_vector
