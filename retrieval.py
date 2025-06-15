import math
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import numpy as np 


# You only need to run these lines once to download the necessary NLTK packages.
# try:
#     stopwords.words('english')
# except LookupError:
#     nltk.download('stopwords')
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')
# try:
#     nltk.data.find('corpora/wordnet')
# except LookupError:
#     nltk.download('wordnet')
# -----------------------------------

def preprocess_text(raw_docs, method='lemmatize'):
    """
    Takes a list of raw document strings and applies all preprocessing steps.

    Args:
        raw_docs (list of str): The list of unprocessed document texts.
        method (str): The word reduction method to use. Can be 'lemmatize' (default)
                      or 'stem'.

    Returns:
        A list of lists, where each inner list contains the processed tokens
        of a single document.
    """
    # Initialize lists and objects for preprocessing.
    processed_docs = []
    
    # 1. TOKENIZATION AND NORMALIZATION (LOWERCASE, PUNCTUATION REMOVAL)
    # The tokenizer will split the document text into a list of words.
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    
    # 2. STOP WORD REMOVAL
    # Load the set of English stop words. Using a set provides fast lookups.
    stop_words = set(stopwords.words('english'))
    
    # 3. STEMMING / LEMMATIZATION
    # Initialize the chosen processor.
    if method == 'lemmatize':
        processor = WordNetLemmatizer()
        process_func = processor.lemmatize
    elif method == 'stem':
        processor = PorterStemmer()
        process_func = processor.stem
    else:
        raise ValueError("Method must be 'lemmatize' or 'stem'")

    # Process each document in the input list.
    for doc in raw_docs:
        # Lowercase the document text.
        doc = doc.lower()
        
        # Use the tokenizer to get a list of alphabetic tokens.
        tokens = tokenizer.tokenize(doc)
        
        # Filter out stop words from the token list.
        filtered_tokens = [token for token in tokens if token not in stop_words]
        
        # Apply the chosen processing (lemmatization or stemming) to each token.
        processed_tokens = [process_func(token) for token in filtered_tokens]
        
        # Add the final list of processed tokens to our main list.
        processed_docs.append(processed_tokens)
        
    return processed_docs


def create_query_vector(query_tokens, idf_scores):
    """Creates a TF-IDF vector for a given list of query tokens."""
    # Create a dictionary to hold the query's vector representation.
    query_vector = {}
    # Use collections.Counter to efficiently count the occurrences of each token
    # in the query. This gives us the raw Term Frequency (TF).
    tf_counts = Counter(query_tokens)
    # Iterate through each unique term and its frequency in the query.
    for term, tf in tf_counts.items():
        # We only consider terms that exist in our main vocabulary (and thus have an IDF score).
        if term in idf_scores:
            # The weight of a term in the query vector is its local TF
            # multiplied by its global IDF score.
            query_vector[term] = tf * idf_scores[term]
    return query_vector


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


def rank_documents(query_vector, document_vectors):
    """
    Ranks documents against a pre-processed query using cosine similarity.
    We have a list of document vectors and a dictionary of IDF scores in artifacrt.
    Returns a list of tuples, where each tuple contains (document_id, similarity_score) in descending order of score.
    """

    # --- Step 1: Calculate similarity scores for all documents ---
    scores = []
    # Enumerate through the document vectors to get both the doc_id and vector.
    for doc_id, doc_vector in enumerate(document_vectors):
        # Calculate the similarity between the query and the current document.
        similarity = cosine_similarity(query_vector, doc_vector)
        
        # We only need to store documents that have a non-zero similarity.
        if similarity > 0:
            # Store the document ID and its similarity score.
            scores.append((doc_id + 1, similarity))
    
    # --- Step 2: Sort the documents by similarity score ---
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores



def apply_rocchio_feedback(original_query_vector, relevant_doc_ids, non_relevant_doc_ids, document_vectors, alpha=1.0, beta=0.75, gamma=0.15):
    """
    Modifies a query vector using the Rocchio algorithm .

    This function adjusts the original query vector by moving it closer to the
    centroid of relevant documents and away from the centroid of non-relevant ones.
    """
    # For vector arithmetic, it's easier to work with dense numpy arrays.
    # First, create a complete vocabulary list and a map from term to index.
    vocab = list(document_vectors[0].keys())
    vocab_map = {term: i for i, term in enumerate(vocab)}
    
    # Create a dense numpy array of zeros for the original query vector (q0).
    q0 = np.zeros(len(vocab))
    # Populate it with weights from the sparse original_query_vector.
    for term, weight in original_query_vector.items():
        if term in vocab_map:
            q0[vocab_map[term]] = weight

    # --- Calculate centroid of relevant documents ---
    # The centroid is the average vector of all relevant documents.
    if relevant_doc_ids:
        # Sum the vectors of all specified relevant documents.
        sum_relevant = np.zeros(len(vocab))
        for doc_id in relevant_doc_ids:
            for term, weight in document_vectors[doc_id].items():
                 if term in vocab_map:
                    sum_relevant[vocab_map[term]] += weight
        # Divide by the number of relevant documents to get the average (centroid).
        centroid_relevant = sum_relevant / len(relevant_doc_ids)
    else:
        centroid_relevant = np.zeros(len(vocab))

    # --- Calculate centroid of non-relevant documents ---
    # The centroid is the average vector of all non-relevant documents.
    if non_relevant_doc_ids:
        sum_non_relevant = np.zeros(len(vocab))
        for doc_id in non_relevant_doc_ids:
            for term, weight in document_vectors[doc_id].items():
                if term in vocab_map:
                    sum_non_relevant[vocab_map[term]] += weight
        centroid_non_relevant = sum_non_relevant / len(non_relevant_doc_ids)
    else:
        centroid_non_relevant = np.zeros(len(vocab))

    # --- Apply the Rocchio formula ---
    # q_modified = alpha * q_original + beta * centroid_relevant - gamma * centroid_non_relevant
    qm = alpha * q0 + beta * centroid_relevant - gamma * centroid_non_relevant

    # أegative weights are not allowed, so we set them to 0.
    qm[qm < 0] = 0

    # Convert the dense numpy vector back to a sparse dictionary format for efficiency,
    # only including terms with a non-zero weight.
    modified_query_vector = {term: qm[i] for term, i in vocab_map.items() if qm[i] > 0}
    
    return modified_query_vector
