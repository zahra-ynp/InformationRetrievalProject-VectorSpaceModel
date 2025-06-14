import pickle

# We will import the functions from the retrieval module we've been building
from retrieval import rank_documents
from retrieval import preprocess_text

def load_artifacts():
    """
    Loads the processed data artifacts from disk.
    """
    print("--- Loading system artifacts... ---")
    try:
        with open("./artifacts/cranfield_vectors.pkl", 'rb') as f:
            doc_vectors = pickle.load(f)

        with open("./artifacts/cranfield_idf.pkl", 'rb') as f:
            idf_scores = pickle.load(f)

        with open("./artifacts/cranfield_raw_docs.pkl", 'rb') as f:
            raw_docs = pickle.load(f)
            
        print("Artifacts loaded successfully.\n")
        return doc_vectors, idf_scores, raw_docs
    except FileNotFoundError:
        print("Error: Could not find artifact files.")
        print("Please make sure you have run the '1_Index_Builder.ipynb' notebook to generate them.")
        return None, None, None

def main():
    """
    The main function to run the application.
    """
    # Load all the necessary data structures into memory
    document_vectors, idf_scores, raw_documents = load_artifacts()

    # If loading failed, exit the program
    if document_vectors is None:
        return

    # Start the interactive loop
    while True:
        # Get user input from the command line
        query = input("Enter your query (or type 'exit' to quit):\n> ")

        # Check for the exit command
        if query.lower() == 'exit':
            break

        # --- Process the query ---
        # 1. Put the query in a list to match the preprocessor's expected input
        query_as_list = [query]
        # 2. Call the preprocessor function
        processed_query_list = preprocess_text(query_as_list)
        # 3. Get the list of tokens from the result
        query_tokens = processed_query_list[0]
        
        # --- Rank documents ---
        # Get the ranked list of (doc_id, score) tuples
        ranked_results = rank_documents(query_tokens, document_vectors, idf_scores)

        # --- Display the results ---
        print(f"\n--- Top 5 Results for '{query}' ---")
        if not ranked_results:
            print("No matching documents found.")
        else:
            # Use enumerate to get the rank number, starting from 1
            for rank, (doc_id, score) in enumerate(ranked_results[:5], 1):
                print(f"\nRank {rank}: Document ID: {doc_id} (Score: {score:.4f})")
                # Retrieve and print the first 200 characters of the raw document
                # doc_id is 1-based, so we access the list with doc_id - 1
                doc_text = raw_documents[doc_id - 1]
                print(f"   Preview: {doc_text[:200]}...")
        
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
