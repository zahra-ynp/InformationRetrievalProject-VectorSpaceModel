import streamlit as st
import pickle

# Import the necessary functions from our custom 'retrieval.py' module.
from retrieval import create_query_vector, rank_documents, preprocess_text, apply_rocchio_feedback

# --- Caching Mechanism ---
# @st.cache_resource tells Streamlit to run this function only once and store the
# result in memory. On subsequent runs, it will use the cached result.
@st.cache_resource
def load_artifacts():
    """
    Loads the processed data artifacts (vectors, IDF scores, raw text) from disk.
    This function is cached so it only runs on the first page load.
    """
    print("--- Loading system artifacts (this should only run once)... ---")
    try:
        # Open the saved artifact files.
        with open("./artifacts/cranfield_vectors.pkl", 'rb') as f:
            doc_vectors = pickle.load(f)

        with open("./artifacts/cranfield_idf.pkl", 'rb') as f:
            idf_scores = pickle.load(f)

        with open("./artifacts/cranfield_raw_docs.pkl", 'rb') as f:
            raw_docs = pickle.load(f)
            
        print("Artifacts loaded successfully.")
        return doc_vectors, idf_scores, raw_docs
    except FileNotFoundError:
        # If the files aren't found, display an error message in the app.
        st.error("Error: Could not find artifact files. Please run '1_Index_Builder.ipynb' first.")
        return None, None, None

def main():
    """
    The main function to define the structure and logic of the Streamlit web application.
    """
    # --- Page Configuration ---
    st.set_page_config(
        page_title="Vector Space Model Search Engine",
        page_icon="🔎",
        layout="wide"
    )

    # --- Load Data ---
    document_vectors, idf_scores, raw_documents = load_artifacts()

    # If loading failed, stop the application gracefully.
    if document_vectors is None:
        return

    # --- UI Elements ---
    st.title("🔎 Vector Space Model Search Engine")
    st.markdown("This app uses a Vector Space Model with TF-IDF weighting to retrieve documents from the Cranfield collection.")

    # Create a text input box for the user to enter their query.
    query = st.text_input("Enter your query:", "")
    
    # Add a checkbox to allow the user to enable pseudo-relevance feedback.
    use_feedback = st.checkbox("Enable Pseudo-Relevance Feedback (assumes top 3 are relevant)")

    # --- Search Logic ---
    # This block of code only runs if the user has typed something into the input box.
    if query:
        # 1. Preprocess the user's query
        query_tokens = preprocess_text([query])[0]
        
        # 2. Create the initial TF-IDF vector for the query.
        query_vector = create_query_vector(query_tokens, idf_scores)
        
        # --- Run Search and Display Results ---
        if use_feedback:
            # --- FEEDBACK PATH ---
            # a. Get initial ranking to find top documents
            initial_results = rank_documents(query_vector, document_vectors)
            
            # b. Get the IDs of the top 3 documents to use for feedback
            # We subtract 1 to convert from 1-based doc ID to 0-based list index
            assumed_relevant_ids = [doc_id - 1 for doc_id, score in initial_results[:3]]
            
            # c. Apply the Rocchio formula to create a new, improved query vector
            if assumed_relevant_ids:
                modified_query_vector = apply_rocchio_feedback(
                    original_query_vector=query_vector,
                    relevant_doc_ids=assumed_relevant_ids,
                    non_relevant_doc_ids=[], # No non-relevant docs in pseudo-feedback
                    document_vectors=document_vectors,
                    idf_scores=idf_scores,
                    alpha=1.0, beta=0.75, gamma=0.0 # Gamma is 0 for pseudo-feedback
                )
   
                
                # d. Re-rank all documents using the new, modified query vector
                ranked_results = rank_documents(modified_query_vector, document_vectors)
                st.subheader(f"Top 10 Results (after feedback) for: \"{query}\"")
            else:
                # If the initial search returned no results, we can't do feedback
                ranked_results = []
                st.subheader(f"Top 10 Results for: \"{query}\"")

        else:
            # --- NO FEEDBACK PATH ---
            # Rank all documents against the original query vector
            ranked_results = rank_documents(query_vector, document_vectors)
            st.subheader(f"Top 10 Results for: \"{query}\"")

        # --- Display Results ---
        if not ranked_results:
            st.warning("No matching documents found.")
        else:
            # Loop through the top 10 results and display them in expanders.
            for rank, (doc_id, score) in enumerate(ranked_results[:10], 1):
                with st.expander(f"**Rank {rank}: Document {doc_id}** (Score: {score:.4f})"):
                    # Retrieve and display the full text of the document
                    doc_text = raw_documents[doc_id - 1]
                    st.write(doc_text)

if __name__ == "__main__":
    main()
