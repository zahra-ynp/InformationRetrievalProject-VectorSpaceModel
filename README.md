# 📚 Vector Space Model Search Engine

This project is a comprehensive implementation of a **Vector Space Model (VSM)** for information retrieval, developed for the *Information Retrieval* course at the **University of Trieste**.

It processes the **Cranfield document collection**, represents documents and queries as **TF-IDF weighted vectors**, ranks them using **cosine similarity**, and refines results using **pseudo-relevance feedback (Rocchio algorithm)**. The system is presented through an interactive **Streamlit web application** and includes a  **evaluation**.

---

## 🧠 Key Features

* TF-IDF vector representation for documents and queries
* Cosine similarity ranking
* Rocchio algorithm for pseudo-relevance feedback
* Evaluation using Precision\@10, Recall\@10, F1-Score
* Interactive UI with Streamlit

---

## 📂 Dataset: The Cranfield Collection

* **Source:** [Cranfield Test Collection](http://ir.dcs.gla.ac.uk/resources/test_collections/cran/)
* **Documents:** 1,400 aeronautical abstracts (`cran.all`)
* **Queries:** 225 natural language queries (`cran.qry`)
* **Relevance Judgments:** Query-document mappings (`cranqrel`)

---

## ⚙️ Components Overview

### 🔧 `retrieval.py`

* Core engine for:

  * Preprocessing
  * Query Vectorization
  * Document Ranking
  * Cosine similarity ranking
  * Rocchio pseudo-relevance feedback

---

### 🏗️ `1_Index_Builder.ipynb`

* One-time setup to:

  * Load and preprocess documents
  * Tokenize, lemmatize, and remove stop-words
  * Compute and save TF-IDF vectors (`.pkl` files)

---

### 📊 `2_Evaluation.ipynb`

* Systematic evaluation:

  * Load artifacts and ground-truth data
  * Run all 225 queries
  * Compute Precision\@10, Recall\@10, F1-Score
    
---

### 🌐 `app.py`

* Streamlit web interface:

  * Input queries
  * Enable/disable relevance feedback
  * View top-ranked documents with similarity scores

---

## 🚀 How to Run

### 1️⃣ Installation

Clone the repository and install required libraries:

 ```bash
 git clone https://github.com/zahra-ynp/InformationRetrievalProject-VectorSpaceModel.git 
 pip install -r requirements.txt 
 ```

---

### 2️⃣ Build Data Artifacts

Run the index builder to process the dataset:

1. Open `1_Index_Builder.ipynb` in Jupyter
2. Execute all cells
3. Ensure `.pkl` files are saved in `./artifacts/`

---

### 3️⃣ Launch the Web App

Start the Streamlit app from your terminal:

```bash
streamlit run app.py
```

A browser window will open with the search engine interface.

---

## 📈 Example Query

Search for: what problems of heat conduction in composite slabs have been solved so far
Toggle feedback: ✅
See a ranked list of relevant research abstracts!

---

## 📑 License

This project is developed for academic purposes under the [MIT License].
