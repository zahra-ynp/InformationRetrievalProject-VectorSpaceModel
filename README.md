# Vector Space Model for Information Retrieval

## Dataset: The Cranfield Collection

This project uses the well-known Cranfield collection, a standard testbed for IR experiments.

* **Source**: [http://ir.dcs.gla.ac.uk/resources/test_collections/cran/](http://ir.dcs.gla.ac.uk/resources/test_collections/cran/)
* **Content**: 1,400 documents consisting of abstracts from aeronautical research papers.
* **Corpus Files**:
    * `cran.all`: The documents file. Each document is marked by a `.I` (ID) and contains text in `.T` (Title) and `.W` (Words/Abstract) fields.
    * `cran.qry`: The queries file. It contains 225 natural language queries.
    * `cranqrel`: The relevance judgments file. This is the ground truth, mapping queries to the documents that are considered relevant for them.
