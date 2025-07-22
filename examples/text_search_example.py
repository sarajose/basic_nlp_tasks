"""
Text Search Example

This example demonstrates how to use the text_search module for:
1. Basic text search with exact matching
2. More advanced search using the TextSearch class (requires word embeddings)
"""

from NLPlib.text_search import search

def main():
    # Sample documents for searching
    documents = [
        "This is the first document about natural language processing.",
        "This document discusses machine learning techniques.",
        "Natural language processing is a subfield of artificial intelligence.",
        "Machine learning models can be trained to understand language.",
        "Text analysis involves processing and understanding natural language.",
    ]
    
    # Basic text search demonstration
    print("Basic text search demonstration:")
    queries = ["document", "natural language", "machine learning", "artificial"]
    
    for query in queries:
        results = search(query, documents)
        print(f"\nSearch query: '{query}'")
        print(f"Found {len(results)} results:")
        for i, doc in enumerate(results):
            print(f"  {i+1}. {doc}")
    
    # Advanced search using TextSearch class
    print("\nAdvanced search using TextSearch class:")
    print("Note: This part requires word embedding files which are not included.")
    print("To use the TextSearch class with embeddings:")
    print("""
    # Initialize with word vector files
    word_vector_files = ['path_to_embeddings.en.gz']
    search_engine = text_search.TextSearch(word_vector_files)
    
    # Index text from a file
    search_engine.index_text('path_to_text.gz', 'en')
    
    # Search using the semantic search engine
    results = search_engine.search(['natural', 'language', 'processing'], 'en', n_matches=2)
    
    # Display results
    for similarity, filename, sentence in results:
        print(f"Score: {similarity:.4f} - {sentence}")
    """)

if __name__ == "__main__":
    main()
