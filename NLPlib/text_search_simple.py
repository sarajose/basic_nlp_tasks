# A simple search implementation that matches the README example

def search(query, documents):
    """
    Simple search function that finds documents matching a query term.
    
    This is a basic implementation for demonstration purposes only.
    For real-world usage, consider using the TextSearch class.
    
    Args:
        query (str): The search query string
        documents (list): List of document strings to search in
        
    Returns:
        list: Documents containing the query term
    """
    query_lower = query.lower()
    results = []
    for doc in documents:
        if query_lower in doc.lower():
            results.append(doc)
    return results
