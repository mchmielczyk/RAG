def validate_query(query):
    if query is None or not query.strip():
        raise ValueError("Query cannot be empty.")

def validate_limit(limit):
    if limit <= 0:
        raise ValueError("Limit must be greater than 0.")