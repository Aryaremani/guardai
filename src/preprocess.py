import re

def clean_text(text):
    """
    Cleans raw text rapidly for large datasets.
    - Removes URLs
    - Keeps only alphabets and numbers
    - Removes extra whitespaces
    
    Note: Lowercasing and stopword removal will be handled 
    by TfidfVectorizer for better performance.
    """
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove special characters (keep alphanumerics and standard punctuation like ?!.,)
    text = re.sub(r'[^a-zA-Z0-9\s\?!\.,]', '', text)
    
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
