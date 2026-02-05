import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans text by:
    1. Validating input type
    2. Lowercasing
    3. Removing special characters/numbers
    4. Removing stopwords
    5. Lemmatization
    """
    # Robustness check
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special chars and numbers (keep only letters and spaces)
    # Using the user-recommended regex for broad compatibility
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Tokenize (simple split by space is faster and sufficient for this level)
    words = text.split()
    
    # Remove stopwords and Lemmatize
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return " ".join(cleaned_words)
