import nltk

def download_nltk_data():
    print("Downloading NLTK data...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('omw-1.4') # Often needed for lemmatizer
    print("NLTK data downloaded successfully.")

if __name__ == "__main__":
    download_nltk_data()
