pip install requests beautifulsoup4 sentence-transformers transformers faiss-cpu

import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss  
from transformers import pipeline

class DataIngestion:
    def __init__(self, urls):
        self.urls = urls
        self.contents = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained embedding model

    def scrape(self):
        for url in self.urls:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            self.contents.append(text)

    def chunk_and_embed(self):
        self.chunks = []
        self.embeddings = []
        for content in self.contents:
            # Segment content into chunks (e.g., sentences)
            sentences = content.split('. ')
            for sentence in sentences:
                if sentence.strip():
                    self.chunks.append(sentence.strip())
                    embedding = self.model.encode(sentence.strip())
                    self.embeddings.append(embedding)

        # Convert to numpy array for FAISS
        self.embeddings = np.array(self.embeddings).astype('float32')

    def store_embeddings(self):
        # Create a FAISS index
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])  # L2 distance
        self.index.add(self.embeddings)  # Add embeddings to the index
