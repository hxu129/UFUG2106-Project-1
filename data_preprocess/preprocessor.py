import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from collections import defaultdict
import re
from typing import List, Set, Dict, Union, Optional

class TextPreprocessor:
    """Base class for text preprocessing"""
    
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if self.lowercase:
            text = text.lower()
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        return ' '.join(text.split())  # Normalize whitespace

class KGramGenerator(TextPreprocessor):
    """Generates k-grams for MinHash"""
    
    def __init__(self, k: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        
    def get_kgrams(self, text: str) -> Set[str]:
        """Generate k-grams from text"""
        cleaned_text = self.clean_text(text)
        # Pad the text for edge k-grams
        padded = f"{'$' * (self.k-1)}{cleaned_text}{'$' * (self.k-1)}"
        return set(padded[i:i+self.k] for i in range(len(padded)-self.k+1))

class TfidfProcessor(TextPreprocessor):
    """Generates TF-IDF representations for SimHash"""
    
    def __init__(self, max_features: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.vectorizer = TfidfVectorizer(
            lowercase=self.lowercase,
            max_features=max_features
        )
        self.is_fitted = False
        
    def fit(self, texts: List[str]):
        """Fit the TF-IDF vectorizer"""
        cleaned_texts = [self.clean_text(text) for text in texts]
        self.vectorizer.fit(cleaned_texts)
        self.is_fitted = True
        
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to TF-IDF vectors"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        cleaned_texts = [self.clean_text(text) for text in texts]
        return self.vectorizer.transform(cleaned_texts)

class HashingProcessor(TextPreprocessor):
    """Generates hash-based feature vectors for bit sampling"""
    
    def __init__(self, n_features: int = 1024, **kwargs):
        super().__init__(**kwargs)
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            lowercase=self.lowercase
        )
        
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to hash-based feature vectors"""
        cleaned_texts = [self.clean_text(text) for text in texts]
        return self.vectorizer.transform(cleaned_texts)

class DataPreprocessor:
    """Main preprocessor class that handles different representations"""
    
    def __init__(self, 
                 kgram_k: int = 3,
                 tfidf_max_features: Optional[int] = None,
                 hashing_n_features: int = 1024):
        self.kgram_generator = KGramGenerator(k=kgram_k)
        self.tfidf_processor = TfidfProcessor(max_features=tfidf_max_features)
        self.hashing_processor = HashingProcessor(n_features=hashing_n_features)
        
    def prepare_for_minhash(self, texts: List[str]) -> List[Set[str]]:
        """Prepare texts for MinHash by generating k-grams"""
        return [self.kgram_generator.get_kgrams(text) for text in texts]
    
    def prepare_for_simhash(self, texts: List[str]) -> np.ndarray:
        """Prepare texts for SimHash using TF-IDF"""
        if not self.tfidf_processor.is_fitted:
            self.tfidf_processor.fit(texts)
        return self.tfidf_processor.transform(texts)
    
    def prepare_for_bit_sampling(self, texts: List[str]) -> np.ndarray:
        """Prepare texts for bit sampling using HashingVectorizer"""
        return self.hashing_processor.transform(texts)

# Example usage:
if __name__ == "__main__":
    # Sample texts
    texts = [
        "This is a sample document.",
        "Another example text for processing.",
        "Third document with some overlap."
    ]
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        kgram_k=3,
        tfidf_max_features=1000,
        hashing_n_features=512
    )
    
    # Get different representations
    minhash_repr = preprocessor.prepare_for_minhash(texts)
    simhash_repr = preprocessor.prepare_for_simhash(texts)
    bit_sampling_repr = preprocessor.prepare_for_bit_sampling(texts) 