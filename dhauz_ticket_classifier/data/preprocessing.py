"""Text preprocessing utilities"""
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    stopwords.words("english")
except Exception:
    nltk.download("stopwords")
    nltk.download("wordnet")


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)
