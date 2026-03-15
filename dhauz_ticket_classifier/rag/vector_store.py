"""ChromaDB vector store wrapper"""
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm.auto import tqdm
import os
import zipfile
import shutil
from ..config import CHROMA_DB_PATH, EMBEDDING_MODEL_NAME


class VectorStore:
    def __init__(self, embedding_model: str = EMBEDDING_MODEL_NAME, persist_dir: str = CHROMA_DB_PATH):
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.persist_dir = persist_dir
        self.db = None

    def create_from_dataframe(self, df):
        docs = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc='Preparing documents'):
            docs.append(Document(page_content=row["text"], metadata={"class": row["class"], "original_text": row["text"]}))
        os.makedirs(self.persist_dir, exist_ok=True)
        self.db = Chroma.from_documents(docs, self.embedding_model, persist_directory=self.persist_dir)

    def load(self):
        self.db = Chroma(embedding_function=self.embedding_model, persist_directory=self.persist_dir)

    def retrieve(self, query, class_filter, k=5):
        return self.db.similarity_search(query, k=k, filter={"class": class_filter})

    def exists(self) -> bool:
        """Return True if the persist directory appears to contain a saved Chroma DB."""
        return os.path.exists(self.persist_dir) and any(os.scandir(self.persist_dir))

    @property
    def persist_directory(self) -> str:
        return self.persist_dir

    def export_to_zip(self, zip_path: str):
        """Export the Chroma DB directory into a zip archive at `zip_path`.

        Args:
            zip_path: target zip file path (will be overwritten)
        """
        zip_path = os.path.abspath(zip_path)
        base_dir = os.path.abspath(self.persist_dir)
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Chroma persist directory not found: {base_dir}")
        # Create zip archive
        shutil.make_archive(base_name=os.path.splitext(zip_path)[0], format='zip', root_dir=base_dir)

    def import_from_zip(self, zip_path: str, overwrite: bool = False):
        """Extract a zip archive into the persist directory.

        Args:
            zip_path: path to zip archive
            overwrite: if True, existing directory will be removed first
        """
        zip_path = os.path.abspath(zip_path)
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Zip archive not found: {zip_path}")
        if os.path.exists(self.persist_dir):
            if overwrite:
                shutil.rmtree(self.persist_dir)
            else:
                raise FileExistsError(f"Persist directory already exists: {self.persist_dir}")
        os.makedirs(self.persist_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(self.persist_dir)
