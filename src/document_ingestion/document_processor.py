"""Document processing module for loading and splitting documents."""

from typing import List, Union
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentProcessor:
    """Handles document loading and processing."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor.

        Args:
            chunk_size: Size of text chunks.
            chunk_overlap: Overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Try splitting at "\n\n" 
        # "RecursiveCharacterTextSplitter splits at correct ending positions"
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_from_url(self, url: str) -> List[Document]:
        """Load document(s) from a URL."""
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load documents from all PDFs inside a directory."""
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a TXT file."""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a single PDF file."""
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents from URLs, PDF files/directories, or TXT files.

        Args:
            sources: List of URLs, file paths, or folder paths.

        Returns:
            List of loaded documents.
        """
        docs: List[Document] = []

        for src in sources:
            src_path = Path(src)

            # URL
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))

            # Single PDF file
            elif src_path.suffix.lower() == ".pdf":
                docs.extend(self.load_from_pdf(src_path))

            # TXT file
            elif src_path.suffix.lower() == ".txt":
                docs.extend(self.load_from_txt(src_path))

            # Directory containing files
            elif src_path.is_dir():
                # Manually scan for supported files (recursive)
                for file_path in src_path.rglob("*"):
                    if file_path.suffix.lower() == ".pdf":
                        docs.extend(self.load_from_pdf(file_path))
                    elif file_path.suffix.lower() == ".txt":
                        docs.extend(self.load_from_txt(file_path))

            else:
                raise ValueError(
                    f"Unsupported source type: {src}. "
                    "Use URL, .txt file, .pdf file, or a directory of PDFs."
                )
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of documents to split.

        Returns:
            List of split documents.
        """
        return self.splitter.split_documents(documents)

    def process_urls(self, sources: List[str]) -> List[Document]:
        """
        Complete pipeline to load and split documents.

        Args:
            sources: List of URLs / paths to process.

        Returns:
            List of processed document chunks.
        """
        docs = self.load_documents(sources)
        return self.split_documents(docs)