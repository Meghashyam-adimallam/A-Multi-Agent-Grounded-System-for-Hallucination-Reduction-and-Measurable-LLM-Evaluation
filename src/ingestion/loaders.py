"""Document loaders for PDF, text, and CSV files."""

from pathlib import Path
from typing import Any, List, Tuple, Type

from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_documents(documents_path: str | Path) -> List[Document]:
    """
    Load documents from a directory. Supports PDF, TXT, MD, and CSV.
    """
    path = Path(documents_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Documents path not found: {path}")

    docs: List[Document] = []
    # (loader_class, extra_kwargs). CSV needs encoding for Windows/Excel files.
    loaders: List[Tuple[Type[Any], dict]] = [
        (PyPDFLoader, {}),
        (TextLoader, {}),
        (TextLoader, {}),  # .md uses TextLoader
        (CSVLoader, {"encoding": "utf-8-sig"}),  # utf-8-sig handles Excel BOM
    ]
    exts = [".pdf", ".txt", ".md", ".csv"]

    for ext, (loader_cls, kwargs) in zip(exts, loaders):
        for file_path in path.rglob(f"*{ext}"):
            if not file_path.is_file():
                continue
            try:
                # Use absolute path so loading works regardless of cwd
                full_path = str(file_path.resolve())
                loader = loader_cls(full_path, **kwargs)
                docs.extend(loader.load())
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")

    return docs
