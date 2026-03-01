"""Document chunking with RecursiveCharacterTextSplitter."""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    separators: List[str] | None = None,
) -> List[Document]:
    """
    Split documents into overlapping chunks for retrieval.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
    )
    return splitter.split_documents(documents)
