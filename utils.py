from langchain.docstore.document import Document
from typing import List, Optional


def find_hit(ground_truth: str, candidates: List[str], prefix_char: Optional[int] = 200):
    for i, candidate in enumerate(candidates):
        if candidate.startswith(ground_truth[:prefix_char]):
            return i
    else:
        return None


def create_documents(chunks: dict):
    documents = []
    for i in range(len(chunks['documents'])):
        page_content = chunks['documents'][i]
        chapter = chunks['metadatas'][i]['chapter']
        file_name = chunks['metadatas'][i]['file_name']
        source = chunks['metadatas'][i]['source']
        metadata = {
            'chapter': chapter,
            'file_name': file_name,
            'source': source
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents
