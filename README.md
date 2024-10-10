### Prototype of a RAG system built with langchain and Llama 3

Experiments to improve quality include:
* embedders fine-tuning
* HyDE [(Gao et al., 2023)](https://aclanthology.org/2023.acl-long.99)
* query planning

[Presentation of the project](https://docs.google.com/presentation/d/1z8g3vZrzmakzrFJbgXKsu8rgrkYW8GhguycNe4YIODg/edit?usp=sharing) (in Russian)

```mermaid
---
title: RAG system architecture
---
flowchart TD
    docs[[Documents]]
    embedder_docs[/Embedder/]
    embedder[/"Embedder\n(same model)"/]
    chroma[(Chroma DB)]
    context
    
    query(user question)
    hyde[/HyDE\]
    planner[/Query planner\]
    
    llm[/LLM\]
    
    answer(answer)
    
    docs --chunking--> embedder_docs
    embedder_docs -.fine-tuning.-> docs
    embedder_docs --> chroma
    
    query --> embedder
    query --> context
    query -.-> hyde & planner
    hyde -.-> embedder
    planner -.-> embedder & embedder
    
    embedder --retrieval--> chroma
    chroma --N paragraphs--> context
    context --> llm
    llm --> answer
    
    docs ~~~ query
```
