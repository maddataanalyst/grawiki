1. Question 1 

I have another conceptual question. I want to replicate functionality of Llama Index or other property graph RAG systems. How should we implement querying texts - we can assume that
  Documents will be too large to fit in LLMs memory. Thats why we do chunking - chunks are smaller.The questio is about the graph RAG and queries - should query to RAG be against       
  chunks or documents too. In practical terms - how should @src/grawiki/db/falkordb.py  implementation behave when cerating indices for vector / term search.       

