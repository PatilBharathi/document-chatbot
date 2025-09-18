Document Chatbot 
=======================================================================================================================================

This project is a document-based conversational chatbot built with Python, FAISS, Hugging Face Transformers, and Streamlit.
It allows users to upload or query documents and receive intelligent, context-aware answers.

======================================================================================================================================
Features

Document Processing: Load and chunk large documents (up to 10,000+ words).

Semantic Search: Uses embeddings + FAISS for efficient retrieval.

Query Handling: Retrieves the most relevant chunks to answer user queries.

Response Generation: Generates concise and context-aware answers.

Interactive Chat UI: Built with Streamlit, includes:

	Chat history

	Clear Chat & Exit buttons

	"Typing…" indicator for better UX

Error Handling: Graceful fallback when no relevant content is found.

Extensible: Modular code (loader, retriever, generator, chatbot UI).