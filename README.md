💬 Document Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) chatbot built with Python, FAISS, Hugging Face Transformers, and Streamlit.
It answers user queries by retrieving relevant content from a PDF manual and generating concise, context-aware responses.

🚀 Features

Document Processing: Splits the manual into overlapping chunks, embeds them with MiniLM, and builds a FAISS index.

Efficient Retrieval: Synonym-based query expansion, threshold filtering, TOC cleanup, and top-k search.

Response Generation: Free, CPU-friendly model (flan-t5-large) generates natural answers.

Post-processing: Deduplication, repetition cleanup, sentence trimming, bullet-point formatting.

Chat UI: Modern Streamlit interface with:

	Chat bubbles (user right, bot left)

	“🤔 Thinking…” spinner while generating

	Clear Chat & Exit buttons
 
📂 Project Structure

<img width="618" height="211" alt="image" src="https://github.com/user-attachments/assets/c7aa3336-3229-4a69-a624-9542c10069a7" />


⚙️ Installation
1. Clone the repo
	git clone https://github.com/PatilBharathi/document-chatbot.git
	cd document-chatbot

2. Create virtual environment
	python -m venv venv
	venv\Scripts\activate      # Windows
	source venv/bin/activate   # Mac/Linux

3. Install dependencies
	pip install -r requirements.txt

▶️ Usage
Step 1: Place your PDF
Add your document as:

	data/user_manual.pdf

Step 2: Build FAISS index
	python document_loader.py

This creates:

	data/faiss_index.bin
	data/chunks.pkl

Step 3: Run the chatbot
	streamlit run chatbot.py

📖 Example Interaction

<img width="997" height="1643" alt="image" src="https://github.com/user-attachments/assets/91cd7d18-46a4-452c-9c35-56a7d302c7e6" />


🛠️ Requirements

	Python 3.9+
	Streamlit
	FAISS (CPU)
	Hugging Face Transformers
	Sentence Transformers
	Torch
	
	See requirements.txt for full list.

🖥️ Hardware Requirements

This project is lightweight and can run fully on CPU.

	Minimum:

		CPU: Dual-core (Intel i3 / AMD equivalent)

		RAM: 8 GB

		Disk: ~2 GB free (for embeddings + model cache)

	Recommended:

		CPU: Quad-core (Intel i5/i7 or AMD Ryzen 5/7)

		RAM: 16 GB (for faster embedding/model inference)

		Disk: ~5 GB free

	Optional GPU:

	If available (CUDA-enabled), Hugging Face Transformers will automatically use it to speed up inference.

	Not required for this project since it uses flan-t5-large (CPU-friendly).
 
🧩 Future Improvements

	Index images via OCR or vision models to support diagrams.
	Hybrid retrieval (BM25 + embeddings).
	Feedback system to refine answers.
	Export chat history as text.
