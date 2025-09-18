💬 Document Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) chatbot built with Python, FAISS, Hugging Face Transformers, and Streamlit.
It answers user queries by retrieving relevant content from a PDF manual and generating concise, bullet-point responses inside a modern chat UI.

🚀 Features

Document Processing

	Extracts text from PDF, removes headers/footers.
	Splits into sentence-aware overlapping chunks (~200 words, 20% overlap).
	Embeds with MiniLM and builds a FAISS index.

Efficient Retrieval

	Query expansion with synonyms.
	Threshold filtering and TOC cleanup.
	Retriever limited to top-2 chunks to reduce noise.

Response Generation

	Uses free, CPU-friendly model flan-t5-large.
	Strict prompting rules enforce concise, bullet-only answers.
 
Post-processing

	Deduplication and repetition cleanup.
	Sentence trimming to remove incomplete fragments.
	Filters out noisy content like "User Manual Rev …".

Chat UI (Streamlit)

	Clean chat bubbles (user → right, bot → left).
	Bullets render properly inside chat bubbles.
	“🤔 Thinking…” spinner while generating.
	Clear Chat and Exit Chatbot buttons.

 
📂 Project Structure

<img width="647" height="216" alt="image" src="https://github.com/user-attachments/assets/82c16735-058e-4a38-8314-94998a33c285" />



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

	Add citations (section + page references) below answers.
	Index images/diagrams via OCR or vision models.
	Hybrid retrieval (BM25 + embeddings).
	Feedback system to refine answers.
	Export chat history as text.
