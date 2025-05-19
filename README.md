# LangChain-PlanetQA
# 🌌 LangChain Planet QA

This project demonstrates a simple question-answering system using LangChain, FAISS, and HuggingFace Transformers. It utilizes a small set of planetary facts and enables question answering through similarity-based retrieval and FLAN-T5 language model.

## 🚀 Features

- Document embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Vector search using FAISS
- Question answering via `google/flan-t5-base`
- Integrated with LangChain's `RetrievalQA` chain

## 📂 Sample Dataset

The example uses three short texts about Earth, Mars, and Jupiter. These are embedded and stored in a FAISS index for retrieval.

## 💡 Sample Query

Which planet is known as the Red Planet?

makefile
Copy
Edit

Output:
Mars is the fourth planet from the Sun and is often called the Red Planet due to its reddish appearance.

perl
Copy
Edit

## 🛠 Installation

```bash
pip install -U langchain langchain-community sentence-transformers faiss-cpu transformers
📄 Code Overview
Create Document objects from raw texts

Embed with HuggingFaceEmbeddings

Store and retrieve with FAISS

Use google/flan-t5-base with LangChain’s RetrievalQA

Query the model and get precise answers

📁 Files
main.py: Main script that runs the QA pipeline

🧠 Future Ideas
Add a web interface (e.g., with Flask or Streamlit)

Use a larger set of documents or load from a file

Enable real-time query logging and feedback collection

📝 License
This project is open-source and free to use under the MIT License.
