# 🧠 Semantic DocSearch

A smart document search engine that understands **meaning**, not just **keywords**.

Upload multiple files (`.pdf`, `.docx`, `.txt`) and ask any question — DocSearch finds the **most relevant sentences** across all documents using **sentence embeddings** and **semantic similarity**.

---

## 🚀 Features

- 🔍 **Semantic Search** using Sentence Transformers (`all-MiniLM-L6-v2`)
- 📄 Supports **PDF**, **Word (DOCX)**, and **Text files**
- 📍 Returns **exact sentence**, location & document
- 🌌 Shows **context before and after**
- ⚡ Fast, modern **Streamlit UI**
- 🧠 No keyword-matching — it understands what you mean

---

## 📁 Example Use Case

Upload `Constitution of India.pdf` and ask:
> _"What are the fundamental rights guaranteed?"_

It will:
- Find the most relevant section
- Show you the **sentence**, document **page**, and nearby context

---

## 🛠 Tech Stack

- `Python`
- `Streamlit` for UI
- `SentenceTransformers` for embeddings
- `PyMuPDF`, `python-docx` for file reading
- `NLTK` for sentence splitting

---

## 🔧 Run Locally

1. Clone the repo  
   `git clone https://github.com/ShubhamBadadale/Semantic-DocSearch-.git`

2. Install dependencies  
   `pip install -r requirements.txt`

3. Run the app  
   `streamlit run Sementic_DocSearch.py`

---

## 📸 Screenshots

_Add your screenshots here for demo!_

---

## 🙋‍♂️ Author

Made with ❤️ by [Shubham Badadale](https://github.com/ShubhamBadadale)  
Built during a Hackathon as part of **Team Vector Vision**

