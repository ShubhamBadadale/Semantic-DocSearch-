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

## 📸 Screenshots

### 🏠 Home Page
![Home](Screenshots/0.Home.png)

### ✍️ Taking Input
![Taking Input](Screenshots/1.Taking_Input.png)

### ✂️ Sentence Extraction
![Sentence Extraction](Screenshots/2.Sentence_Extraction.png)

### 📊 Ready to Search
![Ready to Search](Screenshots/3.Ready_to_Search.png)

### 🧠 Example Result 1
![Example 01](Screenshots/4.Example_No_01.png)
![Example 01.1](Screenshots/5.Example_No_01.1.png)
![Example 01.2](Screenshots/6.Example_No_01.2.png)

### 🧠 Example Result 2
![Example 02.1](Screenshots/7.Example_No_02.1.png)
![Example 02.2](Screenshots/8.Example_No_02.2.png)

