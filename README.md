# Qognify: Vecbeddings – AI-Powered Study Material Generator

Qognify is a lightweight web application that turns any academic PDF or DOCX into an actionable study pack:

* **Summary** – a concise, map-reduce-style overview of the document.
* **Multiple-Choice Questions (MCQs)** – Questions with options each (answer pre-marked).
* **Short-Answer Questions** – Open-ended questions answerable directly from the text.
* **Exploratory (Curiosity) Questions** – Higher-order questions automatically ranked by novelty so learners can dig deeper.

Under the hood the app combines **LangChain** orchestration with **Cohere** LLMs / embeddings and a minimal **Flask** interface. All processing happens on-the-fly; no data is stored on the server.

---

## Features

1. **One-click upload** – drop a PDF or DOCX (≤ 25 MB) and receive study material in seconds.
2. **Multi-stage NLP pipeline**
   * Smart loader (`PyMuPDF`, `docx2txt`, `python-docx`) → raw text
   * Map-reduce summarisation chain
   * Prompt-driven question generation (MCQ & short answer)
   * Embedding-based novelty ranking for curiosity questions
3. **Modern UI** – responsive Bootstrap 5 & FontAwesome icons, no page reloads.
4. **Stateless & secure** – no document retention; only a single required secret (`COHERE_API_KEY`).

---

## Stack

| Layer            | Technology                              |
|------------------|------------------------------------------|
| Front-end        | HTML, Bootstrap 5, custom CSS            |
| Back-end         | Flask 2                                  |
| NLP Orchestration| LangChain                                |
| LLM & Embeddings | Cohere                                   |
| Parsing          | PyMuPDF, docx2txt, python-docx           |
| Vector maths     | NumPy                                    |

---

## Quick Start

### 1. Clone & install

```bash
# Clone
$ git clone https://github.com/your-org/qognify-vecbeddings.git
$ cd Qognify--Vecbeddings

# Create and activate a virtual environment (optional but recommended)
$ python3 -m venv .venv
$ source .venv/bin/activate

# Install dependencies
$ pip install -r requirements.txt
```

### 2. Configure secrets

Set your Cohere token as an environment variable:

```bash
export COHERE_API_KEY="your-cohere-api-key"
```

*(Optional)* Choose a custom port:

```bash
export PORT=8080
```

### 3. Run the app

```bash
python app.py
# or (equivalent)
flask --app app run --host 0.0.0.0 --port ${PORT:-5000}
```

Navigate to http://localhost:5000 and upload a document.

---

## Docker (alternative)

```bash
docker build -t qognify .
docker run -it --rm -e COHERE_API_KEY=your-cohere-api-key -p 5000:5000 qognify
```

---

## 🔍 Project Structure

```
.
├── app.py               # Flask entry-point and NLP pipeline
├── templates/
│   └── index.html       # Bootstrap UI
├── static/
│   └── css/style.css    # Custom styles
├── requirements.txt     # Python dependencies
└── README.md            # Docs (this file)
```

---

## API/Implementation Notes

* **Summarisation** uses `langchain.chains.summarize.load_summarize_chain` (`map_reduce`).
* **Question prompts** live in `app.py` and are adjustable.
* **Novelty ranking** normalises embeddings and computes `1 − cosine_similarity` between question & document vectors.
* Max upload size is capped at **25 MB** via `Flask.MAX_CONTENT_LENGTH`.
* Supported extensions: **.pdf**, **.docx**.
