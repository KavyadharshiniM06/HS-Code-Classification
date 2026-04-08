# 📦 Automated HS Code Classification via Hybrid RAG

An end-to-end AI pipeline designed to classify noisy commercial text (receipts, invoices, OCR) into standardized **Harmonized System (HS) Codes**. This system leverages a **Hybrid Retrieval-Augmented Generation (RAG)** architecture to ensure high accuracy and grounded reasoning.



## 🚀 Key Features

* **Intelligent OCR Cleaning:** Uses **Gemini 2.5 Flash** to semantically filter product descriptions from invoice noise (addresses, totals, tax IDs) without brittle hardcoding.
* **Hybrid Retrieval:** Combines **BM25 (Sparse)** for keyword matching and **BGE-M3 + FAISS (Dense)** for semantic similarity to handle both literal and conceptual matches.
* **Grounded Generation:** Utilizes **Flan-T5** as a local reasoning engine to select the best 6-digit HS code from retrieved candidates, providing a confidence score and explanation.
* **SROIE Integration:** Built-in support for processing the **SROIE 2019** dataset for real-world receipt classification testing.

## 🛠️ System Architecture

The pipeline follows a five-stage process:

1.  **Parsing:** `ReceiptParser` identifies candidate text lines from raw OCR output.
2.  **Cleaning:** `GeminiCleaner` normalizes noisy text into standardized trade descriptions.
3.  **Reformulation:** `QueryReformulator` strips units (KG, ML) and commercial noise for optimized search.
4.  **Hybrid Search:** `HybridRetriever` fetches the Top-K relevant HS codes from a local H6 dataset.
5.  **Classification:** `HSCodeGenerator` performs RAG-based inference to produce the final 6-digit code.



## 📁 Project Structure

```text
.
├── indexing/               # FAISS vector store and H6 metadata
├── pipelines/              # Orchestration (Retrieval, Generation, Main Pipeline)
├── retrievers/             # Hybrid Search logic (BM25 + Vector Search)
├── utils/                  # Gemini cleaning, Regex parsing, & reformulation
├── sroie_loader.py         # Main execution script for SROIE dataset processing
├── requirements.txt        # Project dependencies
└── .env                    # Environment variables (GEMINI_API_KEY)
```
## ⚙️ Installation & Setup
Clone the Repository:

```bash
git clone [https://github.com/KavyadharshiniM06/HS-Code-Classification.git](https://github.com/KavyadharshiniM06/HS-Code-Classification.git)
```
```bash
cd HS-Code-Classification
```
## Install Dependencies:

```bash
pip install -r requirements.txt
```
## Environment Variables:
Create a .env file in the root directory:

```bash
GEMINI_API_KEY=your_api_key_here
```
## Tesseract OCR:
Ensure Tesseract-OCR is installed on your system. Update the path in sroie_loader.py if necessary:
```bash
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## 📊 Usage
To run the pipeline on the SROIE dataset and generate a results CSV:

```bash
python sroie_loader.py
```

To launch the Streamlit demo app:

```bash
streamlit run app.py
```

If you want to upload receipt/invoice images, install Tesseract OCR and ensure it is available on your system PATH.

The output will be saved in results/sroie_icca_rag_results_new.csv, including:

* **Raw Line**: Original OCR text.

* **Cleaned Line**: Gemini-normalized description.

* **Final Prediction**: The 6-digit HS Code.

* **Reasoning**: The LLM's justification for the classification.

## 🛡️ Future Improvements
* **Fine-tuning** : Train Flan-T5 on specific customs rulings for deeper domain expertise.

* **Batch Processing**: Implement async batching for Gemini API calls to increase throughput.

* **Dashboard**: Add a Streamlit UI for manual auditing and verification of HS codes.
