# PDF Chatbot with Retrieval-Augmented Generation (RAG)

A Streamlit-based web application that enables users to upload PDF documents and interact with their content through a conversational AI assistant. The assistant utilizes Retrieval-Augmented Generation (RAG) to provide context-aware responses, enhancing the user experience.

## 🚀 Features

- Upload and process multiple PDF documents.
- Conversational AI assistant powered by Groq's ChatGroq.
- Context-aware responses using RAG methodology.
- Persistent chat history for ongoing interactions.

## 🛠️ Technologies Used

- **Streamlit**: For building the web interface.
- **LangChain**: For document processing and RAG integration.
- **ChatGroq**: Groq's language model for generating responses.
- **Chroma**: Vector store for document embeddings.
- **Hugging Face Transformers**: For embedding generation.

## 📦 Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/divyaece23/RAG_PDF_chatbot_with_chat_history.git
cd RAG_PDF_chatbot_with_chat_history
python -m venv .venv
source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
pip install -r requirements.txt
