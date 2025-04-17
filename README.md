# Document Q&A System

A Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents, process them, and ask questions about their content. The system uses AI to analyze documents and provide accurate answers with highlighted source references.

## Features

- **Document Upload Interface**: Drag-and-drop interface for uploading PDF documents
- **Automatic Document Processing**: AI-powered processing of documents for question answering
- **Conversational Interface**: Chat-like interface for asking questions about your documents
- **Source References**: View the exact sources of information in the answers
- **PDF Viewer with Highlighting**: See the exact location of information in the original documents
- **Docling Integration**: Uses the docling library for high-quality document processing

## System Architecture

The system consists of two main parts:

1. **Frontend**: React application with components for document upload, PDF viewing, and chat interface
2. **Backend**: FastAPI server that handles document processing, storage, and question answering

### Technology Stack

- **Frontend**: React, react-pdf
- **Backend**: Python, FastAPI, docling, ChromaDB (vector database)
- **AI**: OpenAI's embedding and language models

## Setup and Installation

### Prerequisites

- Node.js (v16+)
- Python (v3.9+)
- OpenAI API key

### Backend Setup

1. Clone the repository
2. Create a Python virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Copy the `.env.example` file to `.env` and configure your settings:
   ```
   cp .env.example .env
   ```
5. Set your OpenAI API key in the `.env` file
6. Start the backend server:
   ```
   python rag_api.py
   ```

### Frontend Setup

1. Install Node.js dependencies:
   ```
   npm install
   ```
2. Start the development server:
   ```
   npm start
   ```
3. Open [http://localhost:3000](http://localhost:3000) in your browser

## API Endpoints

The backend provides the following API endpoints:

- `POST /knowledge_bases`: Create a new knowledge base
- `POST /knowledge_bases/{kb_id}/documents`: Upload a document to a knowledge base
- `POST /knowledge_bases/{kb_id}/build`: Process and index documents in a knowledge base
- `GET /knowledge_bases/{kb_id}/documents/{filename}`: Serve a PDF file
- `POST /rag`: Query a knowledge base using RAG

## Usage Guide

1. **Upload Documents**: On the main page, upload your PDF documents using the drag-and-drop interface
2. **Process Documents**: The system will automatically process and index your documents
3. **Ask Questions**: Use the chat interface to ask questions about your documents
4. **View Sources**: Click on the source references to see the exact location in the original document
5. **Navigate PDFs**: Use the PDF viewer controls to zoom, navigate pages, and view highlighted information

## Testing

To test the API, you can use the provided test script:

```
python test_api.py path/to/sample.pdf "What is this document about?"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
