#!/usr/bin/env python3
"""
Test script for the RAG API endpoints.

This script tests the new endpoints for creating knowledge bases, uploading documents,
and building/processing a knowledge base.
"""

import requests
import os
import sys
import json
from pprint import pprint

# URL of the backend API
BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def create_knowledge_base():
    """Test creating a new knowledge base"""
    print("\n=== Testing knowledge base creation ===")
    url = f"{BASE_URL}/knowledge_bases"
    
    data = {
        "name": "Test Knowledge Base",
        "description": "A test knowledge base for API validation"
    }
    
    print(f"POST {url}")
    print(f"Data: {data}")
    
    response = requests.post(url, json=data)
    
    print(f"Status code: {response.status_code}")
    if response.status_code == 201:
        response_data = response.json()
        print("Response:")
        pprint(response_data)
        return response_data["id"]
    else:
        print(f"Error: {response.text}")
        return None

def upload_document(kb_id, file_path):
    """Test uploading a document to a knowledge base"""
    print("\n=== Testing document upload ===")
    url = f"{BASE_URL}/knowledge_bases/{kb_id}/documents"
    
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return False
    
    print(f"POST {url}")
    print(f"File: {file_path}")
    
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "application/pdf")}
        response = requests.post(url, files=files)
    
    print(f"Status code: {response.status_code}")
    if response.status_code == 201:
        response_data = response.json()
        print("Response:")
        pprint(response_data)
        return True
    else:
        print(f"Error: {response.text}")
        return False

def build_knowledge_base(kb_id):
    """Test building a knowledge base"""
    print("\n=== Testing knowledge base build ===")
    url = f"{BASE_URL}/knowledge_bases/{kb_id}/build"
    
    print(f"POST {url}")
    
    response = requests.post(url)
    
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        response_data = response.json()
        print("Response:")
        pprint(response_data)
        return True
    else:
        print(f"Error: {response.text}")
        return False

def ask_question(kb_id, question):
    """Test asking a question to the knowledge base"""
    print("\n=== Testing RAG query ===")
    url = f"{BASE_URL}/rag"
    
    data = {
        "knowledge_base_id": kb_id,
        "question": question
    }
    
    print(f"POST {url}")
    print(f"Data: {data}")
    
    response = requests.post(url, json=data)
    
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        response_data = response.json()
        print("Answer:")
        print(response_data["answer"])
        print("\nSources:")
        for i, chunk in enumerate(response_data["chunks"]):
            print(f"Source {i+1}: {chunk['pdfUrl']}")
            print(f"Text: {chunk['text'][:100]}...")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def main():
    """Run the API tests"""
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <pdf_file_path> [question]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Get the question from command line args or use a default
    question = sys.argv[2] if len(sys.argv) > 2 else "What is this document about?"
    
    # Create a knowledge base
    kb_id = create_knowledge_base()
    if not kb_id:
        sys.exit(1)
    
    # Upload a document
    success = upload_document(kb_id, pdf_path)
    if not success:
        sys.exit(1)
    
    # Build the knowledge base
    success = build_knowledge_base(kb_id)
    if not success:
        sys.exit(1)
    
    # Ask a question
    success = ask_question(kb_id, question)
    if not success:
        sys.exit(1)
    
    print("\n=== All tests completed successfully ===")
    print(f"Knowledge Base ID: {kb_id}")

if __name__ == "__main__":
    main() 