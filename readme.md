# Document Analysis and Entity Extraction

This repository contains tools for analyzing documents, extracting entities, and performing semantic search using Azure OpenAI and LangChain.

## Project Structure

- [entity_extraction.py](entity_extraction.py): Main script for extracting patient demographics from medical documents using a Gradio interface
- [entity_extraction.ipynb](entity_extraction.ipynb): Jupyter notebook with the same functionality as the Python script, with step-by-step execution
- [vector_search.py](vector_search.py): Utility module for semantic search and document processing
- [LICENSE](LICENSE): Apache License 2.0

## Features

- **Entity Extraction**: Extract structured information like names, dates, and demographics from medical documents
- **Semantic Search**: Find relevant information in documents using vector embeddings
- **Data Validation**: Validate extracted information against known data
- **User Interface**: Simple web interface for document analysis using Gradio

## Prerequisites

- Azure OpenAI API key and endpoint
- Python 3.11+
- Required Python packages (see installation section)

## Installation

1. Clone this repository
2. Install required packages: