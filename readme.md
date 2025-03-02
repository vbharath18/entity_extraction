# Document Analysis and Entity Extraction

This repository contains tools for analyzing documents, extracting entities, and performing semantic search using Azure OpenAI and LangChain.

## Project Structure

- [entity_extraction.py](entity_extraction.py): Main script for extracting patient demographics from medical documents using a Gradio interface
- [entity_extraction.ipynb](entity_extraction.ipynb): Jupyter notebook with the same functionality as the Python script, with step-by-step execution
- [vector_search.py](vector_search.py): Utility module for semantic search and document processing
- [config.py](config.py): Configuration settings for API keys and model parameters
- [utils/text_processing.py](utils/text_processing.py): Helper functions for text preprocessing
- [utils/validation.py](utils/validation.py): Functions for validating extracted entities
- [data/sample_documents/](data/sample_documents/): Sample medical documents for testing
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
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables for your Azure OpenAI API keys:
   ```bash
   export AZURE_OPENAI_API_KEY="your-api-key"
   export AZURE_OPENAI_ENDPOINT="your-endpoint"
   ```
   Alternatively, add them to your `.env` file.

## Usage

### Running the Gradio Interface

```bash
python entity_extraction.py
```

This will start a local web server (typically at http://127.0.0.1:7860) where you can upload documents and extract entities.

### Using the Jupyter Notebook

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `entity_extraction.ipynb` and follow the step-by-step instructions.

### Using the Python API

```python
from vector_search import DocumentProcessor
from utils.text_processing import preprocess_text

# Initialize processor
processor = DocumentProcessor()

# Process a document
document_text = "Patient: John Doe, DOB: 01/15/1980..."
processed_text = preprocess_text(document_text)
entities = processor.extract_entities(processed_text)

print(entities)
```

## Examples

### Sample Output

```json
{
  "patient_name": "John Doe",
  "date_of_birth": "1980-01-15",
  "gender": "Male",
  "medical_record_number": "MRN12345",
  "address": "123 Main Street, Anytown, CA 94123",
  "phone_number": "(555) 123-4567"
}
```

## Troubleshooting

- **API Key Issues**: Ensure your Azure OpenAI API keys are correctly set in environment variables or `.env` file
- **Model Availability**: Confirm you have access to the required models in your Azure OpenAI subscription
- **Memory Errors**: For large documents, consider increasing your system's memory allocation or processing documents in smaller chunks

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin new-feature`
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.