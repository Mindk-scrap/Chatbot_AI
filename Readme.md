# Chabtbot_AI

## How to use

1. Clone the repository and install dependencies
```
git clone https://github.com/Mindk-scrap/Chatbot_AI.git
pip install -r requirements.txt
```

2. Specify the PDF link and OPEN_API_KEY to create the embedding model
```
# Example
python ingest.py --pdf_path "my_pdf.pdf" --OPEN_API_KEY "mention_your_key"
```
The following options can also be specified as arguments
- chunk_size:
    Please specify the chunk_size for CharacterTextSplitter within a number less than or equal to 4096.
- chunk_overlap: 
    Please specify the chunk_overlap for CharacterTextSplitter within a number less than or equal to 4096.
- split_mode: 
    Please specify the split mode. (character, recursive_character, nltk, tiktoken)
- faiss_save_path: 
    Please specify the name of the created Faiss object.

3. Enter a question and generate an answer from the extracted text
```
# Example
python main.py --query "How to achieve wisdom?"
```

