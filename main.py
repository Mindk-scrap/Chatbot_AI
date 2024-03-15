# import
import os
import argparse

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/get_answer', methods=['POST'])
def get_answer():
    request_data = request.json
    query = request_data.get('query')
    if not query:
        return jsonify({"error": "Query is missing in the request"}), 400

    args = parse_args_from_request(request_data)  # Define a function to parse args from the request
    result = main(args)
    return jsonify({"result": result})


def parse_args_from_request(request_data):
    # Parse the open_api_key and faiss_save_path from the request JSON
    os.environ["OPENAI_API_KEY"] = args.OPEN_API_KEY
    faiss_save_path = 'faiss_index'
    
    return argparse.Namespace(
        OPEN_API_KEY=open_api_key,
        faiss_save_path=faiss_save_path,
        query=request_data['query'],
        type=request_data['type']
    )


def main(args):
    # Vectorize the question sentences. Then, 5 closely related sentences were extracted from the FAISS database
    os.environ["OPENAI_API_KEY"] = args.OPEN_API_KEY
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(args.faiss_save_path, embeddings)

    query = args.query
    embedding_vector = embeddings.embed_query(query)

    docs_and_scores = db.similarity_search_with_score_by_vector(embedding_vector, k=5)
    if len(docs_and_scores) != 5:
        raise AssertionError("The number of documents returned by the similarity search is not 5.")

    text = "「"
    for doc in docs_and_scores:
        text += doc[0].page_content
    text += "」"

    # Using the PromptTemplate class, create a prompt that refers to the 5 sentences above and returns the answer to the question.
    template = query
    prompt = PromptTemplate(
        input_variables=[],
        template=text + ("Write a prompt for your use case" if args.type == "child" else "Write a prompt for your use case.") + template,
    )
    # Throw the created prompt to gpt-3.5-turbo, get the answer, and display it.
    llm = OpenAI(model_name="gpt-3.5-turbo")
    print(llm(prompt.format()))
    return llm(prompt.format())


if __name__ == "__main__":
    app.run(debug=True)

