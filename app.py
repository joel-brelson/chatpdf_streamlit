from flask import Flask, render_template, request, jsonify, redirect
import signal
import os
import pinecone
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
import time
app = Flask(__name__)

# Set the folder where uploaded PDFs will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

llm = None
qa_chain = None

# Set default values for Replicate API token and Pinecone API key
DEFAULT_REPLICATE_API_TOKEN = 'r8_EGcxcWOmYmujXqdqOzYlgZ9kQSUwSKl2rLeKp'
DEFAULT_PINECONE_API_KEY = '92747e02-8f34-4d91-97a4-e3643e77670e'

def initialize_models(replicate_api=DEFAULT_REPLICATE_API_TOKEN, pinecone_api=DEFAULT_PINECONE_API_KEY):
    start=time.time()
    os.environ['REPLICATE_API_TOKEN'] = replicate_api
    pinecone.init(api_key=pinecone_api, environment='gcp-starter')
    global llm
    llm = Replicate(
        model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
        input={"temperature": 0.75, "max_length": 3000}
    )
    print("models are intialsed")
    print("time to initialise models",time.time()-start)

def initialize_qa_chain(pdf_path):
    start=time.time()
    global qa_chain
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()

    index_name = "pdfchat"
    index = pinecone.Index(index_name)
    vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectordb.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True
    )
    print("qa chain is intialsed")
    print("time to initialise qa chain",time.time()-start)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global llm
    global qa_chain

    if 'pdf_file' not in request.files:
        return redirect(request.url)

    pdf_file = request.files['pdf_file']

    if pdf_file.filename == '':
        return redirect(request.url)

    if pdf_file:
        # Ensure the "uploads" directory exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        # Save the uploaded PDF file to the "uploads" folder
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
        pdf_file.save(pdf_path)

        # Initialize Pinecone and Replicate LLM Model if not initialized
        if llm is None:
            initialize_models()

        if qa_chain is None:
            initialize_qa_chain(pdf_path)

        # Initialize chat history
        chat_history = []

        while True:
            if 'question' in request.form:
                question = request.form.get('question')

                if question.lower() in ["exit", "quit", "q"]:
                    return jsonify({'answer': '', 'exit': True, 'chat_history': chat_history})
                start=time.time()
                result = qa_chain({'question': question, 'chat_history': chat_history})
                answer = result['answer']
                print("result is fetched")
                chat_message = f"You: {question}<br>Bot: {answer}<br>"
                chat_history.append(chat_message)
                print("time to fetch",time.time()-start)
                return jsonify({'answer': answer, 'exit': False, 'chat_message': chat_message, 'chat_history': chat_history})
            else:
                return jsonify({'answer': '', 'exit': False, 'chat_history': chat_history})

if __name__ == '__main__':
    app.run(debug=False)
