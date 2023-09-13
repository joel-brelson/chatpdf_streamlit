import os
import pinecone
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain

llm = None

def initialize_models(replicate_api, pinecone_api):
    os.environ['REPLICATE_API_TOKEN'] = replicate_api
    pinecone.init(api_key=pinecone_api, environment='gcp-starter')
    llm = Replicate(
        model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
        input={"temperature": 0.75, "max_length": 3000}
    )
    return llm  # Return the initialized llm object

def pdfchat(pdf_file_path, replicate_api, pinecone_api):
    global llm  # Use the global llm variable
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()

    if llm is None:  # Check if llm is initialized
        llm = initialize_models(replicate_api, pinecone_api)
        print("LLM is initialized")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()

    index_name = "pdfchat"
    index = pinecone.Index(index_name)
    vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)

    # Set up the Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectordb.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True
    )
    return qa_chain
def chatbot(query):
    result = qa_chain({'question': query, 'chat_history': chat_history})
    print('Answer: ' + result['answer'] + '\n')
    chat_history.append((query, result['answer']))

# Example usage
qa_chain=pdfchat(
    r"C:\Users\Sathvika\Documents\chatpdf\uploads\7. Technical Program Manager.pdf",
    "r8_EGcxcWOmYmujXqdqOzYlgZ9kQSUwSKl2rLeKp",
    "92747e02-8f34-4d91-97a4-e3643e77670e",
)

chat_history = []
while True:
    query = input('Your Question: ')
    if query.lower() in ["exit", "quit", "q"]:
        print('Exiting')
        break
    result = qa_chain({'question': query, 'chat_history': chat_history})
    print('Answer: ' + result['answer'] + '\n')
    chat_history.append((query, result['answer']))
