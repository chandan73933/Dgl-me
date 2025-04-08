import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager 
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def prepare_docs(pdf):
    docs = []
    metadata = []
    content = []
    pdf_reader = PyPDF2.PdfReader(pdf)
    for index in range(len(pdf_reader.pages)):
        doc_page = {'title': pdf + " page " + str(index + 1),
                    'content': pdf_reader.pages[index].extract_text()}
        docs.append(doc_page)
    for doc in docs:
        content.append(doc["content"])
        metadata.append({"title": doc["title"]})
    print("Content and metadata are extracted from the documents")
    return content, metadata

def get_text_chunks(content, metadata):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=256,
        chunk_overlap=128,
    )
    split_docs = text_splitter.create_documents(content, metadatas=metadata)
    print(f"Documents are split into {len(split_docs)} passages")
    return split_docs

def ingest_into_vectordb(split_docs):
    embeddings = HuggingFaceEmbeddings(model_name=r"sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(split_docs, embeddings)
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)
    return db

template = """[INST]
Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
provided context just say, "answer is not available in the context", don't provide the wrong answer
{question}
{context}
[/INST]
"""

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

def get_conversation_chain(vectordb):
    # Create the Ollama instance without any extra parameters
    ollama_llm = Ollama(
        model="llama3.2:3b"
    )

    retriever = vectordb.as_retriever()
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(template)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer'
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ollama_llm,
        retriever=retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        memory=memory,
        return_source_documents=True
    )
    print("Conversational Chain created for the LLM using the vector store")
    return conversation_chain




def greet(query, file):
    content, metadata = prepare_docs(file)
    split_docs = get_text_chunks(content, metadata)
    vectordb = ingest_into_vectordb(split_docs)
    conversation_chain = get_conversation_chain(vectordb)

    user_question = query
    response = conversation_chain({"question": user_question})
    print("Q: ", user_question)    
    print("A: ", response['answer'])
    return response['answer']

# Chat functionality
pdf_path = r'C:\Users\Chandan.Kumar\Downloads\Transformers.pdf'  # Replace with your uploaded PDF path
print("Chat with the document. Type 'exit' to end the chat.")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Ending the chat.")
        break
    response = greet(user_input, pdf_path)
    print("AI:", response)
