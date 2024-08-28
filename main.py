import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Ensure the OpenAI API key is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set. Please set it before running the script.")

# Set up the OpenAI LLM with the API key
llm = OpenAI(temperature=0.7, openai_api_key=api_key)

# Directory containing PDF documents
doc_path = "checkpoint_docs"

# List to store all loaded documents
documents = []

# Loop through each file in the directory and load PDFs
for filename in os.listdir(doc_path):
    if filename.endswith(".pdf"):
        filepath = os.path.join(doc_path, filename)
        loader = PyMuPDFLoader(filepath)
        documents.extend(loader.load())

# Create embeddings and store them in FAISS for quick retrieval
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = FAISS.from_documents(documents, embeddings)

# Create a prompt template for the LLM
prompt_template = """
Given the following documents, please answer the question.

Documents: {documents}
Question: {question}

Answer:
"""
prompt = PromptTemplate(
    input_variables=["documents", "question"],
    template=prompt_template,
)

# Create the LLM chain with the prompt and model
qa_chain = LLMChain(
    llm=llm,
    prompt=prompt
)

def answer_question(query):
    # Retrieve relevant documents
    docs = vectorstore.similarity_search(query, k=3)
    docs_content = "\n".join([doc.page_content for doc in docs])
    
    # Generate an answer using the retrieved documents
    answer = qa_chain.run(documents=docs_content, question=query)
    
    return answer

# Prompt the user for a question
query = input("Please enter your question: ")

# Process the user's query
response = answer_question(query)
print("Answer:", response)