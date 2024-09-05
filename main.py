from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

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
You are an expert IT administrator assistant. Your task is to provide clear, step-by-step instructions based on the following administrator guides and the user's question.

Administrator Guide Content:
{documents}

User Question: {question}

Please provide a response that:
1. Clearly outlines the steps to accomplish the task
2. Includes any relevant safety precautions or prerequisites
3. Mentions any specific commands or configurations, if applicable
4. Warns about potential pitfalls or common mistakes
5. Suggests any best practices related to the task

If the provided documents do not contain enough information to fully answer the question, please state this clearly and provide the best possible advice based on the available information.

Step-by-step Answer:
"""
prompt = PromptTemplate(
    input_variables=["documents", "question"],
    template=prompt_template,
)

# Create the runnable sequence
qa_chain = (
    {"documents": RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def answer_question(query):
    # Retrieve relevant documents
    docs = vectorstore.similarity_search(query, k=3)
    docs_content = "\n".join([doc.page_content for doc in docs])
    
    # Generate an answer using the retrieved documents
    answer = qa_chain.invoke({"documents": docs_content, "question": query})
    
    return answer

# Main loop to continuously prompt for questions
while True:
    # Prompt the user for a question
    query = input("\nPlease enter your question (or type 'exit' to quit): ")
    
    # Check if the user wants to exit
    if query.lower() == 'exit':
        print("Thank you for using the Q&A system. Goodbye!")
        break
    
    # Process the user's query
    response = answer_question(query)
    print("\nAnswer:", response)