# Import required libraries
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Ensure the OpenAI API key is set
# This checks if the OPENAI_API_KEY environment variable is set and raises an error if it's not
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set. Please set it before running the script.")

# Set up the OpenAI Language Model (LLM)
# This creates an instance of the OpenAI LLM with a temperature of 0.7 for some randomness in responses
llm = OpenAI(temperature=0.7, openai_api_key=api_key)

# Directory containing PDF documents
# This specifies the folder where the PDF documents are stored
doc_path = "checkpoint_docs"

# List to store all loaded documents
documents = []

# Loop through each file in the directory and load PDFs
# This section reads all PDF files in the specified directory and loads their content
for filename in os.listdir(doc_path):
    if filename.endswith(".pdf"):
        filepath = os.path.join(doc_path, filename)
        loader = PyMuPDFLoader(filepath)
        documents.extend(loader.load())

# Create embeddings and store them in FAISS for quick retrieval
# This creates vector embeddings for all loaded documents and stores them in a FAISS index for efficient similarity search
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = FAISS.from_documents(documents, embeddings)

# Create a prompt template for the LLM
# This defines the structure of the prompt that will be sent to the LLM
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

# Create the LLM chain with the prompt and model
# This sets up the chain that will process the prompt and generate answers
qa_chain = LLMChain(
    llm=llm,
    prompt=prompt
)

# Function to answer questions
def answer_question(query):
    # Retrieve relevant documents
    # This performs a similarity search to find the 3 most relevant documents for the query
    docs = vectorstore.similarity_search(query, k=3)
    docs_content = "\n".join([doc.page_content for doc in docs])
    
    # Generate an answer using the retrieved documents
    # This sends the relevant document content and the query to the LLM to generate an answer
    answer = qa_chain.run(documents=docs_content, question=query)
    
    return answer

# Prompt the user for a question
query = input("Please enter your question: ")

# Process the user's query
# This calls the answer_question function with the user's input and prints the response
response = answer_question(query)
print("Answer:", response)