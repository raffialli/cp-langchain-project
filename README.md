# Chat with Check Point Admin Guides

## Description

This project implements a question-answering system designed to assist Check Point administrators. It uses natural language processing and machine learning techniques to provide answers to technical questions based on a set of administrator guides. The system reads PDF documents, processes them, and uses OpenAI's language models to generate relevant responses to user queries.

## Features

- Loads and processes multiple PDF documents
- Utilizes FAISS for efficient similarity search of document contents
- Leverages OpenAI's language models for generating human-like responses
- Provides step-by-step answers to IT administration questions
- Includes safety precautions, prerequisites, and best practices in responses

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Anaconda (if using Conda)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/raffialli/cp-langchain-project.git
   cd cp-langchain-project
   ```

2. Create a virtual environment (optional but recommended. I am using Anaconda):
   ```
    conda create --name cp-langchain-project python=3.8
    conda activate cp-langchain-project
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your OpenAI API key to the file:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

## Usage

1. Place your PDF administrator guides in a folder named `checkpoint_docs` in the project root.

2. Run the script:
   ```
   python main.py
   ```

3. Enter your questions when prompted. The system will provide detailed answers based on the content of your administrator guides.

4. Type 'exit' when you're done to quit the program.

## How It Works

1. The system loads all PDF documents from the `checkpoint_docs` folder.
2. It creates embeddings for the document contents and stores them in a FAISS index for quick retrieval.
3. When a question is asked, the system finds the most relevant document sections using similarity search.
4. These relevant sections, along with the question, are sent to the OpenAI language model.
5. The model generates a detailed, step-by-step response, which is then presented to the user.

## Customization

- You can adjust the `temperature` parameter in the `OpenAI` initialization to control the creativity of the responses.
- Modify the prompt template in the code to change the style or focus of the answers.
- Adjust the `k` parameter in the `similarity_search` function to retrieve more or fewer relevant documents for each query.
