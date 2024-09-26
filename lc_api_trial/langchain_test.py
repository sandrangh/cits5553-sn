import os
import sys

# Python file with API key
import constants

# Function to load text files
from langchain_community.document_loaders import TextLoader
# Function to load CSV files
from langchain_community.document_loaders.csv_loader import CSVLoader

from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI

# OPTIONAL CODE: set the API key
# OPTIONAL CODE: os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Get the query from command line arguments
query = sys.argv[1]

# Load the document
loader = CSVLoader('PromptsResponses.csv')

# Create the embeddings
embeddings = OpenAIEmbeddings()

# Create an LLM with the API key
llm = OpenAI(api_key=constants.APIKEY)

# Create the index
index = VectorstoreIndexCreator(
    embedding=embeddings,
    vectorstore_cls=Chroma
).from_loaders([loader])

# Perform the query and print the result
print(index.query(query, llm=llm))
