import os
import csv
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
import constants

# Load environment variables
load_dotenv()

# Define directory for logs
log_dir = 'chat_logs'
os.makedirs(log_dir, exist_ok=True)  # Ensure directory exists

# Create date-specific log filenames within specified directory
current_date = datetime.now().strftime("%Y-%m-%d")
log_filename_csv = os.path.join(log_dir, f"chat_logs_{current_date}.csv")
log_filename_txt = os.path.join(log_dir, f"chat_logs_{current_date}.txt")

# Set up logs for CSV file
if not os.path.exists(log_filename_csv):
    with open(log_filename_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Response', 'Response Time'])  # Write header row if file is new

# Set up logs for TXT file
logging.basicConfig(filename=log_filename_txt, level=logging.INFO, format='%(asctime)s - %(message)s')

class Assistant:
    def __init__(self, file_path, context):
        self.context = context
        self.docs = self.load_text(file_path)
        self.vectorStore = self.create_db(self.docs)
        self.chain = self.create_chain()
        self.chat_history = []

    def load_text(self, file_path):
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()

    def create_db(self, docs):
        embedding = OpenAIEmbeddings(openai_api_key=constants.APIKEY)
        return Chroma.from_documents(docs, embedding=embedding)

    def create_chain(self):
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=constants.APIKEY
        )

        # Custom prompt template for Atlas Map
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a friendly and helpful AI assistant chatbot guiding users ONLY on how to navigate the Atlas map, based on {context}."
             "Your primary goal is to assist users with navigation while being approachable and open to casual conversation."),
            ("system", "Context: {context}"),
            ("system", 
             "Instructions for {context}:"
             "\n1. Engage with users in a friendly manner, responding positively to greetings."
             "\n   For example, if the user says 'Hello,' respond warmly and ask how you can help."
             "\n2. If the user responds with phrases such as ‘ok,’ ‘thank you,’ ‘I understand,’ or ‘what?’, make sure to keep the conversation flowing smoothly. After acknowledging their response, kindly ask if they have any other questions or need further assistance."
             "\n   For positive responses like ‘ok’ or ‘thank you,’ gently ask if there’s anything else they need help with or if they have more questions."
             "\n   If the user says ‘what?’ or seems confused, acknowledge their uncertainty and offer to clarify or provide more information."
             "\n3. Clarify vague, ambiguous, or one-word queries before providing a full response. If the user's input is unclear, misspelled, or potentially mistyped, ask for clarification."
             "\n4. Provide few-shot prompting for data search queries on a specific theme or subcategory. Respond as follows based on the user's query:"
             "\n   User: How do I find data on 'assault'?"
             "\n   Assistant: To find data on assault, open the Atlas map, navigate to the right-hand side pane, and type 'assault' in the search box. If data is available, select the subcategory of interest from the drop-down options."
             "\n   User: What about 'suicide'?"
             "\n   Assistant: To find data on suicide, open the Atlas map, navigate to the right-hand side pane, and type 'suicide' in the search box. If data is available, select the subcategory of interest from the drop-down options."
             "\n   User: How can I see data for 'alcohol-related hospital admissions'?"
             "\n   Assistant: To find data on hospital admissions, open the Atlas map, navigate to the right-hand side pane, and type 'alcohol related' in the search box. If data is available, select the subcategory of interest from the drop-down options."
             "\n5. If the user asks about 'latest data' or data for a specific period, let them know they should first search for their theme of interest. They can then filter the data by clicking the 'calendar' icon and selecting the relevant year."
             "\n6. Always relate your responses to the user's original query, regardless of the theme or indicator."
             "\n7. Never interpret the data, even if the user asks you to. Instead, explain that you can only assist with map navigation queries."
             "\n8. If data is available, provide the exact information as it appears in the text file, without making any changes."
             "\n   For example, if the user asks how results are calculated, always respond with the exact wording provided: 'For more information about how results were calculated, refer to Homepage -> Main Menu options -> Technical Information.'"
             "\n9. If you provide information about external resources, such as the Australian Bureau of Statistics (ABS) website, include a correct and functional clickable link to the relevant site."
             "\n10. ALWAYS provide a link if it is available in the data for a specific query."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("system", 
             "Ensure your responses are concise, clear, and helpful. Limit each response to a maximum of three sentences, and use Australian English spelling.")
        ])
        
        chain = create_stuff_documents_chain(llm=model, prompt=prompt)

        retriever = self.vectorStore.as_retriever(search_kwargs={"k": 1})

        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", 
             f"Given the above conversation about {self.context}, generate a search query to look up relevant information")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm=model,
            retriever=retriever,
            prompt=retriever_prompt
        )

        return create_retrieval_chain(history_aware_retriever, chain)

    def process_chat(self, question):
        start_time = datetime.now()  # Timestamp when question is received
        response = self.chain.invoke({
            "input": question,
            "chat_history": self.chat_history,
            "context": self.context
        })
        end_time = datetime.now()  # Timestamp after response is generated

        # Calculate response time
        response_time = (end_time - start_time).total_seconds()

        self.log_to_csv(question, response["answer"], response_time)
        self.log_chat_history(question, response["answer"])

        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response["answer"]))
        return response["answer"], response_time
        
    def log_to_csv(self, question, answer, response_time):
        with open(log_filename_csv, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([question, answer, response_time])
            
    def log_chat_history(self, question, answer):
        logging.info(f"User: {question}")
        logging.info(f"Assistant: {answer}")

    def reset_chat_history(self):
        self.chat_history = []

class MapAssistant(Assistant):
    def __init__(self):
        super().__init__('prepared_data_ver3.txt', 'map navigation')

if __name__ == '__main__':
    assistant = MapAssistant()

    print("Hello! Welcome to the Atlas Map Navigation Assistant! Are you new to our interactive map platform? (Yes/No)")

    while True:
        user_response = input("You: ").lower()
        if user_response in ['yes', 'y', 'no', 'n']:
            break
        else:
            print("Please answer with 'Yes' or 'No'.")

    if user_response in ['yes', 'y']:
        print("Great! Let's start by familiarising you with the map platform.")
        print("1. Click on the Atlas map.")
        print("2. Navigate to the right-hand side pane.")
        print("3. Click the 'i' icon in the top right-hand corner.")
        print("What specific question can I help you with first?")
    else:
        print("Welcome back! What can I help you with today?")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Thank you for using the Atlas Map Navigation Assistant. Goodbye!")
            break

        try:
            # Unpack response and response time from process_chat
            response, response_time = assistant.process_chat(user_input)
            print(f"Assistant: {response}")
            print(f"Response Time: {response_time:.2f} seconds")
            print("Assistant: Is there another question I can help you with?")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Let's try that again. Could you rephrase your question?")
