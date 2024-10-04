import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()

import constants

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
            model="gpt-4o-mini",  # Make sure this is the correct model name
            temperature=0.2,
            api_key=constants.APIKEY
        )

        prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant chatbot specifically focused giving a tutorial on how to navigate the Atlas map, based on {context}. "
               "Your primary goal is to help users with {context} only."),
    ("system", "Context: {context}"),
    ("system", "Instructions for {context}:"
               "\n1. If given a one-word or vague query, ask for clarification before proceeding."
               "\n2. Once the query is clear, guide the user as follows:"
               "\n   a. If the context is map navigation:"
               "\n      - Direct users to the Atlas maps"
               "\n      - Instruct users to use the theme search box in Atlas maps"
               "\n      - Explain that available data will appear as a dropdown"
               "\n   b. If the context is data dashboard:"
               "\n      - Guide users to the data dashboard"
               "\n      - Instruct users to select a region first"
               "\n3. Only provide more specific guidance from MapAssistant or DashboardAssistant if the user requests additional help after the initial instructions."
               "\n4. Always relate your responses back to the user's original query."),
        MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("system", "Remember to be concise, clear, and helpful in your responses - give a maximum of 3 sentences. "
               "Focus exclusively on {context} and do not discuss the other topic unless explicitly asked."
               "Always start with general guidance before providing specific details.")
        ])

        chain = create_stuff_documents_chain(
            llm=model,
            prompt=prompt
        )

        retriever = self.vectorStore.as_retriever(search_kwargs={"k": 1})

        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", f"Given the above conversation about {self.context}, generate a search query to look up relevant information")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm=model,
            retriever=retriever,
            prompt=retriever_prompt
        )

        return create_retrieval_chain(
            history_aware_retriever,
            chain
        )

    def process_chat(self, question):
        response = self.chain.invoke({
            "input": question,
            "chat_history": self.chat_history,
            "context": self.context
        })
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response["answer"]))
        return response["answer"]

    def reset_chat_history(self):
        self.chat_history = []

class MapAssistant(Assistant):
    def __init__(self):
        super().__init__('Raw data - maps.txt', 'map navigation')

class DashboardAssistant(Assistant):
    def __init__(self):
        super().__init__('Data1.txt', 'data dashboard')

def create_assistant(topic):
    if topic == "map":
        return MapAssistant()
    elif topic == "dashboard":
        return DashboardAssistant()
    else:
        raise ValueError("Invalid topic")

def select_topic():
    while True:
        print("Hello! What would you like help with today? Map navigation or Data dashboard?")
        user_choice = input("You: ").lower()
        if "map" in user_choice:
            return "map"
        elif "dashboard" in user_choice:
            return "dashboard"
        else:
            print("I'm sorry, I didn't understand your choice. Please type 'map' or 'dashboard'.")

if __name__ == '__main__':
    topic = select_topic()
    assistant = create_assistant(topic)

    print(f"Great! I'll help you with {assistant.context}. What would you like to know?")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Ending conversation. Goodbye!")
            break

        # Topic switch logic
        other_topic = "dashboard" if topic == "map" else "map"
        if other_topic in user_input.lower():
            print(f"It seems you're asking about {other_topic}. Would you like to switch topics? (yes/no)")
            switch = input("You: ").lower()
            if switch in ['yes', 'y']:
                topic = other_topic
                assistant = create_assistant(topic)
                assistant.reset_chat_history()  # Reset chat history when switching topics
                print(f"Switched to {assistant.context}. How can I help you?")
                continue

        try:
            response = assistant.process_chat(user_input)
            print("Assistant:", response)
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Let's try that again. Could you rephrase your question?")
