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

    def load_text(self, file_path):
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()

    def create_db(self, docs):
        embedding = OpenAIEmbeddings(openai_api_key=constants.APIKEY)
        return Chroma.from_documents(docs, embedding=embedding)

    def create_chain(self):
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4,
            api_key=constants.APIKEY
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Answer the user's questions based on the {self.context} context: {{context}}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
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

    def process_chat(self, question, chat_history):
        response = self.chain.invoke({
            "input": question,
            "chat_history": chat_history
        })
        return response["answer"]

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
        print("What would you like help with today? Map navigation or Data dashboard?")
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

    print(f"Great! I'll help you with {topic}. What would you like to know?")

    chat_history = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Ending conversation. Goodbye!")
            break

        other_topic = "dashboard" if topic == "map" else "map"
        if other_topic in user_input.lower():
            print(f"It seems you're asking about {other_topic}. Would you like to switch topics? (yes/no)")
            switch = input("You: ").lower()
            if switch in ['yes', 'y']:
                topic = other_topic
                assistant = create_assistant(topic)
                chat_history = []
                print(f"Switched to {topic}. How can I help you?")
                continue
        
        response = assistant.process_chat(user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        
        print("Assistant:", response)
