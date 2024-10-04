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
            temperature=0.4,
            api_key=constants.APIKEY
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Answer the user's questions based on the {self.context} context: {{context}}. Start the conversation with: If given incomplete questions i.e., one-word input, please ask follow up questions. If the user asks a general about finding specific data, please tell the user to search the data topic of interest in the search bar. Only if the user asks for the specific path to the indicator, do you give with specific steps to navigate. Also, after providing an answer, suggest 2 specific follow-up questions directly related to the context of the user's question and provided the questions that you think the user may ask next. If answer responds No to the follow-up questions, continue with conversation."),
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
