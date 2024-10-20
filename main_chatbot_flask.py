# Import necessary libraries
import os
import csv
import logging
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # Enable CORS
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain.chains.history_aware_retriever import create_history_aware_retriever

# Load environment variables
load_dotenv()

# Create date-specific log filenames
current_date = datetime.now().strftime("%Y-%m-%d")
log_filename_csv = f"chat_logs_{current_date}.csv"
log_filename_txt = f"chat_logs_{current_date}.txt"

# Set up logs for CSV file
if not os.path.exists(log_filename_csv):
    with open(log_filename_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Response', 'Response Time'])

# Set up logs for TXT file
logging.basicConfig(filename=log_filename_txt, level=logging.INFO, format='%(asctime)s - %(message)s')

# Initialise Flask app with CORS enabled
app = Flask(__name__)
CORS(app)

class Assistant:
    def __init__(self, file_path, context):
        self.context = context
        self.docs = self.load_text(file_path)
        self.vectorStore = self.create_db(self.docs)
        self.chain = self.create_chain()
        self.chat_history = []  # Chat history for each session
        self.question_count = 0

    # Load text from file
    def load_text(self, file_path):
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()

    # Create vector database
    def create_db(self, docs):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
        return Chroma.from_documents(docs, embedding=embedding)

    # Create conversation chain with new prompt template
    def create_chain(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5,
            api_key=openai_api_key
        )

        # New prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a friendly and helpful AI assistant chatbot guiding users on how to navigate the Atlas map, based on {context}. "
             "Your primary goal is to assist users with navigation while being approachable and open to casual conversation."),
            ("system", "Context: {context}"),
            ("system", 
             "Instructions for {context}:"
             "\n1. Engage with users in a friendly manner, responding positively to greetings."
             "\n   For example, if the user says 'Hello,' respond warmly and ask how you can help."
             "\n2. If the user responds with phrases such as ‘ok,’ ‘thank you,’ ‘I understand,’ or ‘what?’, make sure to keep the conversation flowing smoothly. After acknowledging their response, kindly ask if they have any other questions or need further assistance."
             "\n   For positive responses like ‘ok’ or ‘thank you,’ gently ask if there’s anything else they need help with or if they have more questions."
             "\n   If the user says ‘what?’ or seems confused, acknowledge their uncertainty and offer to clarify or provide more information. This way, you maintain a friendly and engaging conversation while helping them with their needs."
             "\n3. Clarify vague, ambiguous, or one-word queries before providing a full response. If the user's input is unclear, misspelled, or potentially mistyped, ask for clarification."
             "\n4. For data search queries on a specific theme or subcategory, respond exactly with: 'To find data on [theme], open the Atlas map, navigate to the right-hand side pane, and type [theme] in the search box. If data is available, select the subcategory of interest from the drop-down options.'"
             "\n   For example, if the user asks about 'assault,' respond exactly with: 'To find data on assault, open the Atlas map, navigate to the right-hand side pane, and type 'assault' in the search box. If data is available, select the subcategory of interest from the drop-down options.'"
             "\n   For example, if the user asks about 'suicide,' respond exactly with: 'To find data on suicide, open the Atlas map, navigate to the right-hand side pane, and type 'suicide' in the search box. If data is available, select the subcategory of interest from the drop-down options.'"
             "\n   For example, if the user asks about 'alcohol-related hospital admissions,' respond exactly with: 'To find data on hospital admissions, open the Atlas map, navigate to the right-hand side pane, and type 'alcohol related' in the search box. If data is available, select the subcategory of interest from the drop-down options.'"
             "\n5. If the user asks about 'latest data' or data for a specific period, let them know they should first search for their theme of interest. They can then filter the data by clicking the 'calendar' icon and selecting the relevant year."
             "\n6. Always relate your responses to the user's original query, regardless of the theme or indicator."
             "\n7. Never interpret the data, even if the user asks you to. Instead, explain that you can only assist with map navigation queries."
             "\n8. If data is available, provide the exact information exactly as it appears in the text file, without making any changes."
             "\n   For example, if the user asks how results are calculated, always respond with the exact wording provided: 'For more information about how results were calculated, refer to Homepage -> Main Menu options -> Technical Information.'"
             "\n9. If you provide information about external resources, such as the Australian Bureau of Statistics (ABS) website, include a correct and functional clickable link to the relevant site."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("system", 
             "Ensure your responses are concise, clear, and helpful. Limit each response to a maximum of three sentences, and use Australian English spelling.")
        ])

        chain = create_stuff_documents_chain(
            llm=model,
            prompt=prompt
        )

        retriever = self.vectorStore.as_retriever(search_kwargs={"k": 1})
        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", f"Generate a search query based on the conversation about {self.context}.")
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

    # Process user input and generate response
    def process_chat(self, question):
        start_time = datetime.now()
        
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
        main_answer, follow_up = self.split_response(response["answer"])
        
        self.chat_history.append(AIMessage(content=main_answer))
        self.question_count += 1
        
        return main_answer, follow_up

    # Log to CSV file
    def log_to_csv(self, question, answer, response_time):
        with open(log_filename_csv, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([question, answer, response_time])

    # Log chat entry to TXT file
    def log_chat_history(self, question, answer):
        logging.info(f"User: {question}")
        logging.info(f"Assistant: {answer}")

    # Split response into main answer and follow-up question
    def split_response(self, response):
        parts = response.split("Would you like to know more about:")
        main_answer = parts[0].strip()
        follow_up = "Would you like to know more about:" + parts[1].strip() if len(parts) > 1 else ""
        return main_answer, follow_up

class MapAssistant(Assistant):
    def __init__(self):
        super().__init__('prepared_data_ver3.txt', 'map navigation')

# Define chat endpoint (using POST)
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    assistant = MapAssistant()  # Re-initialise for every request
    main_response, follow_up = assistant.process_chat(user_message)
    return jsonify({
        "reply": main_response,
        "follow_up": follow_up
    })

# Define log download endpoint
@app.route("/download_logs", methods=["GET"])
def download_logs():
    try:
        return send_file(log_filename_csv, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "Log file not found."}), 404

# Define endpoint for downloading TXT logs
@app.route("/download_logs_txt", methods=["GET"])
def download_logs_txt():
    try:
        return send_file(log_filename_txt, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
