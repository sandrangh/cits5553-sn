import os
import csv
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import constants

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
        writer.writerow(['Question', 'Response'])  # Write header row if file is new

# Set up logs for TXT file
logging.basicConfig(filename=log_filename_txt, level=logging.INFO, format='%(asctime)s - %(message)s')

class Assistant:
    def __init__(self, file_path):
        self.context = self.load_text(file_path)
        self.chain, self.prompt = self.create_chain()
        self.chat_history = []

    def load_text(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def create_chain(self):
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=constants.APIKEY
        )

        # Create prompt using context loaded from text file
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a helpful AI assistant chatbot guiding users on how to navigate the Atlas map, based on the following information:\n{self.context}\nYour primary goal is to help users with map navigation and provide relevant information regarding the themes and subcategories available."),
            ("system", "Instructions for understanding the context of the user's question:"
                       "\n1. Engage with users in a friendly manner, responding positively to greetings."
                       "\n   For example, if the user says 'Hello,' respond warmly and ask how you can help."
                       "\n2. Clarify vague, ambiguous, or one-word queries before providing a full response. If the user's input is unclear, misspelled, or potentially mistyped, ask for clarification."
                       "\n3. For data search queries on a specific theme or subcategory, respond exactly with: 'To find data on [theme], open the Atlas map, navigate to the right-hand side pane, and type [theme] in the search box. If data is available, select the subcategory of interest from the drop-down options.'"
                       "\n   For example, if the user asks about 'assault,' respond exactly with: 'To find data on assault, open the Atlas map, navigate to the right-hand side pane, and type 'assault' in the search box. If data is available, select the subcategory of interest from the drop-down options.'"
                       "\n   For example, if the user asks about 'suicide,' respond exactly with: 'To find data on suicide, open the Atlas map, navigate to the right-hand side pane, and type 'suicide' in the search box. If data is available, select the subcategory of interest from the drop-down options.'"
                       "\n   For example, if the user asks about 'alcohol-related hospital admissions,' respond exactly with: 'To find data on hospital admissions, open the Atlas map, navigate to the right-hand side pane, and type 'alcohol related' in the search box. If data is available, select the subcategory of interest from the drop-down options.'"
                       "\n4. If the user asks about 'latest data' or data for a specific period, let them know they should first search for their theme of interest. They can then filter the data by clicking the 'calendar' icon and selecting the relevant year."
                       "\n5. Always relate your responses to the user's original query, regardless of the theme or indicator."
                       "\n6. Never interpret the data, even if the user asks you to. Instead, explain that you can only assist with map navigation queries."
                       "\n7. If data is available, provide the exact information exactly as it appears in the text file, without making any changes."
                       "\n   For example, if the user asks how results are calculated, always respond with the exact wording provided: 'For more information about how results were calculated, refer to Homepage -> Main Menu options -> Technical Information.'"
                       "\n8. If you provide information about external resources, such as the Australian Bureau of Statistics (ABS) website, include a correct and functional clickable link to the relevant site."),
            ("human", "{input}"),
            ("system", "Ensure your responses are concise, clear, and helpful. Limit each response to a maximum of three sentences, and use Australian English spelling.")
        ])

        return model, prompt

    def process_chat(self, question):
        # Combine context with user's question
        full_prompt = (
            f"You are a helpful AI assistant chatbot guiding users on how to navigate the Atlas map based on the following information:\n"
            f"{self.context}\n"
            f"User's question: {question}\n"
            "Your primary goal is to help users with map navigation only."
        )

        # Invoke model with combined prompt
        response = self.chain.invoke(full_prompt)  # Pass full prompt directly

        # Access response content properly
        if isinstance(response, dict) and "answer" in response:
            answer = response["answer"]
        else:
            # If response is not a dictionary, access content differently
            answer = response.content if hasattr(response, 'content') else "I'm not sure how to respond."

        self.log_to_csv(question, answer)
        self.log_chat_history(question, answer)

        self.chat_history.append(question)  # Log user question
        self.chat_history.append(answer)  # Log assistant response
        return answer

    # Log to CSV file
    def log_to_csv(self, question, answer):
        with open(log_filename_csv, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([question, answer])  # Write question and response

    # Log chat entry to TXT file
    def log_chat_history(self, question, answer):
        logging.info(f"User: {question}")
        logging.info(f"Assistant: {answer}")

class MapAssistant(Assistant):
    def __init__(self):
        super().__init__('prepared_data_ver3.txt')  # Load data from text file

# Main execution
if __name__ == '__main__':
    assistant = MapAssistant()

    print("Hello! Welcome to the Atlas Map Navigation Assistant! Are you new to our interactive map platform? (Yes/No)")

    # Get user's experience level
    while True:
        user_response = input("You: ").lower()
        if user_response in ['yes', 'y', 'no', 'n']:
            break
        else:
            print("Please answer with 'Yes' or 'No'.")

    # Handle new user
    if user_response in ['yes', 'y']:
        assistant.is_new_user = True
        print("Great! Let's start by familiarising you with the map platform.")
        print("You can start by reading the help screens. Please follow these steps:")
        print("1. Click on the Atlas map.")
        print("2. Navigate to the right-hand side pane.")
        print("3. Click the 'i' icon in the top right-hand corner.")
        print("This will open the help screens. There are three screens covering different aspects of the platform: the National scale, Atlas menu items, and map interactions.")
        print("\nWhat specific question can I help you with first?")
    else:
        print("Welcome back! What can I help you with today?")

    # Main conversation loop
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Ending conversation. Goodbye!")
            break

        try:
            response = assistant.process_chat(user_input)
            print("Assistant:", response)
            print("Assistant:", "Is there another question I can help you with?")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Let's try that again. Could you rephrase your question?")
