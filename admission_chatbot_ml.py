import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import joblib
from prompt_toolkit import prompt
from prompt_toolkit import print_formatted_text as print

class AdmissionChatbot:
    def __init__(self):
        self.model = None
        self.context = []

    def load_data(self, file_path):
        return pd.read_csv(file_path)

    def train_model(self, data):
        model = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
        model.fit(data['question'], data['answer'])
        self.model = model
        joblib.dump(model, '../admission_chatbot_model.pkl')

    def load_model(self, model_path):
        self.model = joblib.load(model_path)

    def update_context(self, user_input):
        self.context.append(user_input)
        if len(self.context) > 5:
            self.context.pop(0)

    def get_contextual_input(self, user_input):
        return ' '.join(self.context + [user_input])

    def get_answer(self, user_input):
        contextual_input = self.get_contextual_input(user_input)
        return self.model.predict([contextual_input])[0]

    def chat(self):
        print("Welcome to the College Admission Helpdesk! How can I assist you with your admission queries?")
        while True:
            user_input = prompt("You: ")
            if user_input.lower() in ["bye", "exit", "quit"]:
                print("Thank you for contacting the Admission Helpdesk. Goodbye!")
                break
            self.update_context(user_input)
            response = self.get_answer(user_input)
            print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot = AdmissionChatbot()

    # Load data and train model
    data = chatbot.load_data(r'C:\Users\vinot\Documents\project 2\admission_qa_dataset.csv')  # Specify the path to your CSV file here
    chatbot.train_model(data)
    chatbot.load_model('../admission_chatbot_model.pkl')

    # Start chat
    chatbot.chat()
