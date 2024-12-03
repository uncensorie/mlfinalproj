import requests


class ChatSimulator:
    def __init__(self, api_key, model_name, context_window=30):
        self.api_key = api_key
        self.model_name = model_name
        self.context_window = context_window
        self.messages = []

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.context_window:
            self.messages.pop(0)

    def get_response(self):
        url = "https://api.deepinfra.com/v1/openai/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"model": self.model_name, "messages": self.messages}
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        return response_json["choices"][0]["message"]["content"]

    def chat(self):
        system_prompt = input("Enter system prompt (press Enter for default): ")
        if not system_prompt:
            system_prompt = "Use <thinking> and <output>"
        self.add_message("system", system_prompt)
        print("Type 'exit' or 'quit' to end the chat.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat.")
                break
            self.add_message("user", user_input)
            response = self.get_response()
            self.add_message("assistant", response)
            print(f"Assistant: {response}")


if __name__ == "__main__":
    api_key = input("Enter your API key: ")
    model_name = input("Enter the model name: ")
    chat_simulator = ChatSimulator(api_key, model_name)
    chat_simulator.chat()
