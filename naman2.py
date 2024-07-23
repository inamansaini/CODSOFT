def chatbot_response(user_input):
    user_input = user_input.lower()
    
    if user_input in ["hello", "hi", "hey"]:
        return "Hello! How can I assist you today?"
    elif "who are you" in user_input or "what are you" in user_input:
        return "I am a simple chatbot."
    elif "weather" in user_input:
        return "Check your local weather app for updates."
    elif "time" in user_input:
        return "Check the clock on your device."
    elif "help" in user_input:
        return "Sure! What do you need help with?"
    else:
        return "I'm not sure how to respond to that."

def chat():
    print("Chatbot: Hi! I'm your assistant. Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        print("Chatbot:", chatbot_response(user_input))

chat()