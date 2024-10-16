# Configure OpenAI KEY
import openai as OpenAI
from dotenv import load_dotenv
from openai import OpenAI
import os
import gradio as gr

# Utility package for English Prompts
import utils
import json
from datetime import datetime


# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from the environment variable
#api_key = os.getenv("HYPERBOLIC_API_KEY") # 'ollama'  
#model = "meta-llama/Llama-3.2-90B-Vision-Instruct"  # "gpt-4o-mini"
#base_url = "https://api.hyperbolic.xyz/v1/" # ollama 'http://localhost:11434/v1/'

api_key = os.getenv("OPENAI_API_KEY") # 'ollama'  
model = "gpt-4o-mini"  # "gpt-4o-mini"
base_url = None


client = OpenAI(
    base_url=base_url,
    api_key=api_key
)


def get_completion_from_messages(messages, 
                                 model="gpt-4o-mini", 
                                 temperature=0, 
                                 max_tokens=500):
    '''
    Encapsulate a function to access LLM

    Parameters: 
    messages: This is a list of messages, each message is a dictionary containing role and content. The role can be 'system', 'user' or 'assistant', and the content is the message of the role.
    model: The model to be called, default is gpt-4o-mini (ChatGPT) 
    temperature: This determines the randomness of the model output, default is 0, meaning the output will be very deterministic. Increasing temperature will make the output more random.
    max_tokens: This determines the maximum number of tokens in the model output.
    '''
    response = client.chat.completions.create(
        messages=messages,
        model=model, 
        temperature=temperature, # This determines the randomness of the model's output
        max_tokens=max_tokens, # This determines the maximum number of tokens in the model's output
    )

    return response.choices[0].message.content

def process_user_message(user_input, all_messages, debug=True):
    """
    Preprocess user messages
    
    Parameters:
    user_input : User input
    all_messages : Historical messages
    debug : Whether to enable DEBUG mode, enabled by default
    """
    # Delimiter
    delimiter = "```"
    
    # Step 1: Use OpenAI's Moderation API to check if the user input is compliant or an injected Prompt
    response = client.moderations.create(input=user_input)
    moderation_output = response.results[0]

    # The input is non-compliant after Moderation API check
    if moderation_output.flagged:
        print("Step 1: Input rejected by Moderation")
        return "Sorry, your request is non-compliant"

    # If DEBUG mode is enabled, print real-time progress
    if debug: 
        print("Step 1: Input passed Moderation check")
        print(f"\n**user_input**: {user_input}\n\n")
    
    # Step 2: Extract products and corresponding categories 
    category_and_product_response = utils.find_category_and_product_only(
        user_input, utils.get_products_and_category())
    #print(category_and_product_response)
    # Convert the extracted string to a list
    category_and_product_list = utils.read_string_to_list(category_and_product_response)
    #print(category_and_product_list)

    if debug: print("Step 2: Extracted product list")

    # Step 3: Find corresponding product information
    product_information = utils.generate_output_string(category_and_product_list)
    if debug: 
        print("Step 3: Found information for extracted products")
        print(f"\n**product_information**: {product_information}\n\n")

    # Step 4: Generate answer based on information
    system_message = f"""
    You are a customer service assistant for a large electronic store. \
    Respond in a friendly and helpful tone, with concise answers. \
    Make sure to ask the user relevant follow-up questions.
    """
    # Insert message
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"{delimiter}{user_input}{delimiter}"},
        {'role': 'assistant', 'content': f"Relevant product information:\n{product_information}"}
    ]
    # Get GPT3.5's answer
    # Implement multi-turn dialogue by appending all_messages
    final_response = get_completion_from_messages(all_messages + messages)
    if debug:print("Step 4: Generated user answer")
    # Add this round of information to historical messages
    all_messages = all_messages + messages[1:]

    # Step 5: Check if the output is compliant based on Moderation API
    response = client.moderations.create(input=final_response)
    moderation_output = response.results[0]

    # Output is non-compliant
    if moderation_output.flagged:
        if debug: print("Step 5: Output rejected by Moderation")
        return "Sorry, we cannot provide that information"

    if debug: print("Step 5: Output passed Moderation check")

    # Step 6: Model checks if the user's question is well answered
    user_message = f"""
    Customer message: {delimiter}{user_input}{delimiter}
    Agent response: {delimiter}{final_response}{delimiter}

    Does the response sufficiently answer the question? answer Yes or No
    """
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]
    # Request model to evaluate the answer
    evaluation_response = get_completion_from_messages(messages)
    if debug: print("Step 6: Model evaluated the answer")

    # Step 7: If evaluated as Y, output the answer; if evaluated as N, feedback that the answer will be manually corrected
    if "Y" in evaluation_response:  # Use 'in' to avoid the model possibly generating Yes
        if debug: print("Step 7: Model approved the answer.")
        return final_response, all_messages, category_and_product_response
    else:
        if debug: print("Step 7: Model disapproved the answer.")
        neg_str = "I apologize, but I cannot provide the information you need. I will transfer you to a human customer service representative for further assistance."
        return neg_str, all_messages

#Visual Interface
#log messages
messages_log = 'messages_log.json'

def log_messages(new_element, filepath):
 # Update the messages_log with the new assistant response
    filepath = 'messages_log.json'

    try:
        with open(filepath, "r") as file:
            # Check if the file is empty
            if file.read().strip() == "":
                data = []  # Initialize with an empty list or dictionary as needed
            else:
                file.seek(0)  # Move the cursor back to the start of the file
                data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, start with an empty list or dictionary
        data = []
    except json.JSONDecodeError:
        # If there is a JSON decoding error, handle it by initializing empty data
        print("Error: The JSON file is not properly formatted.")
        data = []
    
    # Assuming the data is a list of items
    data.append(new_element)

    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)

# Initialize the context as an empty list to keep track of the conversation history
context = []

# Function to collect and process user messages
def collect_messages_en(input_text, debug=True):
    global context  # Use the global messages_log to track all messages

    if debug: print(f"User Input = {input_text}")
    if input_text == "":
        return
    #context = get_messages()
    # Process the user input and get a response
    response, context, product_category = process_user_message(input_text, context, debug=debug)
    context.append({'role':'assistant', 'content':f"{response}"})

    # Get the current timestamp
    current_timestamp = datetime.now()
    formatted_timestamp = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")

    #log the messages
    log_messages({'time_stamp': formatted_timestamp, 'user_input': input_text, 'AI_response': response, 'metadata': product_category}, messages_log)
    # Return the response to be displayed in the Gradio interface
    return response

# Create a Gradio interface for interacting with the assistant
demo = gr.Interface(
    fn=collect_messages_en, 
    inputs=gr.Textbox(lines=3, label="Inquiries", placeholder="Ask us anything..."),
    outputs="text",
    title="Customer Service Assistant",
    description="Ask questions about products or services.",
)

if __name__ == "__main__":
    demo.launch()

# user_input = "tell me about the smartx pro phone and the fotosnap camera, the dslr one. Also what tell me about your tvs"

