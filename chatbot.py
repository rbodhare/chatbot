import fitz  # PyMuPDF
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

# Load pre-trained model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

# Define the path to your PDF file
file_path = 'C:/Users/Admin/Desktop/Chatbot/input4.pdf'

# Read the context from the PDF file
doc = fitz.open(file_path)
context = ""
for page_num in range(doc.page_count):
    page = doc[page_num]
    context += page.get_text()

def get_answer(question, context):
    # Split the context into chunks of 450 tokens
    chunk_size = 450
    context_chunks = [context[i:i + chunk_size] for i in range(0, len(context), chunk_size)]
    
    max_score = 0
    best_answer = "I'm sorry, I don't understand."
    
    for chunk in context_chunks:
        # Tokenize the input question and context chunk
        inputs = tokenizer(question, chunk, return_tensors='pt')
        
        # Get model output
        outputs = model(**inputs)
        
        # Get start and end position of answer
        start_position = torch.argmax(outputs.start_logits)
        end_position = torch.argmax(outputs.end_logits)
        
        # Get the answer from the context chunk
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_position:end_position + 1]))
        score = torch.max(outputs.start_logits) + torch.max(outputs.end_logits)
        
        # Update best answer if the current chunk's answer score is higher
        if score > max_score:
            max_score = score
            best_answer = answer
    
    return best_answer

print("Welcome to the Chatbot!")
print("Type 'exit' to end the chat.")

while True:
    # Get user input
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    
    # Get chatbot response
    response = get_answer(user_input, context)
    print("Chatbot:", response)
