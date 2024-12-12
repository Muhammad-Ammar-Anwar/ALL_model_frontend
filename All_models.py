import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GPTNeoForCausalLM, GPT2Tokenizer
import pdfplumber
import os

# Define the device for inference (GPU or CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TARGET_LENGTH = 800  # Adjust for a more detailed summary

# Load the fine-tuned model and tokenizer for summarization
fine_tuned_output_dir = "ShahzaibDev/flant5-finetuned-summarizer"
fine_tuned_model = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_output_dir).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_output_dir)

# Load GPT-Neo model for text generation
base_model_name = "EleutherAI/gpt-neo-125M"
gpt_neo_tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
gpt_neo_model = GPTNeoForCausalLM.from_pretrained(base_model_name).to(DEVICE)

# Function to extract text from a PDF file using pdfplumber
def extract_text_from_pdf(pdf_file_path):
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"The file at {pdf_file_path} does not exist.")
    
    with pdfplumber.open(pdf_file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Define the prompt for summarization
def get_prompt(doc):
    """Format prompts for text summarization using FLAN-T5 models."""
    prompt = "Summarize the following document:\n\n"
    prompt += f"{doc}"
    prompt += "\n\n Summary:"
    return prompt

# Generate response (summary) from the model
def get_response(prompt, model, tokenizer):
    """Generate a text summary from the prompt."""
    # Tokenize the prompt
    encoded_input = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=1024  # Adjust for large inputs
    )

    # Move the inputs to the same device as the model (GPU or CPU)
    model_inputs = encoded_input.to(DEVICE)

    # Generate the response
    generated_ids = model.generate(
        **model_inputs,
        max_length=MAX_TARGET_LENGTH,
        num_beams=6,  # Increase the number of beams for better diversity
        early_stopping=True,
        no_repeat_ngram_size=3,  # Prevent repetition
        temperature=0.7,  # Increase randomness
        top_k=50  # Control the randomness of the output
    )

    # Decode the response back to text
    decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return decoded_output

# Function for inference with GPT-Neo model
def infer_with_gpt_neo(prompt, model, tokenizer, max_length=200):
    """Generate a response from the GPT-Neo model given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,  # Controls randomness
        top_p=0.9,  # Top-p sampling for diverse outputs
        do_sample=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI
def main():
    st.title("Text Processing and GPT-Neo Generation")
    
    # Dropdown menu for selecting task (Summarization or Text Generation)
    task_option = st.selectbox("Choose an option:", ["Summarization", "GPT-Neo Text Generation"])

    # User input box
    user_input = st.text_area("Enter the text or prompt:", height=150)

    # Button to trigger generation or summarization
    if st.button("Generate"):
        if task_option == "Summarization":
            if user_input:
                prompt = get_prompt(user_input)
                summary = get_response(prompt, fine_tuned_model, tokenizer)
                st.subheader("Generated Summary:")
                st.write(summary)
            else:
                st.error("Please enter some text to summarize.")
        
        elif task_option == "GPT-Neo Text Generation":
            if user_input:
                response = infer_with_gpt_neo(user_input, gpt_neo_model, gpt_neo_tokenizer)
                st.subheader("Generated Response:")
                st.write(response)
            else:
                st.error("Please enter a prompt to generate a response.")

if __name__ == "__main__":
    main()
