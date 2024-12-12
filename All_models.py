import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSeq2SeqLM, GPTNeoForCausalLM, GPT2Tokenizer
from peft import PeftModel
import pdfplumber
import os

# Define the device for inference (GPU or CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dropdown menu options
models = [
    "Llama-2 QnA",
    "Medical Question Answering",
    "Fine-Tuned GPT-Neo Text Generator",
    "Text Summarizer with FLAN-T5"
]

# Streamlit page configuration
st.set_page_config(page_title="Multi-Model Interface", layout="wide")

# Helper functions for each model

def load_llama_model():
    base_model_id = "NousResearch/Llama-2-7b-chat-hf"
    peft_model_id = "ShahzaibDev/Llama2-7B-Qna"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        quantization_config=quantization_config
    )
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    model.eval()
    return model, tokenizer

def load_medical_model():
    base_model_id = "ShahzaibDev/Biomistral_Model_weight_files"
    peft_model_id = "ShahzaibDev/biomistral-medqa-finetune"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        quantization_config=bnb_config
    )
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    model.eval()
    return model, tokenizer

def load_gpt_neo_model():
    base_model_name = "EleutherAI/gpt-neo-125M"
    pretrained_model_name = "AmmarA22/gptneo-wikitext-quantized"

    tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPTNeoForCausalLM.from_pretrained(pretrained_model_name).to(DEVICE)
    return model, tokenizer

def load_flan_t5_model():
    fine_tuned_output_dir = "ShahzaibDev/flant5-finetuned-summarizer"
    tokenizer = AutoTokenizer.from_pretrained(fine_tuned_output_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_output_dir).to(DEVICE)
    return model, tokenizer

def extract_text_from_pdf(pdf_file_path):
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"The file at {pdf_file_path} does not exist.")
    
    with pdfplumber.open(pdf_file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def trim_text(text, max_words=1400):
    words = text.split()
    return " ".join(words[:max_words])

def infer_with_finetuned_gpt_neo(prompt, model, tokenizer, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_prompt(doc):
    return f"Summarize the following document:\n\n{doc}\n\n Summary:"

def get_response(prompt, model, tokenizer):
    encoded_input = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=1024
    )
    model_inputs = encoded_input.to(DEVICE)
    generated_ids = model.generate(
        **model_inputs,
        max_length=800,
        num_beams=6,
        early_stopping=True,
        no_repeat_ngram_size=3,
        temperature=0.7,
        top_k=50
    )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# UI Components
st.title("Multi-Model Interface")
st.markdown("Select a model from the dropdown menu and interact with it.")

model_choice = st.selectbox("Choose a Model", models)

if model_choice == "Llama-2 QnA":
    model, tokenizer = load_llama_model()
    st.subheader("Llama-2 QnA")
    prompt = st.text_area("Enter your question:", height=150)
    if st.button("Generate Response"):
        if prompt.strip():
            formatted_prompt = f"<s>[INST] {prompt.strip()} [/INST]"
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.write(response)
        else:
            st.error("Please enter a valid question.")

elif model_choice == "Medical Question Answering":
    model, tokenizer = load_medical_model()
    st.subheader("Medical Question Answering")
    question = st.text_area("Enter your question:")
    question_type = st.text_input("Enter question type (optional):")
    if st.button("Get Answer"):
        if question.strip():
            eval_prompt = f"From the MedQuad MedicalQA Dataset: Given the following medical question and question type, provide an accurate answer:\n\n### Question type:\n{question_type}\n\n### Question:\n{question}\n\n### Answer:" if question_type else question
            inputs = tokenizer(eval_prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=300)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.write(answer)
        else:
            st.error("Please enter a question.")

elif model_choice == "Fine-Tuned GPT-Neo Text Generator":
    model, tokenizer = load_gpt_neo_model()
    st.subheader("Fine-Tuned GPT-Neo Text Generator")
    prompt = st.text_area("Enter your prompt:", height=150)
    max_length = st.slider("Maximum Length of Response", min_value=50, max_value=500, value=200, step=10)
    if st.button("Generate Response"):
        if prompt.strip():
            response = infer_with_finetuned_gpt_neo(prompt, model, tokenizer, max_length)
            st.write(response)
        else:
            st.error("Please enter a prompt to generate a response.")

elif model_choice == "Text Summarizer with FLAN-T5":
    model, tokenizer = load_flan_t5_model()
    st.subheader("Text Summarizer with FLAN-T5")
    option = st.selectbox("Choose Input Type", ["Enter Text", "Upload PDF"])

    if option == "Enter Text":
        input_text = st.text_area("Enter the text you want to summarize:", height=200)
        if st.button("Summarize"):
            if input_text:
                prompt = get_prompt(input_text)
                summary = get_response(prompt, model, tokenizer)
                st.write(summary)
            else:
                st.error("Please enter some text to summarize.")

    elif option == "Upload PDF":
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded_file is not None:
            with st.spinner("Extracting text from the uploaded PDF..."):
                text = extract_text_from_pdf(uploaded_file.name)
                text = trim_text(text)
                prompt = get_prompt(text)
                summary = get_response(prompt, model, tokenizer)
                st.write(summary)
