import streamlit as st
import torch
import os
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSeq2SeqLM,
    GPTNeoForCausalLM, GPT2Tokenizer
)
from peft import PeftModel
import pdfplumber

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dropdown menu options
MODEL_OPTIONS = [
    "Llama-2 QnA",
    "Medical Question Answering",
    "Fine-Tuned GPT-Neo",
    "Text Summarizer with FLAN-T5"
]

st.title("Multi-Model Application")
st.sidebar.header("Select a Model")
selected_model = st.sidebar.selectbox("Model", MODEL_OPTIONS)

# Cache resource for model loading
@st.cache_resource
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

@st.cache_resource
def load_medqa_model():
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

@st.cache_resource
def load_gptneo_model():
    base_model_name = "EleutherAI/gpt-neo-125M"
    pretrained_model_name = "AmmarA22/gptneo-wikitext-quantized"

    tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPTNeoForCausalLM.from_pretrained(pretrained_model_name).to(device)
    return model, tokenizer

@st.cache_resource
def load_flan_t5_model():
    fine_tuned_output_dir = "ShahzaibDev/flant5-finetuned-summarizer"
    tokenizer = AutoTokenizer.from_pretrained(fine_tuned_output_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_output_dir).to(device)
    return model, tokenizer

# UI Components for each model
def llama_2_qna_ui():
    st.header("Llama-2 QnA")
    model, tokenizer = load_llama_model()

    question = st.text_area("Enter your question:", placeholder="What is generative AI?", height=150)
    if st.button("Generate Response"):
        if question.strip():
            formatted_prompt = f"<s>[INST] {question.strip()} [/INST]"
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
            with st.spinner("Generating response..."):
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
            st.subheader("Response:")
            st.write(response)
        else:
            st.error("Please enter a valid question.")

def medical_qa_ui():
    st.header("Medical Question Answering")
    model, tokenizer = load_medqa_model()

    question = st.text_area("Enter your question:")
    question_type = st.text_input("Enter question type (optional):")

    if st.button("Get Answer"):
        if question.strip():
            prompt = f"From the MedQuad MedicalQA Dataset:\n### Question type:\n{question_type}\n### Question:\n{question}\n### Answer:" if question_type else question
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with st.spinner("Generating response..."):
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=300)
                answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.subheader("Response:")
            st.write(answer)
        else:
            st.error("Please enter a question.")

def gptneo_ui():
    st.header("Fine-Tuned GPT-Neo")
    model, tokenizer = load_gptneo_model()

    prompt = st.text_area("Enter your prompt:", placeholder="Type something to generate a response...")
    max_length = st.slider("Maximum Length of Response", min_value=50, max_value=500, value=200, step=10)

    if st.button("Generate Response"):
        if prompt.strip():
            with st.spinner("Generating response..."):
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_length=max_length,
                        num_return_sequences=1,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True
                    )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.subheader("Generated Response:")
            st.write(response)
        else:
            st.error("Please enter a prompt.")

def flan_t5_ui():
    st.header("Text Summarizer with FLAN-T5")
    model, tokenizer = load_flan_t5_model()

    option = st.selectbox("Choose Input Type", ["Enter Text", "Upload PDF"])

    def extract_text_from_pdf(pdf_file):
        with pdfplumber.open(pdf_file) as pdf:
            return "".join(page.extract_text() for page in pdf.pages)

    if option == "Enter Text":
        input_text = st.text_area("Enter the text you want to summarize:", height=200)
        if st.button("Summarize"):
            if input_text.strip():
                prompt = f"Summarize the following document:\n{input_text}\n\n Summary:"
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with st.spinner("Generating summary..."):
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs, max_length=800, num_beams=6, early_stopping=True
                        )
                    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                st.subheader("Generated Summary:")
                st.write(summary)
            else:
                st.error("Please enter some text to summarize.")

    elif option == "Upload PDF":
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            text = extract_text_from_pdf(uploaded_file)
            if text.strip():
                prompt = f"Summarize the following document:\n{text[:800]}\n\n Summary:"
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with st.spinner("Generating summary..."):
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs, max_length=800, num_beams=6, early_stopping=True
                        )
                    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                st.subheader("Generated Summary:")
                st.write(summary)
            else:
                st.error("The uploaded PDF contains no readable text.")

# Render UI based on
