import streamlit as st
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPTNeoForCausalLM, GPT2Tokenizer, AutoModelForSeq2SeqLM
)
from peft import PeftModel
import pdfplumber

# Force CPU usage
DEVICE = "cpu"

# Load Models and Tokenizers
@st.cache_resource
def load_model(model_id, tokenizer_id, is_peft=False, quant_config=None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if quant_config:
        model = AutoModelForCausalLM.from_pretrained(model_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)

    if is_peft:
        model = PeftModel.from_pretrained(model, model_id)

    model.eval()
    return model, tokenizer

# Load specific models
@st.cache_resource
def load_gpt_neo():
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPTNeoForCausalLM.from_pretrained("AmmarA22/gptneo-wikitext-quantized").to(DEVICE)
    return model, tokenizer

@st.cache_resource
def load_flan_t5():
    fine_tuned_model = AutoModelForSeq2SeqLM.from_pretrained("ShahzaibDev/flant5-finetuned-summarizer").to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("ShahzaibDev/flant5-finetuned-summarizer")
    return fine_tuned_model, tokenizer

# Generate responses
def generate_response(prompt, model, tokenizer, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=0.7,
        top_k=50
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Summarization-specific response generation
def summarize_text(prompt, model, tokenizer):
    encoded_input = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=1024
    ).to(DEVICE)
    generated_ids = model.generate(
        **encoded_input,
        max_length=800,
        num_beams=6,
        no_repeat_ngram_size=3
    )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return " ".join([page.extract_text() for page in pdf.pages])

# Streamlit UI
st.title("Multi-Model Frontend")
st.markdown("Choose a model, enter your input, and get the response!")

# Dropdown for model selection
models = [
    "Llama-2 QnA",
    "Medical QA with BioMistral",
    "GPT-Neo Text Generator",
    "FLAN-T5 Summarizer"
]

model_choice = st.selectbox("Select a Model", models)

if model_choice == "Llama-2 QnA":
    model, tokenizer = load_model(
        "NousResearch/Llama-2-7b-chat-hf",
        "NousResearch/Llama-2-7b-chat-hf",
        is_peft=True
    )
    prompt = st.text_area("Enter your question:", "", height=150)
    if st.button("Generate Response"):
        if prompt.strip():
            response = generate_response(f"<s>[INST] {prompt.strip()} [/INST]", model, tokenizer)
            st.write(response)
        else:
            st.error("Please enter a valid question.")

elif model_choice == "Medical QA with BioMistral":
    model, tokenizer = load_model(
        "ShahzaibDev/Biomistral_Model_weight_files",
        "ShahzaibDev/Biomistral_Model_weight_files",
        is_peft=True
    )
    question = st.text_area("Enter your question:", "", height=150)
    question_type = st.text_input("Enter question type (optional):")
    if st.button("Get Answer"):
        if question.strip():
            eval_prompt = f"### Question type: {question_type}\n### Question: {question}\n### Answer:" if question_type else question
            response = generate_response(eval_prompt, model, tokenizer)
            st.write(response)
        else:
            st.error("Please enter a valid question.")

elif model_choice == "GPT-Neo Text Generator":
    model, tokenizer = load_gpt_neo()
    prompt = st.text_area("Enter your prompt:", "", height=150)
    max_length = st.slider("Maximum Length of Response", 50, 500, 200, 10)
    if st.button("Generate Response"):
        if prompt.strip():
            response = generate_response(prompt, model, tokenizer, max_length)
            st.write(response)
        else:
            st.error("Please enter a valid prompt.")

elif model_choice == "FLAN-T5 Summarizer":
    model, tokenizer = load_flan_t5()
    input_type = st.selectbox("Input Type", ["Enter Text", "Upload PDF"])
    if input_type == "Enter Text":
        input_text = st.text_area("Enter text to summarize:", "", height=200)
        if st.button("Summarize"):
            if input_text.strip():
                prompt = f"Summarize the following:\n{input_text}\nSummary:"
                summary = summarize_text(prompt, model, tokenizer)
                st.write(summary)
            else:
                st.error("Please enter valid text to summarize.")
    elif input_type == "Upload PDF":
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
        if uploaded_file:
            text = extract_text_from_pdf(uploaded_file)
            prompt = f"Summarize the following:\n{text}\nSummary:"
            summary = summarize_text(prompt, model, tokenizer)
            st.write(summary)
        else:
            st.error("Please upload a PDF file.")
