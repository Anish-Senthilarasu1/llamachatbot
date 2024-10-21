from fastapi import FastAPI
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline 
from huggingface_hub import login 

app = FastAPI()

@app.get('/')
async def root():
    return {"Senthy Chatbot Home."} 

my_secret_key = #i removed my key 
model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'

login(token=my_secret_key)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=my_secret_key)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map='auto')

text_generator = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
)

def get_response(prompt):
    response = text_generator(prompt)
    gen_text = response[0]['generated_text']
    return gen_text

    

@app.get('/ai/{prompt}')
async def ai_model(prompt):
    llama_response = get_response(prompt)
    return{llama_response}
