import getpass
from huggingface_hub import login
import os

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print(torch.cuda.is_available())
token = "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"
login(token=token)

# 1: testing
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", use_auth_token=token)
inputs = tokenizer("Student: Hello, how are you? Helpful AI chatbot: <chat> fine thank you. </chat> Student: ", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))

# alpaca dataset. we merge the provided one with other datasets found online
alpaca = pd.read_parquet("hf://datasets/tatsu-lab/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")
df = pd.read_json("dev_data.json")
print(df.head(5))