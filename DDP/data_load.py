import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from torch.utils.data import random_split

print("Loading dataset...")
dataset = load_dataset("roneneldan/TinyStories", split="train[:10%]")

print("Setting up tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", max_length=256, truncation=True)

print("Tokenizing dataset...")
val = dataset.map(tokenize_function, batched=True)
val.set_format("torch", columns=["input_ids", "attention_mask"])

print("Splitting data...")
total_len = len(val)
train_len = int(0.8 * total_len)
val_len = int(0.1 * total_len)
test_len = total_len - (train_len + val_len)

train_data, val_data, test_data = random_split(val, [train_len, val_len, test_len])

print(f"Train size: {len(train_data)}")
print(f"Val size: {len(val_data)}")
print(f"Test size: {len(test_data)}")

# IMPORTANT: Save the preprocessed data!
print("\nSaving preprocessed data...")
torch.save({
    'train_data': train_data,
    'val_data': val_data,
    'test_data': test_data,
    'tokenizer': tokenizer
}, 'preprocessed_data.pt')

print(" Data saved to 'preprocessed_data.pt'")
print("\nYou can now use this file for training without re-downloading!")
