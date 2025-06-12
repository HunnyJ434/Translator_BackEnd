import pandas as pd
import subprocess
import torch
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Download data using your original wget command
subprocess.run([
    "wget", "-O", "data.txt",
    "https://raw.githubusercontent.com/HunnyJ434/Fibonacci-using-Dynamic-Programming/refs/heads/main/output1.txt"
])

# Read and preprocess data
data = []
with open("data.txt", "r") as file:
    for line in file:
        if ":" in line:
            input_text, output_text = line.strip().split(":", 1)
            data.append({"Input": input_text.strip(), "Output": output_text.strip()})

# Convert to DataFrame and Hugging Face dataset
df = pd.DataFrame(data)
df["text"] = df["Input"] + " => " + df["Output"]

dataset = Dataset.from_pandas(df[["text"]])
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenize and mask pad tokens
def tokenize(example):
    encodings = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )
    encodings["labels"] = [
        -100 if token == tokenizer.pad_token_id else token
        for token in encodings["input_ids"]
    ]
    return encodings

train_dataset = train_dataset.map(tokenize)
eval_dataset = eval_dataset.map(tokenize)

# Training arguments
training_args = TrainingArguments(
    output_dir="./wordpair-gpt2",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    report_to="none",
    save_total_limit=1
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train
trainer.train()

# Save model/tokenizer
model.save_pretrained("./wordpair-gpt2")
tokenizer.save_pretrained("./wordpair-gpt2")

print("✅ Training complete. Model saved to './wordpair-gpt2'.")
