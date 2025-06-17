from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load fine-tuned model and tokenizer
model_path = "./wordpair-gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Set model to evaluation mode
model.eval()

print("🔤 Ojibwe-to-English Translator")
print("Type an Ojibwe word or phrase (or type 'exit' to quit):")

while True:
    user_input = input("Ojibwe: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Exiting translator.")
        break

    prompt = f"Ojibwe: {user_input} => English:"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = translated_text.replace(prompt, "").strip()
    print(f"English: {result}\n")
