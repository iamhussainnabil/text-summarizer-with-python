from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Define the model name or path
model_name = "facebook/bart-large-cnn"

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Save the tokenizer and model to a directory on your local machine
tokenizer.save_pretrained('tokenizer')
model.save_pretrained('model')

print("Model and tokenizer saved successfully to:")
