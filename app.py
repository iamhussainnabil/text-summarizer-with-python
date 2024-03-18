from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Load the saved tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('tokenizer')
model = AutoModelForSeq2SeqLM.from_pretrained('model')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        # Get the input text from the form
        input_text = request.form['text']

        # Tokenize the input text
        inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)

        # Generate the summary
        summary_ids = model.generate(inputs.input_ids, num_beams=4, max_length=150, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
