from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, request, jsonify

# Lade das GPT-2-Modell und den Tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 100)

    # Tokenisiere den Eingabe-Text
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generiere den Text
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({'response': text})

if __name__ == '__main__':
    # Binde den Server an alle verfügbaren IP-Adressen (0.0.0.0), damit er von außen zugänglich ist
    app.run(host='0.0.0.0', port=4000)
