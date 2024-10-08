from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, request, jsonify

app = Flask(__name__)

try:
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    print("Model loaded successfully.")
    model.to('cpu')

except Exception as e:
    print(f"Error loading model: {e}")

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

@app.route('/generate', methods=['POST'])
def generate_text():
    input_data = request.json
    prompt = input_data.get('prompt', '')

    if not prompt.strip():
        return jsonify({'error': 'Prompt must not be empty'}), 400

    inputs = tokenizer.encode_plus(prompt, return_tensors='pt', padding='longest', truncation=True)

    if inputs['input_ids'].size(1) == 0:
        return jsonify({'error': 'Tokenization resulted in empty input'}), 400

    try:
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=100,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    except Exception as e:
        return jsonify({'error': f'Error generating text: {e}'}), 500

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)
