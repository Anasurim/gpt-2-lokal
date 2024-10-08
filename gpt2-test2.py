from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

prompt = "Once upon a time"
inputs = tokenizer.encode(prompt, return_tensors='pt')

attention_mask = torch.ones(inputs.shape, dtype=torch.long)

outputs = model.generate(
    inputs,
    attention_mask=attention_mask,
    max_length=50,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
