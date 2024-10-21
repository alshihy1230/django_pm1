from flask import Flask, request, jsonify
from transformers import GPTNeoForCausalLM, AutoTokenizer

app = Flask(__name__)

# تحميل النموذج والمحول
model_name = "EleutherAI/gpt-neo-1.3B"
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route("/generate", methods=["get"])
def generate():
    
    input_text = "hello"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=100, do_sample=True, temperature=0.9)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"generated_text": generated_text})
