from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os


app = Flask(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_NAME = "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Get max context length and the determine cushion for response
MAX_CONTEXT_LENGTH = model.config.max_position_embeddings
print(f"Max context length: {MAX_CONTEXT_LENGTH}")
ROOM_FOR_RESPONSE = 512

model = model.half().cuda()


@app.route('/generate', methods=['POST'])
def generate():
    content = request.json
    inp = content.get("text", "")
    input_ids = tokenizer.encode(inp, return_tensors="pt")

    # Calc current size
    print("Context length is currently", input_ids.shape[1], "tokens. Allowed amount is", MAX_CONTEXT_LENGTH-ROOM_FOR_RESPONSE, "tokens.")
    # determine if we need to trim
    if input_ids.shape[1] > (MAX_CONTEXT_LENGTH-ROOM_FOR_RESPONSE):
        print("Trimming a bit")
        # trim as needed AT the first dimension
        input_ids = input_ids[:, -(MAX_CONTEXT_LENGTH-ROOM_FOR_RESPONSE):]
    
    input_ids = input_ids.cuda()

    with torch.cuda.amp.autocast():
        output = model.generate(input_ids, max_length=2048, do_sample=True, early_stopping=True, num_return_sequences=1, eos_token_id=model.config.eos_token_id)

    decoded = tokenizer.decode(output[0], skip_special_tokens=False)

    return jsonify({'generated_text': decoded})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Set the host to '0.0.0.0' to make it accessible from your local network