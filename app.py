import os
import requests
import logging
import json
from flask import Flask, request, jsonify
from transformers import RobertaForQuestionAnswering, RobertaTokenizerFast, pipeline
from time import sleep

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# GitHub repository URL for the model files
model_repo_url = "https://github.com/amsy572/ipamam/raw/main/qa_model"
model_dir = "/tmp/qa_model"  # Path to save downloaded model files

# Ensure the model directory exists
os.makedirs(model_dir, exist_ok=True)

# List of expected model files and their raw URLs
model_files = [
    ("config.json", f"{model_repo_url}/config.json"),
    ("merges.txt", f"{model_repo_url}/merges.txt"),
    ("model.safetensors", f"{model_repo_url}/model.safetensors"),
    ("tokenizer_config.json", f"{model_repo_url}/tokenizer_config.json"),
    ("special_tokens_map.json", f"{model_repo_url}/special_tokens_map.json"),
    ("training_args.bin", f"{model_repo_url}/training_args.bin"),
    ("vocab.json", f"{model_repo_url}/vocab.json"),
    ("context.json", f"{model_repo_url}/context.json")
]

# Download the model files with retry logic
for file_name, file_url in model_files:
    success = False
    for attempt in range(3):  # Retry up to 3 times
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(os.path.join(model_dir, file_name), 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded {file_name} successfully.")
            success = True
            break
        else:
            logger.error(f"Failed to download {file_name} from {file_url}. Status code: {response.status_code}")
            logger.debug(response.text)
            sleep(2)  # Wait before retrying
    
    if not success:
        logger.error(f"Failed to download {file_name} after 3 attempts.")
        exit(1)  # Exit if any model file fails to download

# Load the tokenizer and model with error handling
try:
    tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
    model = RobertaForQuestionAnswering.from_pretrained(model_dir)
    logger.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logger.error("Error loading model or tokenizer:", exc_info=True)
    exit(1)  # Exit the application if loading fails

# Initialize QA pipeline
try:
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
except Exception as e:
    logger.error("Error initializing QA pipeline:", exc_info=True)
    exit(1)

# Load all contexts from the JSON file
context_file_path = os.path.join(model_dir, "context.json")
try:
    with open(context_file_path, 'r', encoding='utf-8') as f:
        contexts = json.load(f)  # Load all contexts as a list
    if not contexts or not all("question" in entry and "context" in entry for entry in contexts):
        logger.error("Invalid context data format or empty contexts.")
        raise ValueError("Invalid context data format or empty contexts.")
    logger.info("Loaded contexts successfully.")
except (FileNotFoundError, ValueError) as e:
    logger.error(f"Error loading context file {context_file_path}: {e}")
    contexts = []

def get_answer(question, dataset):
    for entry in dataset:
        if "question" not in entry or "context" not in entry:
            logger.error("Entry missing required keys: %s", entry)
            continue  # Skip entries with missing keys
        
        if question.lower() in entry["question"].lower():  # Simple matching based on question similarity
            context = entry["context"]
            break
    else:
        return "No matching context found in the dataset."

    # Tokenize the input
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=384)
    
    # Get model predictions
    outputs = model(**inputs)
    start_idx = outputs.start_logits.argmax()
    end_idx = outputs.end_logits.argmax()
    
    # Decode the answer
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_idx:end_idx + 1])
    )
    return answer

@app.route('/answer', methods=['POST'])
def handle_request():
    try:
        data = request.json
        question = data.get("question", "")
        context_indices = data.get("context_indices", [])  # List of indices to use

        # Validate input
        if not question:
            logger.warning("Received request with no question.")
            return jsonify({"error": "Question is required"}), 400  # Return 400 Bad Request

        # Select contexts based on provided indices
        if context_indices:
            try:
                selected_contexts = " ".join([contexts[i]['context'] for i in context_indices])
            except IndexError:
                logger.error("Invalid context index provided.")
                return jsonify({"error": "Invalid context index"}), 400  # Return 400 Bad Request
        else:
            selected_contexts = " ".join([ctx['context'] for ctx in contexts])  # Combine all contexts

        # Get the answer using the QA function
        result = get_answer(question, contexts)
        logger.info("Answer generation successful.")
        return jsonify({
            "question": question,
            "context": selected_contexts,
            "answer": result,
        })
    except Exception as e:
        logger.error("Error processing request:", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host="0.0.0.0", port=8000, debug=False)
