from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model_path = "/Users/user/Desktop/Ipamm/hajj_qa_model"  # Replace with your actual path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

# Initialize QA pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

@app.route('/answer', methods=['POST'])
def get_answer():
    data = request.json
    question = data.get("question", "")
    context =  "There are professional laundry services, but not that many coin-operated launderettes. Detergent to wash your light clothing is also available in the local market nearby your hotels."
    
    if not question or not context:
        return jsonify({"error": "Both question and context are required"}), 400

    # Get the answer using the QA pipeline
    result = qa_pipeline(question=question, context=context)
    return jsonify({
        # "question": question,
        # "context": context,
        "answer": result['answer'],
        "score": result['score'],
        # "start_index": result['start'],
        # "end_index": result['end']
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8000")
