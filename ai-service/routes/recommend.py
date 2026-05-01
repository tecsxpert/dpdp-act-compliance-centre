import os
import json
from flask import Blueprint, request, jsonify
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

recommend_bp = Blueprint('recommend', __name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def load_prompt():
    """Helper function to load the prompt template from Day 4"""
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'recommend_prompt.txt')
    with open(prompt_path, 'r', encoding='utf-8') as file:
        return file.read()

@recommend_bp.route('/recommend', methods=['POST'])
def recommend():
    # 1. Validate Input
    data = request.get_json()
    if not data or 'input_text' not in data:
        return jsonify({"error": "Bad Request: Missing 'input_text' in JSON body"}), 400
    
    raw_input = data['input_text']
    
    # 2. Load and Format Prompt
    try:
        prompt_template = load_prompt()
        formatted_prompt = prompt_template.replace('{user_input}', raw_input)
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: Failed to load prompt. {str(e)}"}), 500

    # 3. Call Groq API
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.3, # Factual/objective tone
            response_format={"type": "json_object"} # Force JSON output
        )
        
        # Parse and return the JSON response
        ai_response = json.loads(response.choices[0].message.content)
        return jsonify(ai_response), 200

    except Exception as e:
        return jsonify({"error": f"Groq API error: {str(e)}"}), 500