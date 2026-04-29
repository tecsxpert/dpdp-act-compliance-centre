import os
import json
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify
from groq import Groq
from dotenv import load_dotenv
#load the api key from .env file
load_dotenv()

describe_bp = Blueprint('describe', __name__)

# Initialize the Groq client automatically using the .env file
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def load_prompt():
    """Helper function to load the prompt template from Day 2"""
    # Assuming app.py is run from the ai-service root folder
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'describe_prompt.txt')
    with open(prompt_path, 'r', encoding='utf-8') as file:
        return file.read()

@describe_bp.route('/describe', methods=['POST'])
def describe():
    # 1. Validate Input
    data = request.get_json()
    if not data or 'input_text' not in data:
        return jsonify({"error": "Bad Request: Missing 'input_text' in JSON body"}), 400
    
    raw_input = data['input_text']
    
    # 2. Load and Format Prompt
    try:
        prompt_template = load_prompt()
        # Inject the user's input into the {user_input} placeholder
        formatted_prompt = prompt_template.replace('{user_input}', raw_input)
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: Failed to load prompt. {str(e)}"}), 500

    # 3. Call Groq API
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # Using the 70b model specified in the tech stack
            messages=[
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.3, # 0.3 for factual/objective outputs as per guide
            response_format={"type": "json_object"} # Forces the LLM to output valid JSON
        )
        
        # Parse the string response into a Python dictionary
        ai_response = json.loads(response.choices[0].message.content)
        
        # 4. Append the generated_at timestamp as required
        ai_response['generated_at'] = datetime.now(timezone.utc).isoformat()
        
        # 5. Return the structured JSON
        return jsonify(ai_response), 200

    except Exception as e:
        return jsonify({"error": f"Groq API error: {str(e)}"}), 500