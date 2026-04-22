import os
from groq import Groq
from dotenv import load_dotenv
#load the api key from .env file
load_dotenv()
client=Groq(api_key=os.getenv("GROQ_API_KEY"))   #set up the gorq client using the api key stored .env
#send the user message to the groq and get a response back
def queryapi(prompt):
    """Query the Groq API with a prompt"""
    try:
        response=client.chat.completions.create(
            model="llama-3.1-8b-instant",   #this used for groq model 
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        #return the text part of the response
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"  #if something goes wrong print the error instead of crashing
if __name__ == "__main__":
    userprompt=input("Enter your prompt: ")
    print("Querying Groq...")
    result = queryapi(userprompt)
    print("Response:")
    print(result)