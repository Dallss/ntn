import requests
import json

class OllamaHandler:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"

    def ask_model(self, model, prompt, stream=False):
        """
        Sends a prompt to the Ollama server.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }

        try:
            response = requests.post(self.generate_url, json=payload)
            response.raise_for_status()
            
            # Ollama returns a JSON object for the full response if stream is False
            return response.json().get("response", "No response received.")
        
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {str(e)}"