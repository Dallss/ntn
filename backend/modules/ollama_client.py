import requests
import json

_ollama_base_url = "http://localhost:11434"
_ollama_model_file = "perfume-extractor:beta"

class OllamaHandler:
    def __init__(self, model_file=_ollama_model_file, base_url=_ollama_base_url):
        self.model_file = model_file
        self.generate_url = f"{base_url}/api/generate"

    def ask_model(self, user_input):
        """
        Sends user input to Ollama via /api/generate and returns the parsed JSON inside 'response'.
        """
        payload = {
            "model": self.model_file,
            "prompt": user_input,
            "temperature": 0,
            "max_tokens": 512,
            "stream": False
        }

        try:
            r = requests.post(self.generate_url, json=payload)
            r.raise_for_status()
            raw_json = r.json()
            
            # 'response' is a JSON string, so parse it
            if "response" in raw_json:
                parsed_response = json.loads(raw_json["response"])
                return parsed_response
            return raw_json
        except Exception as e:
            return {"error": f"Connection error: {str(e)}"}
