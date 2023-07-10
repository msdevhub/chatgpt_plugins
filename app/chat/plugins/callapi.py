from .plugin import PluginInterface
from typing import Dict
import requests
from bs4 import BeautifulSoup



class CallAPIPlugin(PluginInterface):
    def get_name(self) -> str:
        """
        return the name of the plugin (should be snake case)
        """
        return "call_rest_api"
    
    def get_description(self) -> str:
        return "Sends a request to the REST API"
    

    def get_parameters(self) -> Dict:
        """
        Return the list of parameters to execute this plugin in the form of
        JSON schema as specified in the OpenAI documentation:
        https://platform.openai.com/docs/api-reference/chat/create#chat/create-parameters
        """
        parameters = {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "The HTTP method to be used",
                    "enum": ["GET", "POST", "PUT", "DELETE"],
                },
                "url": {
                    "type": "string",
                    "description": "The URL of the endpoint. Value placeholders must be replaced with actual values.",
                },
                "body": {
                    "type": "string",
                    "description": "A string representation of the JSON that should be sent as the request body.",
                },
            },
            "required": ["method", "url"]
        }
        return parameters
    
    def execute(self, **kwargs) -> Dict:
        """
        Execute the plugin and return a JSON response.
        The parameters are passed in the form of kwargs
        """

        # Send a GET request to the URL
        response = requests.get(kwargs['url'])

        # Create a BeautifulSoup object to parse the HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the text content from the parsed HTML
        text_content = soup.get_text()
        return {"content": text_content}

        
