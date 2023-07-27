import logging
import openai
import requests
import json
from typing import List, Dict
import uuid

from .plugins.callapi import CallAPIPlugin
from .plugins.plugin import PluginInterface
from .plugins.websearch import WebSearchPlugin
from .plugins.webscraper import WebScraperPlugin
from .plugins.pythoninterpreter import PythonInterpreterPlugin
import os

logging.basicConfig(
    level=logging.INFO,
    filename="gpt.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")

GPT_MODEL = "gpt-35-turbo-16k"
SYSTEM_PROMPT = """
    You are a helpful AI assistant name is GPT BOY. You answer the user's queries.
    When you are not sure of an answer, you take the help of
    functions provided to you.
    NEVER make up an answer if you don't know, just respond
    with "I don't know" when you don't know.
"""
# get env from env file
openai.api_type = os.getenv("OPEN_AI_TYPE")
openai.api_base = os.getenv("OPEN_AI_ENDPOINT")
openai.api_version = "2023-07-01-preview"
openai.api_key = OPEN_AI_KEY


class Conversation:
    """
    This class represents a conversation with the ChatGPT model.
    It stores the conversation history in the form of a list of messages.
    """

    available_apis = [
        # {
        #     "method": "GET",
        #     "url": "/users?page=[page_id]",
        #     "description": "Lists employees. The response is paginated. You may need to request more than one to get them all. For example,/users?page=2.",
        # },
        # {
        #     "method": "GET",
        #     "url": "/users/[user_id]",
        #     "description": "Returns information about the employee identified by the given id. For example,/users/2",
        # },
        # {
        #     "method": "POST",
        #     "url": "/users",
        #     "description": "Creates a new employee profile. This function accepts JSON body containing two fields: name and job",
        # },
        # {
        #     "method": "PUT",
        #     "url": "/users/[user_id]",
        #     "description": "Updates employee information. This function accepts JSON body containing two fields: name and job. The user_id in the URL must be a valid identifier of an existing employee.",
        # },
        # {
        #     "method": "DELETE",
        #     "url": "/users/[user_id]",
        #     "description": "Removes the employee identified by the given id. Before you call this function, find the employee information and make sure the id is correct. Do NOT call this function if you didn't retrieve user info. Iterate over all pages until you find it or make sure it doesn't exist",
        # },
        # =====================#
        {
            "method": "GET",
            "url": "/industry?page=[page_id]",
            "description": "Lists industries. The response is paginated. You may need to request more than one to get them all. For example,/industry?page=2.",
        },
        {
            "method": "GET",
            "url": "/industry/[industry_id]",
            "description": "Returns information about the industry identified by the given id. For example,/industry/2",
        },
        {
            "method": "POST",
            "url": "/industry",
            "description": "Creates a new industry. This function accepts JSON body containing four fields: id,name,created,modified",
        },
        {
            "method": "PUT",
            "url": "/industry/[industry_id]",
            "description": "Updates industry information. This function accepts JSON body containing one fields: name. The industry_id in the URL must be a valid identifier of an existing industry.",
        },
        {
            "method": "DELETE",
            "url": "/industry/[industry_id]",
            "description": "Removes the industry identified by the given id. Before you call this function, find the industry information and make sure the id is correct. Do NOT call this function if you didn't retrieve user info. Iterate over all pages until you find it or make sure it doesn't exist",
        },
    ]

    def __init__(self):
        self.conversation_history: List[Dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "You have access to the following APIs: "
                + json.dumps(self.available_apis),
            },
            {
                "role": "user",
                "content": "If a function requires an identifier, list all first to find the proper value. You may need to list more than one page",
            },
            {
                "role": "user",
                "content": "If you were asked to create, update, or delete a user, perform the action and reply with a confirmation telling what you have done.",
            },
        ]

    def add_message(self, role, content):
        self.conversation_history.append({"role": role, "content": content})


class ChatSession:
    """
    Represents a chat session.
    Each session has a unique id to associate it with the user.
    It holds the conversation history
    and provides functionality to get new response from ChatGPT
    for user query.
    """

    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.conversation = Conversation()
        self.plugins: Dict[str, PluginInterface] = {}
        self.register_plugin(CallAPIPlugin())
        # self.register_plugin(WebSearchPlugin())
        # self.register_plugin(WebScraperPlugin())
        # self.register_plugin(PythonInterpreterPlugin())
        # self.conversation.add_message("system", SYSTEM_PROMPT)

    def register_plugin(self, plugin: PluginInterface):
        """
        Register a plugin for use in this session
        """
        self.plugins[plugin.get_name()] = plugin

    def get_messages(self) -> List[Dict]:
        """
        Return the list of messages from the current conversaion
        """
        if len(self.conversation.conversation_history) == 1:
            return []
        return self.conversation.conversation_history[4:]  # skip message

    def _get_functions(self) -> List[Dict]:
        """
        Generate the list of functions that can be passed to the chatgpt
        API call.
        """
        return [self._plugin_to_function(p) for p in self.plugins.values()]

    def _plugin_to_function(self, plugin: PluginInterface) -> Dict:
        """
        Convert a plugin to the function call specification as
        required by the ChatGPT API:
        https://platform.openai.com/docs-reference/chat/create#chat/create-functions
        """
        function = {}
        function["name"] = plugin.get_name()
        function["description"] = plugin.get_description()
        function["parameters"] = plugin.get_parameters()
        return function

    def _execute_plugin(self, func_call) -> str:
        """
        If a plugin exists for the given function call, execute it.
        """
        func_name = func_call.get("name")

        logging.info(f"Executing plugin {func_name}")
        if func_name in self.plugins:
            arguments = json.loads(func_call.get("arguments"))
            plugin = self.plugins[func_name]
            plugin_response = plugin.execute(**arguments)
        else:
            plugin_response = {"error": f"No plugin found with name {func_call}"}

        # We need to pass the plugin response back to ChatGPT
        # so that it can process it. In order to do this we
        # need to append the plugin response into the conversation
        # history. However, this is just temporary so we make a
        # copy of the messages and then append to that copy.
        logging.info(f"Response from plugin {func_name}: {plugin_response}")
        messages = list(self.conversation.conversation_history)
        messages.append(
            {
                "role": "function",
                "content": json.dumps(plugin_response),
                "name": func_name,
            }
        )
        next_chatgpt_response = self._chat_completion_request(messages)

        # If ChatGPT is asking for another function call, then
        # we need to call _execute_plugin again. We will keep
        # doing this until ChatGPT keeps returning function_call
        # in its response. Although it might be a good idea to
        # cut it off at some point to avoid an infinite loop where
        # it gets stuck in a plugin loop.
        if next_chatgpt_response.get("function_call"):
            return self._execute_plugin(next_chatgpt_response.get("function_call"))
        return next_chatgpt_response.get("content")

    def get_chatgpt_response(self, user_message: str) -> str:
        """
        For the given user_message,
        get the response from ChatGPT
        """
        logging.info(f"Begin get GPT response: {user_message}")
        self.conversation.add_message("user", user_message)
        try:
            chatgpt_response = self._chat_completion_request(
                self.conversation.conversation_history
            )

            if chatgpt_response.get("function_call"):
                chatgpt_message = self._execute_plugin(
                    chatgpt_response.get("function_call")
                )
            else:
                chatgpt_message = chatgpt_response.get("content")
            self.conversation.add_message("assistant", chatgpt_message)
            return chatgpt_message
        except Exception as e:
            logging.error(e)
            return "something went wrong"

    def _chat_completion_request(self, messages: List[Dict]):
        # headers = {
        #     "Content-Type": "application/json",
        #     "Authorization": "Bearer " + "a67a5c5c489d4da2a558525722e7c36e",
        # }
        # json_data = {"model": GPT_MODEL, "messages": messages, "temperature": 0.7}
        # if self.plugins:
        #     json_data.update({"functions": self._get_functions()})
        try:
            # get response from openai with stream output

            response = openai.ChatCompletion.create(
                engine="gpt-35-turbo-16k",
                messages=messages,
                # stream=True,
                temperature=0,
                functions=self._get_functions(),
                function_call="auto",
            )

            logging.info(f"ChatGPT response: {json.dumps(response['choices'])}")
            return response["choices"][0]["message"]

            # response = requests.post(
            #     "https://resley-openai.openai.azure.com/openai/deployments/gpt-35-turbo-16k/chat/completions?api-version=2023-03-15-preview",
            #     headers=headers,
            #     json=json_data,
            # )
            # return response.json()["choices"][0]["message"]
        except Exception as e:
            logging.error("Unable to generate ChatCompletion response")
            logging.error(f"Exception: {e}")
            return e
