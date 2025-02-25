import os

from openai import OpenAI

from ..base import VannaBase


class OpenAI_Chat(VannaBase):
    def __init__(self, client=None, config=None):
        VannaBase.__init__(self, config=config)

        # Ensure config is a dictionary
        config = config or {}

        # default parameters - can be overrided using config
        self.temperature = config.get("temperature", 0.7)
        self.model = config.get("model", "gpt-4o-mini")

        # Raise exceptions for deprecated parameters
        for deprecated_param in ["api_type", "api_base", "api_version"]:
            if deprecated_param in config:
                raise ValueError(
                    f"Passing {deprecated_param} is now deprecated. Please pass an OpenAI client instead."
                )

        if client is not None:
            self.client = client
            return

        if client is None:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            return

        if "api_key" in config:
            self.client = OpenAI(api_key=config["api_key"])

    def system_message(self, message: str) -> dict:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> dict:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> dict:
        return {"role": "assistant", "content": message}

    def generate_response(self, prompt, num_tokens):
        print(f"Using model {self.model} for {num_tokens} tokens (approx)")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            stop=None,
            temperature=self.temperature,
        )
        return response

    def submit_prompt(self, prompt, **kwargs) -> dict:
        if prompt is None:
            raise ValueError("Prompt is None")

        if len(prompt) == 0:
            raise ValueError("Prompt is empty")

        # Count the number of tokens in the message log
        # Use 4 as an approximation for the number of characters per token
        num_tokens = 0
        for message in prompt:
            num_tokens += len(message["content"]) / 4

        response = self.generate_response(prompt, num_tokens)

        # Find the first response from the chatbot that has text in it (some responses may not have text)
        for choice in response.choices:
            if "text" in choice:
                return choice.text

        # If no response with text is found, return the first response's content (which may be empty)
        return response.choices[0].message.content
