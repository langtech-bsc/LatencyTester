from runner.methods.methods_manager import MethodManager, BaseTest
from openai import OpenAI



@MethodManager.register_method("OpenAIChat")
class OpenAIChat(BaseTest):
    def __init__(self, api_url, api_key="none", model="tgi"):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(base_url=self.api_url, api_key=self.api_key)

    
    def _complation(self, stream=False):
        messages = [
            {"role": "user", "content": "Hey, how's your day going?"},
            {"role": "assistant", "content": "I'm just a language model, so I don't have days, but I'm here and ready to help! How about you?"},
            {"role": "user", "content": "I'm doing well! Quick question: What's the capital of Japan?"},
            {"role": "assistant", "content": "The capital of Japan is Tokyo."},
            {"role": "user", "content": "Thanks! Can you summarize the theory of relativity and its implications in simple terms?"}
        ]
        
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            max_tokens=1000,
            temperature=0.4,
        )
    def invoke(self):
        _ = self._complation(stream=False)


@MethodManager.register_method("OpenAIChatStream")
class OpenAIChatStream(OpenAIChat):
    def __init__(self, api_url, api_key="none", model="tgi"):
        super().__init__(api_url, api_key, model)

    def invoke(self):
        chat_completion = self._complation(stream=True)
        for message in chat_completion:
            _ = message.choices[0].delta.content