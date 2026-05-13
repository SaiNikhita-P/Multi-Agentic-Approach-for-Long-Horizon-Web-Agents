import logging
import re
import os
import inspect
import tiktoken
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("main")

from openai import OpenAI
client = OpenAI()


def num_tokens_from_messages(messages, model):
    """Return the number of tokens used by a list of messages.
    Borrowed from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    """
    if model in {
        "GPT-3-5-turbo-chat",
        "GPT-3-5-16k-turbo-chat",
        "gpt-3.5-16k-turbo-chat",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-3.5-turbo-1106",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    else:
        raise NotImplementedError(
            f"num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."
        )
    """

    tokens_per_message = 0
    tokens_per_name = 0

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


MAX_TOKENS = {
    "gpt-3.5-turbo": 4097,
    "GPT-3-5-turbo-chat": 4097,
    "gpt-3.5-turbo-0301": 4097,
    "gpt-3.5-turbo-0613": 4097,
    "gpt-3.5-turbo-16k-0613": 16385,
    "gpt-3.5-turbo-1106": 16385,
    "gpt-4": 8192,
    "gpt-4o": 16385,
    "GPT-3-5-16k-turbo-chat": 16385,
    "gpt-4-32k": 32000,
}


def get_mode(model: str) -> str:
    """Check if the model is a chat model."""

    """
    if model in [
        "GPT-3-5-turbo-chat",
        "GPT-3-5-16k-turbo-chat",
        "gpt-3.5-16k-turbo-chat",
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4",
        "gpt-4o",
        "gpt-4-32k-0613",
    ]:
        return "chat"
    elif model in [
        "davinci-002",
        "gpt-3.5-turbo-instruct-0914",
    ]:
        return "completion"
    else:
        raise ValueError(f"Unknown model: {model}")
    """
    return "chat"

def generate_response(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    stop_tokens: list[str] | None = None,
    use_tools: bool = False,
) -> tuple[str, dict[str, int]]:
    """Send a request to the OpenAI API."""

    logger.info(
        f"Send a request to the language model from {inspect.stack()[1].function}"
    )
    gen_kwargs = {}

    if get_mode(model) == "chat":
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stop=stop_tokens if stop_tokens else None,
            **gen_kwargs,
            # n=20
            n=10
        )
        breakpoint()
        message = response.choices[0].message.content
    else:
        prompt = "\n\n".join(m["content"] for m in messages) + "\n\n"
        response = openai.Completion.create(
            prompt=prompt,
            engine=model,
            temperature=temperature,
            stop=stop_tokens if stop_tokens else None,
        )
        message = response["choices"][0]["text"]
    info = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    if message is None: message = ""

    return message, info


def extract_from_response(response: str, backtick="```") -> str:
    if backtick == "```":
        # Matches anything between ```<optional label>\n and \n```
        pattern = r"```(?:[a-zA-Z]*)\n?(.*?)\n?```"
    elif backtick == "`":
        pattern = r"`(.*?)`"
    else:
        raise ValueError(f"Unknown backtick: {backtick}")
    match = re.search(
        pattern, response, re.DOTALL
    )  # re.DOTALL makes . match also newlines
    if match:
        extracted_string = match.group(1)
    else:
        extracted_string = ""

    return extracted_string

import logging
import re
import os
import inspect
import tiktoken
import requests
from openai import OpenAI

logger = logging.getLogger("main")


class Caller:
    """Unified caller for OpenAI and Ollama."""

    def __init__(self, model):
        self.model = model
        self.MAX_TOKENS = {
             "GPT-3-5-turbo-chat": 4097,
             "gpt-3.5-turbo-0301": 4097,
             "gpt-3.5-turbo-0613": 4097,
             "gpt-3.5-turbo-16k-0613": 16385,
             "gpt-3.5-turbo-1106": 16385,
             "gpt-4": 8192,
             "gpt-4o": 16385,
             "gpt-4o-mini":128000,
             "GPT-3-5-16k-turbo-chat": 16385,
             "gpt-4-32k": 32000,
             "gpt-3.5-turbo": 16385,
             "gpt-35-turbo": 8192,
             "deepseek-r1": 57344,
             "deepseek-v3": 57344,
             "llama3.2-vision:latest": 8192,  # adjust depending on your Ollama model
             "mistral": 8192,
             "phi3:mini": 4096,
             "qwen2.5:1.5b": 32000,
             "qwen2.5:7b": 32000,
             "qwen2.5": 32000,

         }
        # self.use_ollama = "phi3:mini" in model or "mistral" in model or "llama" in model  # heuristic
        # self.use_ollama = ":" in model or model.lower() in ["phi3", "mistral", "llama"]
        self.use_ollama = any(x in model.lower() for x in ["phi3", "mistral", "llama", "qwen"])


        # if not self.use_ollama:
        self.client = OpenAI(api_key="YOUR_API_KEY")
        self.api_base = "http://127.0.0.1:5002/api"  # Ollama local API base
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        self.api_calls = 0

    # changing caller function to call Qwen instead of GPT!
    def call(self, messages, temperature=0.0, n=1, stop_tokens=None):
    # Call either OpenAI or Ollama based on model type.
        try:
            if self.use_ollama:
                # Combine all chat messages into a single prompt string
                prompt = ""
                for m in messages:
                    role = m["role"].capitalize()
                    prompt += f"{role}: {m['content']}\n"

                # Construct the Ollama payload
                payload = {
                    "model": self.model,
                    "prompt": "Answer concisely.\n" + prompt,  
                    "temperature": temperature,
                    "num_predict": 100,  
                    "stream": False     
                }

               
                response = requests.post(f"{self.api_base}/generate", json=payload, timeout=300)

                response.raise_for_status()
                data = response.json()

                result = data.get("response", "").strip()
                self.api_calls += 1 
                if stop_tokens:
                    for token in stop_tokens:
                        if token in result:
                            result = result.split(token)[0].strip()
                            break

                return result
            else:
                # OpenAI model call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    stop=stop_tokens if stop_tokens else None,
                    n=n,
                )
                self.api_calls += 1 

                # Track token usage stats
                self.completion_tokens += response.usage.completion_tokens
                self.prompt_tokens += response.usage.prompt_tokens
                self.total_tokens += response.usage.total_tokens
               


                return response.choices[0].message.content.strip()

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama HTTP error: {e}")
            return ""
        except Exception as e:
            logger.error(f"Error in call: {e}")
            return ""
        
    #--------uncomment the below call() function and comment the above one if you want to use local qwen model instead of GPT or ollama!----------------    

    # def call(self, messages, temperature=0.0, n=1, stop_tokens=None):

    #     try:
    #         # IMPORTANT: explicitly use Qwen version
    #         from mind2web.custom.llm import generate_response as qwen_generate

    #         responses, info = qwen_generate(
    #             messages=messages,
    #             temperature=temperature,
    #             stop_tokens=stop_tokens,
    #             n=n
    #         )

    #         self.api_calls += 1

    #         # keep stats safe
    #         self.prompt_tokens += info.get("prompt_tokens", 0)
    #         self.completion_tokens += info.get("completion_tokens", 0)
    #         self.total_tokens += info.get("total_tokens", 0)

    #         if n == 1:
    #             return responses[0]
    #         else:
    #             return responses

    #     except Exception as e:
    #         logger.error(f"Error in Caller.call: {e}")
    #         return "" if n == 1 else [""] * n
    
    def get_embedding(self, message):
        """Return the embedding of a message."""
        try:
            encoding = self.client.embeddings.create(input=message, model=self.model, encoding_format="float").data[0].embedding
            return encoding
        except Exception as e:
            print(e)
            return None

    def num_tokens_from_messages(self, messages, model):
        """Return the number of tokens used by a list of messages.
        Borrowed from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "GPT-3-5-turbo-chat",
            "GPT-3-5-16k-turbo-chat",
            "gpt-3.5-16k-turbo-chat",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-3.5-turbo-1106",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4o",
            "gpt-3.5-turbo",
            "gpt-35-turbo",
            "gpt-4o-mini",
            "deepseek-r1",
            "deepseek-v3"
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def get_mode(self, model: str) -> str:
        """Check if the model is a chat model."""
        if model in [
            "GPT-3-5-turbo-chat",
            "GPT-3-5-16k-turbo-chat",
            "gpt-3.5-16k-turbo-chat",
            "gpt-3.5-turbo-0301",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k-0613",
            "gpt-3.5-turbo",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4",
            "gpt-4o",
            "gpt-4-32k-0613",
        ]:
            return "chat"
        elif model in [
            "davinci-002",
            "gpt-3.5-turbo-instruct-0914",
        ]:
            return "completion"
        else:
            raise ValueError(f"Unknown model: {model}")

    def extract_from_response(self, response: str, backtick="```") -> str:
        if backtick == "```":
            # Matches anything between ```<optional label>\n and \n```
            pattern = r"```(?:[a-zA-Z]*)\n?(.*?)\n?```"
        elif backtick == "`":
            pattern = r"`(.*?)`"
        else:
            raise ValueError(f"Unknown backtick: {backtick}")
        match = re.search(
            pattern, response, re.DOTALL
        )  # re.DOTALL makes . match also newlines
        if match:
            extracted_string = match.group(1)
        else:
            extracted_string = ""

        return extracted_string

    def compute_cost(self):
        if "gpt-4o-mini" in self.model:
            input_price = 0.15/1000
            output_price = 0.6/1000
            embedding_price = 0.0001
        else:
            raise ValueError(f"Unknown model: {self.model}")
        rate = 1
        cost = float(self.prompt_tokens) / 1000 * input_price * rate + \
               float(self.completion_tokens) / 1000 * output_price * rate 
        return "{:.2f}".format(cost)

