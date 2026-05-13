import logging
import re
import os
import inspect
import tiktoken

logger = logging.getLogger("main")

import openai
openai.api_key = os.environ["OPENAI_API_KEY"]
from openai import OpenAI
client = OpenAI()


def generate_response(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    stop_tokens: list[str] | None = None,
    use_tools: bool = False,
    n: int = 20
) -> tuple[list[str], dict[str, int]]:
    """Send a request to the OpenAI API."""

    logger.info(
        f"Send a request to the language model from {inspect.stack()[1].function}"
    )
    gen_kwargs = {}

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stop=stop_tokens if stop_tokens else None,
        n=n,
        **gen_kwargs
    )

    message = [*map(lambda x: x.message.content, response.choices)]

    info = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return message, info

# ------uncomment the below code and comment the above code if you want to use local qwen model instead of GPT----------


# import logging
# import inspect
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# logger = logging.getLogger("main")

# _MODEL = None
# _TOKENIZER = None


# def load_model():
#     global _MODEL, _TOKENIZER

#     if _MODEL is None:
#         logger.info("Loading Qwen model (4-bit)...")

#         model_name = "Qwen/Qwen2.5-7B-Instruct"

#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             # llm_int8_enable_fp32_cpu_offload=True
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4"
#         )

#         _TOKENIZER = AutoTokenizer.from_pretrained(
#             model_name,
#             trust_remote_code=True
#         )

#         _MODEL = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             quantization_config=bnb_config,
#             device_map="auto",
#             torch_dtype=torch.float16,
#             trust_remote_code=True
#         )

#         _MODEL.eval()

#         logger.info("Qwen loaded successfully")

#     return _MODEL, _TOKENIZER


# def generate_response(
#     messages,
#     model=None,
#     temperature=1,
#     stop_tokens=None,
#     use_tools=False,
#     n=20
# ):

#     logger.info(
#         f"Send a request to local Qwen model from {inspect.stack()[1].function}"
#     )

#     model, tokenizer = load_model()

#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )

#     device = next(model.parameters()).device
#     inputs = tokenizer(text, return_tensors="pt").to(device)

#     do_sample = (n > 1 and temperature > 0)
#     with torch.no_grad():
#         outputs = model.generate(
#         **inputs,
#         max_new_tokens=32,
#         do_sample=do_sample,
#         temperature=temperature if do_sample else None,
#         top_p=0.9 if do_sample else None,
#         num_return_sequences=n,
#         pad_token_id=tokenizer.eos_token_id
#     )

#     responses = []

#     for i in range(n):
#         out = outputs[i]

#         decoded = tokenizer.decode(
#             out[inputs["input_ids"].shape[1]:],
#             skip_special_tokens=True
#         )

#         if stop_tokens:
#             for token in stop_tokens:
#                 if token in decoded:
#                     decoded = decoded.split(token)[0]
#                     break

#         responses.append(decoded.strip())

#     # Dummy token usage (to match original interface)
#     info = {
#         "prompt_tokens": 0,
#         "completion_tokens": 0,
#         "total_tokens": 0,
#     }

#     return responses, info

