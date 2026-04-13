def build_header(access_token: str) -> dict:
    return {
        "Content-Type": "application/json",
        "code_company": "OKP",
        "code_service": "merchant_commercial",
        "Authorization": f"Bearer {access_token}",
    }


def build_stream_body(
    model: str,
    commercial_prompt: str,
    system_prompt: str,
) -> dict:
    model_name = ""

    if model == "gemma3:27b":
        model_name = "gemma3-27b-vllm"

    if not model_name:
        raise ValueError("wrong model")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": commercial_prompt},
    ]

    return {
        "model": model_name,
        "stream": True,
        "temperature": 1,
        "top_k": 40,
        "top_p": 0.9,
        "presence_penalty": 0.6,
        "frequency_penalty": 0.4,
        "stop": ["\nUser:", "\nAssistant:"],
        "messages": messages,
        "think": False,
    }


def get_url(model: str) -> str:
    if model == "gemma3:27b":
        return "https://vaiv-gemma3.xhub.co.kr/v1/chat/completions"

    raise ValueError("wrong model")
