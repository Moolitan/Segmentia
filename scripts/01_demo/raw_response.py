import openai
import os
vllm_api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
vllm_port = 8000
client = openai.OpenAI(
    api_key=vllm_api_key,
    base_url=f"http://localhost:{vllm_port}/v1"
)

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="Qwen3",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools,
    tool_choice="auto",
    extra_body={"chat_template_kwargs": {"enable_thinking": True}, "min_p": 0},
    temperature=0.6,
)

print(response.choices[0].message)