from __future__ import annotations

import argparse

from common import add_common_args, base_url, request_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a tiny decode request to vLLM.")
    add_common_args(parser)
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "prompt": "Answer with exactly one word: OK",
        "max_tokens": 4,
        "temperature": 0,
    }
    response = request_json(
        "POST",
        f"{base_url(args.host, args.port)}/v1/completions",
        api_key=args.api_key,
        payload=payload,
        timeout=120.0,
    )
    text = response["choices"][0].get("text", "")
    print("tiny decode ok")
    print("output:", repr(text))


if __name__ == "__main__":
    main()
