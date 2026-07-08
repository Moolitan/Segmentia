from __future__ import annotations

import argparse
import os

from common import (
    DEFAULT_MODEL_PATH,
    add_common_args,
    default_log_file,
    start_vllm_server,
    stop_server,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start vLLM with CSKCache connector.")
    add_common_args(parser)
    parser.add_argument("--model-path", default=os.environ.get("VLLM_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096")))
    parser.add_argument("--gpu-util", type=float, default=float(os.environ.get("VLLM_GPU_UTIL", "0.85")))
    parser.add_argument("--wait-timeout-s", type=int, default=900)
    parser.add_argument("--stop", action="store_true", help="Stop the pid recorded for this port.")
    args = parser.parse_args()

    if args.stop:
        stop_server(args.port)
        return

    start_vllm_server(
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        served_name=args.model,
        api_key=args.api_key,
        log_file=default_log_file(args.port),
        max_model_len=args.max_model_len,
        gpu_util=args.gpu_util,
        wait_timeout_s=args.wait_timeout_s,
    )
    print("Server is left running for test03. Stop with: python test02_server_start.py --stop")


if __name__ == "__main__":
    main()
