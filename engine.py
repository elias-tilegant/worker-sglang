import subprocess
import time
import requests
import os


class SGlangEngine:
    def __init__(
        self,
        model=os.getenv("MODEL_PATH") or os.getenv("MODEL_NAME"),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 30000)),
    ):
        self.model = model
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.process = None

    def start_server(self):
        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        # Ordered option groups. The first non-empty environment variable in a
        # group wins, which lets newer names coexist with legacy worker names.
        option_groups = [
            [("MODEL_PATH", "--model-path"), ("MODEL_NAME", "--model-path")],
            [("TOKENIZER_PATH", "--tokenizer-path")],
            [("TOKENIZER_MODE", "--tokenizer-mode")],
            [("LOAD_FORMAT", "--load-format")],
            [("DTYPE", "--dtype")],
            [("CONTEXT_LENGTH", "--context-length")],
            [("QUANTIZATION", "--quantization")],
            [("SERVED_MODEL_NAME", "--served-model-name")],
            [("CHAT_TEMPLATE", "--chat-template")],
            [("JSON_MODEL_OVERRIDE_ARGS", "--json-model-override-args")],
            [("MEM_FRACTION_STATIC", "--mem-fraction-static")],
            [("MAX_RUNNING_REQUESTS", "--max-running-requests")],
            [("MAX_TOTAL_TOKENS", "--max-total-tokens")],
            [("CHUNKED_PREFILL_SIZE", "--chunked-prefill-size")],
            [("MAX_PREFILL_TOKENS", "--max-prefill-tokens")],
            [("SCHEDULE_POLICY", "--schedule-policy")],
            [("SCHEDULE_CONSERVATIVENESS", "--schedule-conservativeness")],
            [("MAMBA_SCHEDULER_STRATEGY", "--mamba-scheduler-strategy")],
            [("KV_CACHE_DTYPE", "--kv-cache-dtype")],
            [
                ("TP_SIZE", "--tp-size"),
                ("TENSOR_PARALLEL_SIZE", "--tensor-parallel-size"),
            ],
            [("PIPELINE_PARALLEL_SIZE", "--pipeline-parallel-size")],
            [("EXPERT_PARALLEL_SIZE", "--expert-parallel-size")],
            [("STREAM_INTERVAL", "--stream-interval")],
            [("RANDOM_SEED", "--random-seed")],
            [("LOG_LEVEL", "--log-level")],
            [("LOG_LEVEL_HTTP", "--log-level-http")],
            [("API_KEY", "--api-key")],
            [("FILE_STORAGE_PATH", "--file-storage-path")],
            [("DATA_PARALLEL_SIZE", "--data-parallel-size")],
            [("LOAD_BALANCE_METHOD", "--load-balance-method")],
            [("ATTENTION_BACKEND", "--attention-backend")],
            [("SAMPLING_BACKEND", "--sampling-backend")],
            [("TOOL_CALL_PARSER", "--tool-call-parser")],
            [("REASONING_PARSER", "--reasoning-parser")],
            [("SPECULATIVE_ALGORITHM", "--speculative-algorithm")],
            [("SPECULATIVE_DRAFT_MODEL_PATH", "--speculative-draft-model-path")],
            [("SPECULATIVE_DRAFT_MODEL_REVISION", "--speculative-draft-model-revision")],
            [("SPECULATIVE_DRAFT_LOAD_FORMAT", "--speculative-draft-load-format")],
            [
                (
                    "SPECULATIVE_DRAFT_MODEL_QUANTIZATION",
                    "--speculative-draft-model-quantization",
                )
            ],
            [("SPECULATIVE_NUM_STEPS", "--speculative-num-steps")],
            [("SPECULATIVE_EAGLE_TOPK", "--speculative-eagle-topk")],
            [("SPECULATIVE_NUM_DRAFT_TOKENS", "--speculative-num-draft-tokens")],
            [("SPECULATIVE_ACCEPT_THRESHOLD_SINGLE", "--speculative-accept-threshold-single")],
            [("SPECULATIVE_ACCEPT_THRESHOLD_ACC", "--speculative-accept-threshold-acc")],
            [("SPECULATIVE_ATTENTION_MODE", "--speculative-attention-mode")],
            [("SPECULATIVE_TOKEN_MAP", "--speculative-token-map")],
            [("LORA_PATHS", "--lora-paths")],
            [("NSA_PREFILL_BACKEND", "--nsa-prefill-backend")],
            [("NSA_DECODE_BACKEND", "--nsa-decode-backend")],
        ]

        # Boolean flags
        boolean_flags = [
            "SKIP_TOKENIZER_INIT",
            "TRUST_REMOTE_CODE",
            "LOG_REQUESTS",
            "SHOW_TIME_COST",
            "DISABLE_RADIX_CACHE",
            "DISABLE_CUDA_GRAPH",
            "DISABLE_OUTLINES_DISK_CACHE",
            "ENABLE_TORCH_COMPILE",
            "ENABLE_P2P_CHECK",
            "ENABLE_FLASHINFER_MLA",
            "TRITON_ATTENTION_REDUCE_IN_FP32",
            "ENABLE_MIXED_CHUNK",
            "ENABLE_OVERLAP",
            "ENABLE_METRICS",
            "ENABLE_CACHE_REPORT",
        ]

        # Add options from environment variables only if they are set.
        for option_group in option_groups:
            for env_var, option in option_group:
                value = os.getenv(env_var)
                if value is not None and value != "":
                    if env_var == "LORA_PATHS":
                        for lora_path in value.split(","):
                            command.extend([option, lora_path.strip()])
                    else:
                        command.extend([option, value])
                    break

        # Add boolean flags only if they are set to true
        for flag in boolean_flags:
            if os.getenv(flag, "").lower() in ("true", "1", "yes"):
                command.append(f"--{flag.lower().replace('_', '-')}")

        self.process = subprocess.Popen(command, stdout=None, stderr=None)
        print(f"Server started with PID: {self.process.pid}")

    def wait_for_server(self, timeout=900, interval=5):
        start_time = time.time()
        health_url = f"{self.base_url}/health"
        models_url = f"{self.base_url}/v1/models"
        while time.time() - start_time < timeout:
            if self.process and self.process.poll() is not None:
                raise RuntimeError(
                    f"SGLang server process exited with code {self.process.returncode}"
                )

            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    print("Server is ready!")
                    return True
            except requests.RequestException:
                pass

            try:
                response = requests.get(models_url, timeout=5)
                if response.status_code == 200:
                    print("Server is ready!")
                    return True
            except requests.RequestException:
                pass

            elapsed = int(time.time() - start_time)
            print(f"Waiting for server... ({elapsed}s / {timeout}s)")
            time.sleep(interval)
        raise TimeoutError("Server failed to start within the timeout period.")

    def shutdown(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            print("Server shut down.")
