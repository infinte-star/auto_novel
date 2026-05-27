from __future__ import annotations

import argparse
import concurrent.futures
import os
import time
from dataclasses import dataclass
from typing import Iterable

try:
    from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, OpenAI
except ModuleNotFoundError as exc:
    OPENAI_IMPORT_ERROR = exc

    class _MissingOpenAIError(Exception):
        pass

    APIConnectionError = APIError = APIStatusError = APITimeoutError = _MissingOpenAIError  # type: ignore[assignment]
    OpenAI = None  # type: ignore[assignment]
else:
    OPENAI_IMPORT_ERROR = None


DEFAULT_BASE_URLS = [
    "https://token-plan-sgp.xiaomimimo.com/v1",
    "https://token-plan-cn.xiaomimimo.com/v1",
]

DEFAULT_KEYS: list[str] = []


@dataclass(frozen=True)
class CheckResult:
    base_url: str
    key: str
    ok: bool
    elapsed: float
    error: str = ""


def mask_key(key: str) -> str:
    if len(key) <= 14:
        return key[:4] + "***"
    return f"{key[:6]}...{key[-6:]}"


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        value = value.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def load_keys(path: str | None) -> list[str]:
    if not path:
        env_value = os.getenv("TOKEN_PLAN_KEYS", "")
        values = [env_value] if env_value else DEFAULT_KEYS
        keys = []
        for value in values:
            keys.extend(part.strip() for part in value.split(","))
        return unique_preserve_order(keys)

    keys = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            keys.extend(part.strip() for part in line.split(","))
    return unique_preserve_order(keys)


def parse_base_urls(values: list[str] | None) -> list[str]:
    if not values:
        env_value = os.getenv("TOKEN_PLAN_BASE_URLS", "")
        values = [env_value] if env_value else DEFAULT_BASE_URLS

    urls = []
    for value in values:
        urls.extend(part.strip() for part in value.split(","))
    return unique_preserve_order(urls)


def normalize_error(exc: Exception) -> str:
    if isinstance(exc, APIStatusError):
        body = ""
        try:
            body = str(exc.response.text)
        except Exception:
            body = ""
        message = body.strip() or str(exc)
        return f"HTTP {exc.status_code}: {message[:240]}"
    if isinstance(exc, APITimeoutError):
        return "timeout"
    if isinstance(exc, APIConnectionError):
        return f"connection error: {str(exc)[:220]}"
    if isinstance(exc, APIError):
        return f"api error: {str(exc)[:240]}"
    return f"{type(exc).__name__}: {str(exc)[:240]}"


def should_retry(exc: Exception) -> bool:
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 429, 500, 502, 503, 504}
    return isinstance(exc, (APITimeoutError, APIConnectionError))


def check_one(base_url: str, key: str, model: str, timeout: float, retries: int, retry_delay: float) -> CheckResult:
    if OpenAI is None:
        raise RuntimeError("Missing dependency: run `pip install -r requirements.txt` before checking keys.")
    started = time.perf_counter()
    client = OpenAI(base_url=base_url, api_key=key, timeout=timeout, max_retries=0)
    last_error = ""
    for attempt in range(retries + 1):
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0,
            )
            return CheckResult(base_url=base_url, key=key, ok=True, elapsed=time.perf_counter() - started)
        except Exception as exc:
            last_error = normalize_error(exc)
            if attempt >= retries or not should_retry(exc):
                break
            time.sleep(retry_delay * (attempt + 1))

    return CheckResult(
        base_url=base_url,
        key=key,
        ok=False,
        elapsed=time.perf_counter() - started,
        error=last_error,
    )


def print_result(result: CheckResult, show_full_key: bool) -> None:
    status = "OK " if result.ok else "BAD"
    key = result.key if show_full_key else mask_key(result.key)
    detail = "" if result.ok else f" | {result.error}"
    print(f"{status} | {result.base_url} | {key} | {result.elapsed:.2f}s{detail}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check token-plan API keys against OpenAI-compatible base URLs.")
    parser.add_argument("--model", default="mimo-v2.5-pro", help="model name used for the test request")
    parser.add_argument("--base-url", action="append", help="base URL to test; can be used multiple times or comma-separated")
    parser.add_argument("--keys-file", help="optional file containing keys, one per line or comma-separated")
    parser.add_argument("--timeout", type=float, default=20.0, help="request timeout in seconds")
    parser.add_argument("--workers", type=int, default=8, help="parallel request count")
    parser.add_argument("--retries", type=int, default=1, help="retry count for rate limits and transient errors")
    parser.add_argument("--retry-delay", type=float, default=2.0, help="base retry delay in seconds")
    parser.add_argument("--show-full-key", action="store_true", help="print full keys instead of masked keys")
    args = parser.parse_args()

    if OPENAI_IMPORT_ERROR is not None:
        parser.error("missing dependency: run `pip install -r requirements.txt` before checking keys")

    keys = load_keys(args.keys_file)
    base_urls = parse_base_urls(args.base_url)
    if not keys:
        parser.error("no keys provided; set TOKEN_PLAN_KEYS or pass --keys-file")
    jobs = [(base_url, key) for base_url in base_urls for key in keys]
    ok_results: list[CheckResult] = []

    duplicate_count = len(DEFAULT_KEYS) - len(unique_preserve_order(DEFAULT_KEYS))
    if duplicate_count and not args.keys_file:
        print(f"Loaded {len(keys)} unique keys ({duplicate_count} duplicate removed).")
    else:
        print(f"Loaded {len(keys)} keys.")
    print(f"Testing {len(base_urls)} base URL(s), {len(jobs)} request(s), model={args.model}\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(check_one, base_url, key, args.model, args.timeout, args.retries, args.retry_delay)
            for base_url, key in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print_result(result, args.show_full_key)
            if result.ok:
                ok_results.append(result)

    print("\nUsable keys:")
    if not ok_results:
        print("(none)")
        return 1

    for result in sorted(ok_results, key=lambda item: (item.base_url, item.key)):
        key = result.key if args.show_full_key else mask_key(result.key)
        print(f"{result.base_url} | {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
