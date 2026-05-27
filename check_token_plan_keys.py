from __future__ import annotations

import argparse
import concurrent.futures
import os
import time
from dataclasses import dataclass
from typing import Iterable

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, OpenAI


DEFAULT_BASE_URLS = [
    "https://token-plan-sgp.xiaomimimo.com/v1",
    "https://token-plan-cn.xiaomimimo.com/v1",
]

DEFAULT_KEYS = [
    "tp-c7tf4dc4tcjqihzbgf4gxeowhre1puvjuzq6jujiacmtl89g",
    "tp-cbngbpgso5k15fhgfdj7th0cz1lnvub4yazr97o9ek2fwmzn",
    "tp-c4ncuopszt9d1xqi53xomryce4ovf1kzbq9yz21eryhdym63",
    "tp-sbz20egchfb2pt10vxxu24xlmrpuh0ily9ki0y6z3f1ao1t9",
    "tp-sgvnquz17psyyg2wbhz4x3dl4fnt1uzsx78j5t4eppcj31pv",
    "tp-cj1uodq9mt5ewqj1hn2fzvghrme2bbracfkkns0hxfrcpde0",
    "tp-shq2lvrd319s5czml8kfx8pp4o02kggik4mqw3avb25ib8fc",
    "tp-cn4gp8wfzw188wv8tko6e2xv3o0lkpwus38e48qkhql5nyn4",
    "tp-cxdksnfo646rghr9dnn8ov5neilt3nfc371o09c1wlrpz42e",
    "tp-cmjv75smkeprvhaxferim1docvt50feh0ogmbnvrihuun1nz",
    "tp-c4d16u7uceoi71w96iroav23evjn71l5nphyplumpev1x4cz",
    "tp-cg02yumls4jgyb1cifqi2o94vviffgybwnkc5v9shiwmszr9",
    "tp-cdfljltlwj93aie7ylea1yw2ls65omnbjg428cms3i1shtpq",
    "tp-sxe7iiq3tqhbqszkub3y03bxeycjfs5mrz3kc0b76vaqqjhn",
    "tp-smnl8944f3eqcfkwmtbeflw3y3jp9lsym0y3a70vzjrw1fey",
    "tp-ccg7kir0ez6cfjc28j8ae1mlktldrml3nhf3bji414gwfw17",
    "tp-cstfxavyq3q2q6mun7frz7i6ie80emav0z0kgz48tg7odyrd",
    "tp-crh4kxm2lhpspb096vsoe5620rva7hm3la2h2h5664lhxlif",
    "tp-c5otezemijw448of5eycvpw6qxr2wujzc9v5olz52gl0tjk8",
    "tp-ccg7kir0ez6cfjc28j8ae1mlktldrml3nhf3bji414gwfw17",
    "tp-cfsjk3qip63iw4bh16nyr6pf2kh82biay0h2hfs94ad7pur1",
    "tp-cany1u6z7kvlsro1bvns4lxltq2oauza5gj8cwifhr2aq4ov",
    "tp-cha0k3ethcflvzsglypazgimesg86n7onop4sedr0tn01ysh",
    "tp-c9sp7c797gb21flxskvl6hh6z510hkqzf77kr8wp4zxadmsd",
    "tp-cwfj9fg5qz8ufy8bf1rswlp6cj640mi6m0xttz8a0vce9yjy",
    "tp-c0zunlz92jw2zmxthmzg4fvdtopbbvn03ud4gx3swafdvu26",
    "tp-se8c5mmrf1hauy0qyutlk9bek2gdjoiyfn6sr68ug2ojfxbg",
    "tp-cvhy3ebo25viv07y7qlnlh3u4sk8ltyfvfkzgxoomerxstov",
    "tp-c6haieuwzhs3763206by8em1nmeixql89967966rpk3or9bf",
    "tp-cphcwcftmrzgbo2560huu80x9ojognk3jpvpruiot25t2ahf",
    "tp-smusanjj754dxky6fkyh07ffrtznrcmboc9td8ypml2xvech",
    "tp-cxsrtcib8gfllm4hkrsxntdtt8hg67wefztdbhgi70264cv5",
    "tp-cwzec7c52g1bpsf0enyrqa8aqx7lhgdnsqqgxcec5h9vaqv0",
    "tp-s7g9npcud7h7zu9wst8om8bztv27idtnut0unr8pbxv86411",
    "tp-c0bptuhm4rhbz5tdque8m7494nywj5aqz8dbc4tbpjk6jdm0",
    "tp-s4k0utff4u18dcyx5zplvp6iywq7ttandz1zw1iew7luqw6p",
    "tp-clkcaxu7epll3hcv1jee659dna0cjvq1jal5cuxfs3x4qdzv",
    "tp-che3qx8gv7yb2ifhyhe8dnr4czm6zcss27sw5lq5z55nmcmb",
]


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
        return unique_preserve_order(DEFAULT_KEYS)

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

    keys = load_keys(args.keys_file)
    base_urls = parse_base_urls(args.base_url)
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
