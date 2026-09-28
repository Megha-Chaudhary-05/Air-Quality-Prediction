"""Drives a real Streamlit session over the websocket to smoke-test the app.

Streamlit only executes the script once a client connects, so this opens a
session against a running server, waits for the script to finish, and reports
any element or exception the server emitted.

The wire protocol matters here: Streamlit sends **binary protobuf** ForwardMsgs
(``streamlit.proto.ForwardMsg_pb2``), not JSON. An earlier version of this tool
parsed JSON, so it silently matched nothing and reported "no exceptions raised"
for a completely broken app.

Usage:
    python tools/smoke_test.py [ws://host:port/_stcore/stream]
"""

import asyncio
import sys
from collections import Counter
from urllib.parse import urlparse, urlunparse

import requests
import websockets
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg

# Script finish states reported by session_status.
FINISHED_STATES = {"finished", "rerun"}


def to_ws(url):
    parsed = urlparse(url)
    scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
    return urlunparse(parsed._replace(scheme=scheme))


def to_http(url):
    parsed = urlparse(url)
    scheme = {"ws": "http", "wss": "https"}.get(parsed.scheme, parsed.scheme)
    return urlunparse(parsed._replace(scheme=scheme))


async def drive(url, settle_timeout=90):
    """Connects, drains forward messages, returns (counts, errors, notes)."""
    counts = Counter()
    errors = []
    notes = []
    elapsed = 0.0
    async with websockets.connect(url, max_size=None, open_timeout=30,
                                  subprotocols=["streamlit"]) as ws:
        while elapsed < settle_timeout:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=15)
            except asyncio.TimeoutError:
                break

            if isinstance(raw, str):
                counts["<str-frame>"] += 1
                continue

            msg = ForwardMsg()
            try:
                msg.ParseFromString(raw)
            except Exception:
                counts["<unparsed-frame>"] += 1
                continue

            kind = msg.WhichOneof("type")
            counts[f"delta:{kind}"] += 1
            elapsed += 0.01

            if kind == "delta":
                element = msg.delta.new_element
                if element.WhichOneof("type") in ("exception", "alert"):
                    body = element.exception.message if element.WhichOneof(
                        "type") == "exception" else element.alert.body
                    if element.WhichOneof("type") == "exception":
                        errors.append(
                            (element.exception.type or "Exception", body))
                    else:
                        notes.append(("alert", body))
                elif element.WhichOneof("type") == "text":
                    if element.text.body.strip():
                        notes.append(("text", element.text.body[:200]))
            elif kind == "session_status":
                state = msg.session_status.script_finished
                if str(state).endswith("_finished") or str(state).endswith("_rerun"):
                    counts["<script-finished>"] += 1

    return counts, errors, notes


def main():
    url = to_ws(sys.argv[1] if len(sys.argv) > 1
                else "http://localhost:8501")
    base = to_http(url)

    health = requests.get(f"{base}/_stcore/health", timeout=30)
    health.raise_for_status()
    print(f"server health: {health.text.strip()} (HTTP {health.status_code})")
    print(f"driving:       {url}/_stcore/stream")

    counts, errors, notes = asyncio.run(drive(f"{url}/_stcore/stream"))

    deltas = {k: v for k, v in counts.items() if k.startswith("delta:")}
    total = sum(deltas.values())
    print(f"\nreceived {sum(counts.values())} forward messages "
          f"({total} element deltas)")
    for kind, n in sorted(deltas.items(), key=lambda kv: -kv[1]):
        print(f"  {kind:<32} {n}")
    odd = [k for k in counts if not k.startswith("delta:") and k != "<script-finished>"]
    for k in odd:
        print(f"  !! unexpected frame type: {k} ({counts[k]})")

    if errors:
        print(f"\n{len(errors)} EXCEPTION(S):")
        for etype, body in errors:
            print(f"  [{etype}] {body}")
        print("\nFAILED")
        return 1

    if total == 0:
        print("\nFAILED: no elements rendered - script did not run.")
        return 1

    print("\nno exceptions raised")
    return 0


if __name__ == "__main__":
    sys.exit(main())
