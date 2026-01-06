from __future__ import annotations

import base64
import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import json

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import requests

# -------------------------
# Config / setup
# -------------------------
load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("Missing DATABASE_URL in environment/.env")


engine: Engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    # Optional: prevent extremely long waits
    pool_timeout=10,
)

app = FastAPI(title="AI Assistant (DB + Plot)")

# -------------------------
# Request / response models
# -------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    # Optional: pass a short convo history if you want
    history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    text: str
    plot_png_base64: Optional[str] = None


# -------------------------
# Safety helpers (server-side)
# -------------------------
FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy)\b|;",
    re.IGNORECASE,
)

def enforce_read_only_sql(sql: str) -> None:
    s = sql.strip()
    if not re.search(r"\bselect\b", s, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")
    if FORBIDDEN_SQL.search(s):
        raise ValueError("Forbidden SQL detected (write/DDL or multi-statement).")

def add_hard_limit(sql: str, limit: int = 200) -> str:
    """
    Simple hard limit. For production, prefer:
    - allowlisted views
    - SQL parser + rewrite
    - statement timeout
    """
    # If user already includes LIMIT, keep the smaller one? (simple approach: always append)
    return f"{sql.rstrip()}\nLIMIT {int(limit)}"


# -------------------------
# Tool implementations
# -------------------------
def run_sql(sql: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    enforce_read_only_sql(sql)
    safe_sql = add_hard_limit(sql, limit=200)

    with engine.connect() as conn:
        # Optional: set statement timeout for Postgres
        # conn.execute(text("SET LOCAL statement_timeout = 2000"))
        result = conn.execute(text(safe_sql), params or {})
        rows = [dict(r._mapping) for r in result.fetchall()]

    return {"rows": rows, "row_count": len(rows), "truncated": True if len(rows) == 200 else False}


def make_plot(
    title: str,
    kind: str,
    x_key: str,
    y_key: str,
    data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if kind not in ("line", "bar"):
        raise ValueError("kind must be 'line' or 'bar'")
    if not data:
        raise ValueError("No data provided for plot.")

    df = pd.DataFrame(data)
    if x_key not in df.columns or y_key not in df.columns:
        raise ValueError(f"Missing x_key or y_key in data. Have: {list(df.columns)}")

    # Try to coerce x-axis to datetime if possible
    try:
        df[x_key] = pd.to_datetime(df[x_key])
        df = df.sort_values(x_key)
    except Exception:
        pass

    plt.figure()
    if kind == "line":
        plt.plot(df[x_key], df[y_key])
    else:
        plt.bar(df[x_key], df[y_key])

    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)

    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return {"png_base64": b64}


# -------------------------
# OpenAI tool schemas
# -------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Run a read-only SQL SELECT query to fetch data needed to answer the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A single SELECT statement."},
                    "params": {
                        "type": "object",
                        "description": "Optional bind parameters for the SQL query.",
                        "additionalProperties": True,
                    },
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_plot",
            "description": "Create a simple chart from provided rows and return a base64-encoded PNG.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "kind": {"type": "string", "enum": ["line", "bar"]},
                    "x_key": {"type": "string"},
                    "y_key": {"type": "string"},
                    "data": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["title", "kind", "x_key", "y_key", "data"],
            },
        },
    },
]

def ollama_chat(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools

    r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def get_tool_calls(ollama_resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Ollama returns tool calls under message.tool_calls :contentReference[oaicite:6]{index=6}
    msg = ollama_resp.get("message", {}) or {}
    return msg.get("tool_calls", []) or {}


# -------------------------
# Tool-calling loop (Responses API)
# -------------------------
def extract_tool_calls(resp: Any) -> List[Dict[str, Any]]:
    """
    The SDK returns a structured response object. We’ll look through output items for tool calls.
    """
    calls: List[Dict[str, Any]] = []
    for item in getattr(resp, "output", []) or []:
        # Tool calls typically appear as items with type == "tool_call"
        if getattr(item, "type", None) == "tool_call":
            calls.append(
                {
                    "id": getattr(item, "id", None),
                    "name": getattr(item, "name", None),
                    "arguments": getattr(item, "arguments", None) or {},
                }
            )
    return calls


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    system_instructions = (
        "You are a helpful assistant for an internal analytics app.\n"
        "You do not know any table or data unless you call run_sql.\n"
        "If the user asks about schemas/tables/columns/rows, ALWAYS call run_sql.\n"
        "- If you need facts from the database, call run_sql.\n"
        "- Only request data that is necessary.\n"
        "- If user asks for a chart, call make_plot after you have the data.\n"
        "- Never ask for or reveal secrets.\n"
        "- Be concise."
    )

    # Ollama messages are the same idea: list of {role, content}
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_instructions}]
    if req.history:
        messages.extend(req.history)
    messages.append({"role": "user", "content": req.message})

    plot_b64: Optional[str] = None

    # Agent loop: model -> tool_calls -> execute -> tool messages -> model ... :contentReference[oaicite:7]{index=7}
    for _ in range(6):  # max turns to avoid infinite loops
        resp = ollama_chat(messages=messages, tools=TOOLS)

        assistant_msg = resp.get("message", {}) or {}
        tool_calls = assistant_msg.get("tool_calls") or []

        # Always append the assistant message to the conversation
        # (may contain content, or tool_calls, or both depending on model)
        messages.append(assistant_msg)

        # No tool calls => final natural language answer
        if not tool_calls:
            final_text = assistant_msg.get("content", "") or ""
            return ChatResponse(text=final_text, plot_png_base64=plot_b64)

        # Execute all tool calls returned
        for tc in tool_calls:
            fn = (tc.get("function") or {})
            name = fn.get("name")
            args = fn.get("arguments") or {}

            try:
                if name == "run_sql":
                    result = run_sql(sql=args["sql"], params=args.get("params"))

                    messages.append({
                        "role": "tool",
                        "tool_name": "run_sql",
                        "content": json.dumps(result, ensure_ascii=False),
                    })


                elif name == "make_plot":
                    result = make_plot(
                        title=args["title"],
                        kind=args["kind"],
                        x_key=args["x_key"],
                        y_key=args["y_key"],
                        data=args["data"],
                    )
                    plot_b64 = result.get("png_base64")
                    messages.append({
                        "role": "tool",
                        "tool_name": "make_plot",
                        "content": "ok",
                    })

                else:
                    messages.append({
                        "role": "tool",
                        "tool_name": name or "unknown",
                        "content": f"error: unknown tool {name}",
                    })

            except Exception as e:
                messages.append({
                    "role": "tool",
                    "tool_name": name or "unknown",
                    "content": f"error: {e}",
                })

    # If we exit the loop, return whatever we have
    return ChatResponse(
        text="I couldn't finish the request within the tool loop limit. Try narrowing the question.",
        plot_png_base64=plot_b64,
    )