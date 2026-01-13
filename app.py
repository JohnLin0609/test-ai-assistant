from __future__ import annotations
################################################# Record what you update and the date in here. #################################################
### Under each date, separate each update by a title of summary which will be used as git commit.
### Like : --- Initial commit ---
# Update Log
# 2026-01-08
# --- Schema guidance and validation ---
# - Added schema.sql hints + form_name filtering/joining for spc_measurements analysis.
# - Tightened SQL/schema validation and made MAX_SQL_ROWS default no-limit (0).
# - Hardened schema parsing/trace for varied column formats.
# --- Plotting and summaries ---
# - Expanded plotting (multi-plot, scatter/hist) with richer auto-plot summaries.
# --- Trace visibility ---
# - Added trace logging in responses for full tool/assistant workflow visibility.
# --- Instruction tuning ---
# - Simplified system instructions, improved category targeting, and ignored model limits unless requested.
# 2026-01-09
# --- Category and schema routing ---
# - Treat category/form lookup questions as tool-required to avoid empty responses.
# - Default category lookups to product_categories and skip extra schema fetches unless joins are needed.
# - Apply category_name filters to base category tables when no join is required.
# - Fix datetime column detection and widen analysis intent for "production values".
# - Count-style questions return a single COUNT(*) without day grouping.
# - Resolve unclear terms by probing product/sub-category/form names in order.
# - For category measurement queries, include form/field context and filter out archived sub-categories.
# - Prefer spc_measurements as base table for measurement/production requests.
# --- Response tone and safety ---
# - Return human-friendly replies for single-row lookups and avoid datetime ranges on numeric IDs.
# - Add common-chat handling with safe fallbacks and direct date/time replies.
# - Avoid answering non-time questions with date replies; add safe weather fallback.
# - Ensure final replies follow the user's language (English/Chinese).
# --- Tool routing ---
# - Treat spc/measurement/production/data queries as DB requests (avoid generic chat replies).
# --- Workflow visibility ---
# - Add workflow + query logging in responses and return friendlier error messages.
# --- Analysis summaries ---
# - Provide opinion-focused analysis summaries with sample rows and references.
# 2026-01-13
# --- Update log maintenance ---
# - Refreshed update log date per request.
# - Documented a local uvicorn run hint in requirements.txt.

import ast
import base64
import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import json
from datetime import datetime
import hashlib
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
import requests
from collections import deque, defaultdict

# -------------------------
# Config / setup
# -------------------------
load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
DATABASE_URL = os.getenv("DATABASE_URL")
MAX_SQL_ROWS = int(os.getenv("MAX_SQL_ROWS", "0"))
MAX_TOOL_TURNS = int(os.getenv("MAX_TOOL_TURNS", "10"))
QUERY_RESULTS_DIR = os.getenv("QUERY_RESULTS_DIR", "query_results")
PLOT_RESULTS_DIR = os.getenv("PLOT_RESULTS_DIR", "plot_results")
MAX_SCHEMA_TABLES = int(os.getenv("MAX_SCHEMA_TABLES", "50"))
MAX_SCHEMA_COLUMNS = int(os.getenv("MAX_SCHEMA_COLUMNS", "200"))
MAX_SCHEMA_HOPS = int(os.getenv("MAX_SCHEMA_HOPS", "6"))
SCHEMA_FILE = Path(os.getenv("SCHEMA_FILE", "schema.sql"))


def _env_bool(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_schema_file() -> Dict[str, List[str]]:
    if not SCHEMA_FILE.exists():
        return {}
    try:
        text = SCHEMA_FILE.read_text(encoding="utf-8")
    except Exception:
        return {}

    tables: Dict[str, List[str]] = {}
    current_table: Optional[str] = None
    columns: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("create table"):
            match = re.match(r"create table\s+([^\s(]+)", line, re.IGNORECASE)
            if match:
                table = match.group(1).strip('"').split(".")[-1].strip('"')
                current_table = table
                columns = []
            continue
        if current_table:
            if line.startswith(");"):
                if columns:
                    tables[current_table] = columns
                current_table = None
                columns = []
                continue
            if line.lower().startswith((
                "constraint",
                "primary",
                "foreign",
                "unique",
                "check",
            )):
                continue
            col_match = re.match(r"\"([^\"]+)\"", line)
            if not col_match:
                col_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", line)
            if not col_match:
                continue
            col = col_match.group(1)
            if col not in columns:
                columns.append(col)
    return tables


SAVE_QUERY_RESULTS = _env_bool("SAVE_QUERY_RESULTS", "1")
SAVE_PLOTS = _env_bool("SAVE_PLOTS", "1")
SCHEMA_FILE_TABLES = _load_schema_file()

if not OLLAMA_HOST:
    raise RuntimeError("Missing OLLAMA_HOST in environment/.env")
if not OLLAMA_MODEL:
    raise RuntimeError("Missing OLLAMA_MODEL in environment/.env")
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
    plot_saved_to: Optional[str] = None
    query_saved_to: Optional[str] = None
    trace: Optional[List[Dict[str, Any]]] = None
    plots: Optional[List[Dict[str, Any]]] = None
    workflow: Optional[List[str]] = None
    queries: Optional[List[Dict[str, Any]]] = None


# validate parameters
def _validate_named_params(sql: str, params: Optional[Any]) -> None:
    params_obj: Any = params or {}
    if isinstance(params_obj, str):
        try:
            params_obj = json.loads(params_obj)
        except json.JSONDecodeError:
            params_obj = {}
    if isinstance(params_obj, (list, tuple)):
        # Positional params are validated later during normalization.
        return
    if not isinstance(params_obj, dict):
        params_obj = {}
    names = set(re.findall(r"(?<!:):([A-Za-z_][A-Za-z0-9_]*)", sql))
    names.update(re.findall(r"%\(([A-Za-z_][A-Za-z0-9_]*)\)s", sql))
    missing = names - set(params_obj.keys())
    if missing:
        raise ValueError(f"Missing SQL params: {sorted(missing)}")


SQL_TABLE_RE = re.compile(r"\b(from|join)\s+([A-Za-z_][A-Za-z0-9_.]*)", re.IGNORECASE)
SQL_TABLE_COL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")
SQL_FILTER_COL_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\b\s*(=|<>|!=|>=|<=|>|<|like|ilike|in|between|is)\b",
    re.IGNORECASE,
)
SQL_KEYWORDS = {
    "and",
    "or",
    "not",
    "null",
    "true",
    "false",
    "like",
    "ilike",
    "in",
    "between",
    "is",
    "select",
    "from",
    "join",
    "where",
    "group",
    "order",
    "limit",
}


def _extract_sql_tables(sql: str) -> List[str]:
    tables: List[str] = []
    for _, name in SQL_TABLE_RE.findall(sql or ""):
        base = name.split(".")[-1]
        if base and base not in tables:
            tables.append(base)
    return tables


def _unknown_column_message(table: str, column: str, available: List[str]) -> str:
    preview = ", ".join(available[:20]) + (" ..." if len(available) > 20 else "")
    return f"Unknown column '{column}' in table '{table}'. Available columns: {preview}"


def _record_schema(
    cache: Dict[str, Dict[str, Any]],
    schema: Dict[str, Any],
) -> bool:
    table = schema.get("table")
    columns = schema.get("columns") or []
    if not table or not isinstance(columns, list):
        return False
    names: List[str] = []
    for col in columns:
        if isinstance(col, dict) and col.get("name"):
            names.append(col.get("name"))
        elif isinstance(col, str):
            col_name = col.strip()
            if col_name:
                names.append(col_name)
    if not names:
        return False
    cache[table] = {
        "columns": names,
        "columns_lower": {name.lower() for name in names},
    }
    return True


def _validate_sql_columns(
    sql: str,
    schema_cache: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    tables = _extract_sql_tables(sql)
    if not tables:
        return None
    missing_tables = [t for t in tables if t not in schema_cache]
    if missing_tables:
        return f"Schema required for tables: {', '.join(missing_tables)}. Call describe_table first."
    for table, column in SQL_TABLE_COL_RE.findall(sql or ""):
        base = table.split(".")[-1]
        if base in schema_cache:
            columns_lower = schema_cache[base]["columns_lower"]
            if column.lower() not in columns_lower:
                return _unknown_column_message(
                    base,
                    column,
                    schema_cache[base]["columns"],
                )
    if len(tables) == 1:
        table = tables[0]
        columns_lower = schema_cache[table]["columns_lower"]
        for column, _ in SQL_FILTER_COL_RE.findall(sql or ""):
            col_lower = column.lower()
            if col_lower in SQL_KEYWORDS:
                continue
            if col_lower not in columns_lower:
                return _unknown_column_message(
                    table,
                    column,
                    schema_cache[table]["columns"],
                )
    return None


def _require_args(args: Dict[str, Any], required: List[str], tool_name: str) -> None:
    missing = [k for k in required if k not in args or args[k] in (None, "", [])]
    if missing:
        raise ValueError(f"{tool_name} missing required args: {missing}")

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

def add_hard_limit(sql: str, limit: int) -> str:
    """
    Simple hard limit. For production, prefer:
    - allowlisted views
    - SQL parser + rewrite
    - statement timeout
    """
    cleaned = sql.strip().rstrip(";")
    # If the query already includes LIMIT, wrap it to enforce a hard cap.
    if re.search(r"\blimit\b", cleaned, re.IGNORECASE):
        return f"SELECT * FROM ({cleaned}) AS limited_subquery LIMIT {int(limit)}"
    return f"{cleaned}\nLIMIT {int(limit)}"


def _normalize_sql_and_params(
    sql: str,
    params: Optional[Any],
) -> Tuple[str, Dict[str, Any]]:
    if params is None:
        return sql, {}

    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            return sql, {}

    if isinstance(params, dict):
        if "?" in sql:
            try:
                items = sorted(params.items(), key=lambda kv: int(kv[0]))
            except Exception as exc:
                raise ValueError(
                    "SQL uses '?' placeholders; provide params as a list in order."
                ) from exc
            values = [v for _, v in items]
            return _normalize_sql_and_params(sql, values)
        if re.search(r"%\([A-Za-z_][A-Za-z0-9_]*\)s", sql):
            new_sql = re.sub(r"%\(([A-Za-z_][A-Za-z0-9_]*)\)s", r":\1", sql)
            return new_sql, params
        return sql, params

    if isinstance(params, (list, tuple)):
        if "?" in sql:
            parts = sql.split("?")
            count = len(parts) - 1
            if count != len(params):
                raise ValueError(f"Expected {count} params but got {len(params)}.")
            new_sql = ""
            for i, part in enumerate(parts):
                new_sql += part
                if i < count:
                    new_sql += f":p{i+1}"
            return new_sql, {f"p{i+1}": params[i] for i in range(len(params))}
        if re.search(r"\$\d+", sql):
            def repl(match: re.Match[str]) -> str:
                idx = int(match.group(1))
                if idx <= 0 or idx > len(params):
                    raise ValueError("Positional params count mismatch.")
                return f":p{idx}"
            new_sql = re.sub(r"\$(\d+)", repl, sql)
            return new_sql, {f"p{i+1}": params[i] for i in range(len(params))}
        if len(params) == 0:
            return sql, {}
        raise ValueError("Positional params provided but SQL has no placeholders.")

    raise ValueError("Unsupported params type; use object or list.")


def _resolve_table_name(table: str, tables: List[str]) -> str:
    candidate = table.split(".")[-1]
    if candidate in tables:
        return candidate
    lowered = candidate.lower()
    for t in tables:
        if t.lower() == lowered:
            return t
    raise ValueError(f"Unknown table: {table}")


def get_schema(
    table: Optional[str] = None,
    max_tables: Optional[int] = None,
) -> Dict[str, Any]:
    inspector = inspect(engine)
    tables = sorted(inspector.get_table_names())

    if table:
        table_name = _resolve_table_name(table, tables)
        columns = inspector.get_columns(table_name) or []
        pk = inspector.get_pk_constraint(table_name) or {}
        fks = inspector.get_foreign_keys(table_name) or []

        col_items = []
        for col in columns[:MAX_SCHEMA_COLUMNS]:
            col_items.append({
                "name": col.get("name"),
                "type": str(col.get("type")),
                "nullable": bool(col.get("nullable", False)),
            })

        fk_items = []
        for fk in fks:
            referred_table = fk.get("referred_table")
            constrained_columns = fk.get("constrained_columns") or []
            referred_columns = fk.get("referred_columns") or []
            if not referred_table or not constrained_columns or not referred_columns:
                continue
            fk_items.append({
                "name": fk.get("name"),
                "constrained_columns": constrained_columns,
                "referred_table": referred_table,
                "referred_columns": referred_columns,
            })

        return {
            "table": table_name,
            "columns": col_items,
            "primary_key": pk.get("constrained_columns") or [],
            "foreign_keys": fk_items,
            "column_truncated": len(columns) > MAX_SCHEMA_COLUMNS,
        }

    limit = max_tables if max_tables is not None else MAX_SCHEMA_TABLES
    limit = max(1, int(limit))
    truncated = len(tables) > limit
    return {
        "tables": tables[:limit],
        "table_count": len(tables),
        "truncated": truncated,
    }


def describe_table(table: str) -> Dict[str, Any]:
    # Validate table name to avoid injection.
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table or ""):
        raise ValueError("Invalid table name")
    return get_schema(table=table)


def find_join_path(
    from_table: str,
    to_table: str,
    max_hops: Optional[int] = None,
) -> Dict[str, Any]:
    inspector = inspect(engine)
    tables = sorted(inspector.get_table_names())
    src = _resolve_table_name(from_table, tables)
    dst = _resolve_table_name(to_table, tables)

    edges = []
    for table in tables:
        for fk in inspector.get_foreign_keys(table) or []:
            referred_table = fk.get("referred_table")
            constrained_columns = fk.get("constrained_columns") or []
            referred_columns = fk.get("referred_columns") or []
            if not referred_table or not constrained_columns or not referred_columns:
                continue
            edges.append({
                "from_table": table,
                "to_table": referred_table,
                "from_columns": constrained_columns,
                "to_columns": referred_columns,
            })

    adj: Dict[str, List[Tuple[Dict[str, Any], str]]] = defaultdict(list)
    for edge in edges:
        adj[edge["from_table"]].append((edge, "forward"))
        adj[edge["to_table"]].append((edge, "reverse"))

    max_steps = int(max_hops) if max_hops is not None else MAX_SCHEMA_HOPS
    max_steps = max(1, max_steps)

    queue = deque([(src, [])])
    visited = {src}

    while queue:
        current, path = queue.popleft()
        if len(path) >= max_steps:
            continue
        for edge, direction in adj.get(current, []):
            if direction == "forward":
                next_table = edge["to_table"]
                left_table = edge["from_table"]
                left_cols = edge["from_columns"]
                right_table = edge["to_table"]
                right_cols = edge["to_columns"]
            else:
                next_table = edge["from_table"]
                left_table = edge["to_table"]
                left_cols = edge["to_columns"]
                right_table = edge["from_table"]
                right_cols = edge["from_columns"]

            if next_table in visited:
                continue

            step = {
                "left_table": left_table,
                "left_columns": left_cols,
                "right_table": right_table,
                "right_columns": right_cols,
                "direction": direction,
            }
            new_path = path + [step]
            if next_table == dst:
                join_clauses = []
                for item in new_path:
                    conditions = [
                        f"{item['left_table']}.{l} = {item['right_table']}.{r}"
                        for l, r in zip(item["left_columns"], item["right_columns"])
                    ]
                    join_clauses.append(
                        f"JOIN {item['right_table']} ON " + " AND ".join(conditions)
                    )
                return {
                    "from_table": src,
                    "to_table": dst,
                    "hops": len(new_path),
                    "path": new_path,
                    "join_clauses": join_clauses,
                }

            visited.add(next_table)
            queue.append((next_table, new_path))

    return {
        "from_table": src,
        "to_table": dst,
        "hops": 0,
        "path": [],
        "join_clauses": [],
        "error": f"No join path found within {max_steps} hops.",
    }


def _save_query_result(
    sql: str,
    params: Optional[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    row_count: int,
    truncated: bool,
) -> Optional[str]:
    try:
        base = Path(QUERY_RESULTS_DIR)
        base.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        params_blob = json.dumps(params or {}, sort_keys=True, default=str)
        digest = hashlib.sha1(f"{sql}|{params_blob}".encode("utf-8")).hexdigest()[:8]
        filename = f"{ts}_{digest}.json"
        path = base / filename
        payload = {
            "saved_at_utc": ts,
            "sql": sql,
            "params": params or {},
            "row_count": row_count,
            "truncated": truncated,
            "rows": rows,
        }
        path.write_text(json.dumps(payload, ensure_ascii=True, default=str, indent=2))
        return str(path)
    except Exception:
        return None


def _save_plot_bytes(png_bytes: bytes) -> Optional[str]:
    try:
        base = Path(PLOT_RESULTS_DIR)
        base.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        digest = hashlib.sha1(png_bytes).hexdigest()[:8]
        filename = f"{ts}_{digest}.png"
        path = base / filename
        path.write_bytes(png_bytes)
        return str(path)
    except Exception:
        return None


# -------------------------
# Tool implementations
# -------------------------
def run_sql(
    sql: str,
    params: Optional[Any] = None,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    sql, params = _normalize_sql_and_params(sql, params)
    enforce_read_only_sql(sql)
    requested: Optional[int] = None
    if max_rows is not None:
        try:
            requested = int(max_rows)
        except (TypeError, ValueError):
            requested = None
        if requested is not None and requested <= 0:
            requested = None
    if MAX_SQL_ROWS > 0:
        if requested is None:
            limit = MAX_SQL_ROWS
        else:
            limit = max(1, min(requested, MAX_SQL_ROWS))
    else:
        limit = max(1, requested) if requested else None
    safe_sql = add_hard_limit(sql, limit=limit) if limit else sql

    with engine.connect() as conn:
        # Optional: set statement timeout for Postgres
        # conn.execute(text("SET LOCAL statement_timeout = 2000"))
        result = conn.execute(text(safe_sql), params or {})
        rows = [dict(r._mapping) for r in result.fetchall()]

    truncated = True if limit and len(rows) == limit else False
    saved_to: Optional[str] = None
    if SAVE_QUERY_RESULTS:
        saved_to = _save_query_result(sql, params, rows, len(rows), truncated)

    response: Dict[str, Any] = {
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
    }
    if saved_to:
        response["saved_to"] = saved_to
    return response


def make_plot(
    title: str,
    kind: str,
    x_key: str,
    y_key: str,
    data: List[Dict[str, Any]],
    legend_label: Optional[str] = None,
) -> Dict[str, Any]:
    if kind not in ("line", "bar", "scatter", "hist"):
        raise ValueError("kind must be 'line', 'bar', 'scatter', or 'hist'")
    if not data:
        raise ValueError("No data provided for plot.")

    df = pd.DataFrame(data)
    if kind == "hist":
        if x_key not in df.columns:
            raise ValueError(f"Missing x_key in data. Have: {list(df.columns)}")
    else:
        if x_key not in df.columns or y_key not in df.columns:
            raise ValueError(f"Missing x_key or y_key in data. Have: {list(df.columns)}")

    if kind != "hist":
        # Try to coerce x-axis to datetime if possible
        try:
            df[x_key] = pd.to_datetime(df[x_key])
            df = df.sort_values(x_key)
        except Exception:
            pass

    plt.figure()
    label = legend_label or y_key
    if kind == "line":
        plt.plot(df[x_key], df[y_key], label=label)
    elif kind == "bar":
        plt.bar(df[x_key], df[y_key], label=label)
    elif kind == "scatter":
        plt.scatter(df[x_key], df[y_key], label=label)
    else:
        series = pd.to_numeric(df[x_key], errors="coerce").dropna()
        if series.empty:
            raise ValueError(f"No numeric data for histogram in {x_key}.")
        bins = min(30, max(5, int(len(series) ** 0.5)))
        plt.hist(series, bins=bins, label=label)

    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel("frequency" if kind == "hist" else y_key)
    if label:
        plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)

    png_bytes = buf.read()
    saved_to: Optional[str] = None
    if SAVE_PLOTS:
        saved_to = _save_plot_bytes(png_bytes)
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    result: Dict[str, Any] = {"png_base64": b64}
    if saved_to:
        result["saved_to"] = saved_to
    return result


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
                    "max_rows": {
                        "type": "integer",
                        "description": "Optional requested row cap (server enforces a hard maximum).",
                    },
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": "Get table list or schema details (columns, primary key, foreign keys).",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Optional table name to describe. If omitted, returns table list.",
                    },
                    "max_tables": {
                        "type": "integer",
                        "description": "Optional cap on number of tables returned when listing.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": "Describe a single table (columns, primary key, foreign keys).",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                },
                "required": ["table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_join_path",
            "description": "Find a foreign-key join path between two tables.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_table": {"type": "string"},
                    "to_table": {"type": "string"},
                    "max_hops": {
                        "type": "integer",
                        "description": "Optional max number of hops to search.",
                    },
                },
                "required": ["from_table", "to_table"],
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
                    "kind": {"type": "string", "enum": ["line", "bar", "scatter", "hist"]},
                    "x_key": {"type": "string"},
                    "y_key": {"type": "string"},
                    "data": {"type": "array", "items": {"type": "object"}},
                    "legend_label": {
                        "type": "string",
                        "description": "Optional label to show in the plot legend.",
                    },
                },
                "required": ["title", "kind", "x_key", "y_key", "data"],
            },
        },
    },
]

TOOL_NAME_SET = {tool["function"]["name"] for tool in TOOLS}

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


def _coerce_args(args: Any) -> Dict[str, Any]:
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {}
    if isinstance(args, dict):
        return args
    return {}


def _normalize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    if not tool_calls:
        return []
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]
    if not isinstance(tool_calls, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if name not in TOOL_NAME_SET:
            continue
        args = _coerce_args(fn.get("arguments") or {})
        normalized.append({"function": {"name": name, "arguments": args}})
    return normalized


def _extract_json_objects(text: str) -> List[Any]:
    decoder = json.JSONDecoder()
    objects: List[Any] = []
    idx = 0
    while True:
        match = re.search(r"{", text[idx:])
        if not match:
            break
        start = idx + match.start()
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            idx = start + 1
            continue
        objects.append(obj)
        idx = start + end
    return objects


def _tool_calls_from_obj(obj: Any) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        name = obj.get("name")
        params = obj.get("parameters", obj.get("arguments"))
        if name in TOOL_NAME_SET:
            args = _coerce_args(params)
            calls.append({"function": {"name": name, "arguments": args}})
    elif isinstance(obj, list):
        for item in obj:
            calls.extend(_tool_calls_from_obj(item))
    return calls


def _ast_literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _python_call_to_args(name: str, call: ast.Call) -> Optional[Dict[str, Any]]:
    positional = [_ast_literal(arg) for arg in call.args]
    keywords = {
        kw.arg: _ast_literal(kw.value)
        for kw in call.keywords
        if kw.arg
    }

    def pick_arg(idx: int, key: str) -> Any:
        if key in keywords and keywords[key] is not None:
            return keywords[key]
        if len(positional) > idx:
            return positional[idx]
        return None

    if name == "run_sql":
        sql_value = pick_arg(0, "sql")
        if not isinstance(sql_value, str):
            return None
        args: Dict[str, Any] = {"sql": sql_value}
        params_value = pick_arg(1, "params")
        if params_value is not None:
            args["params"] = params_value
        max_rows = pick_arg(2, "max_rows")
        if max_rows is not None:
            args["max_rows"] = max_rows
        return args

    if name == "get_schema":
        args: Dict[str, Any] = {}
        table_value = pick_arg(0, "table")
        if isinstance(table_value, str) and table_value:
            args["table"] = table_value
        max_tables = pick_arg(1, "max_tables")
        if max_tables is not None:
            args["max_tables"] = max_tables
        return args

    if name == "describe_table":
        table_value = pick_arg(0, "table")
        if not isinstance(table_value, str) or not table_value:
            return None
        return {"table": table_value}

    if name == "find_join_path":
        from_table = pick_arg(0, "from_table")
        to_table = pick_arg(1, "to_table")
        if not isinstance(from_table, str) or not isinstance(to_table, str):
            return None
        args = {"from_table": from_table, "to_table": to_table}
        max_hops = pick_arg(2, "max_hops")
        if max_hops is not None:
            args["max_hops"] = max_hops
        return args

    if name == "make_plot":
        keys = ["title", "kind", "x_key", "y_key", "data", "legend_label"]
        args: Dict[str, Any] = {}
        for idx, key in enumerate(keys):
            value = pick_arg(idx, key)
            if value is None:
                continue
            args[key] = value
        if not all(k in args for k in ("title", "kind", "x_key", "y_key", "data")):
            return None
        return args

    return None


def _python_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    if "<|python_tag|>" in text:
        content = text.replace("<|python_tag|>", "").strip()
    else:
        if not any(f"{name}(" in text for name in TOOL_NAME_SET):
            return []
        content = text

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    calls: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def handle_call(node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            return
        if name not in TOOL_NAME_SET:
            return
        args = _python_call_to_args(name, node)
        if args is None:
            return
        try:
            signature = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
        except TypeError:
            signature = f"{name}:{args}"
        if signature in seen:
            return
        seen.add(signature)
        calls.append({"function": {"name": name, "arguments": args}})

    for stmt in tree.body:
        if isinstance(stmt, ast.Expr):
            value = stmt.value
        elif isinstance(stmt, ast.Assign):
            value = stmt.value
        else:
            continue
        if isinstance(value, ast.Call):
            handle_call(value)
        for node in ast.walk(value):
            if isinstance(node, ast.Call):
                handle_call(node)

    return calls


def get_tool_calls(assistant_msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Ollama returns tool calls under message.tool_calls.
    normalized = _normalize_tool_calls(assistant_msg.get("tool_calls"))
    if normalized:
        return normalized

    # Some models still use function_call.
    function_call = assistant_msg.get("function_call")
    if isinstance(function_call, dict):
        name = function_call.get("name")
        if name in TOOL_NAME_SET:
            args = _coerce_args(function_call.get("arguments") or {})
            return [{"function": {"name": name, "arguments": args}}]

    # Fallback: parse inline JSON or python-tag tool calls from content.
    content = assistant_msg.get("content") or ""
    if not content:
        return []

    calls: List[Dict[str, Any]] = []
    for obj in _extract_json_objects(content):
        calls.extend(_tool_calls_from_obj(obj))
    if calls:
        return calls

    python_calls = _python_tool_calls_from_text(content)
    if python_calls:
        return python_calls

    return []


def _tool_signature(tool_calls: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name")
        args = fn.get("arguments") or {}
        try:
            args_blob = json.dumps(args, sort_keys=True, default=str)
        except TypeError:
            args_blob = str(args)
        parts.append(f"{name}:{args_blob}")
    return "|".join(parts)


def _safe_str(value: Any, max_len: int = 40) -> str:
    try:
        text = str(value)
    except Exception:
        text = "<unprintable>"
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _trace_event(
    trace: List[Dict[str, Any]],
    event_type: str,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    event = {"ts": _utc_timestamp(), "type": event_type}
    if payload:
        event.update(payload)
    trace.append(event)


def _trace_tool_args(name: Optional[str], args: Any) -> Dict[str, Any]:
    if not isinstance(args, dict):
        return {"raw_args": _safe_str(args, max_len=200)}
    sanitized = dict(args)
    if name == "make_plot":
        data = sanitized.pop("data", None)
        if isinstance(data, list):
            sanitized["data_rows"] = len(data)
        elif data is not None:
            sanitized["data_rows"] = 1
    return sanitized


def _trace_schema_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if "table" in result:
        columns: List[str] = []
        for col in result.get("columns", []) or []:
            if isinstance(col, dict) and col.get("name"):
                columns.append(col.get("name"))
            elif isinstance(col, str):
                col_name = col.strip()
                if col_name:
                    columns.append(col_name)
        return {
            "table": result.get("table"),
            "column_count": len(columns),
            "columns": columns[:50],
        }
    tables = result.get("tables", []) or []
    return {
        "table_count": result.get("table_count"),
        "truncated": result.get("truncated"),
        "tables": tables[:50],
    }


def _wants_plot(message: str) -> bool:
    return bool(re.search(r"\b(plot|graph|chart|visuali[sz]e)\b", message, re.IGNORECASE))


def _needs_opinion(message: str) -> bool:
    return bool(re.search(
        r"\b(opinion|insight|analysis|analy[sz]e|trend|interpret|explain|why)\b",
        message,
        re.IGNORECASE,
    ))


def _is_analysis_request(message: str) -> bool:
    return bool(re.search(
        r"\b(analy[sz]e|analysis|trend|insight|opinion|interpret|explain|pattern|correlation|variance|distribution|production|values?|measurements?)\b",
        message,
        re.IGNORECASE,
    ))


def _is_measurement_request(message: str) -> bool:
    return bool(re.search(
        r"\b(spc|measurement|measurements|value|values|production|inspect|trend|condition)\b",
        message,
        re.IGNORECASE,
    ))


def _is_count_request(message: str) -> bool:
    return bool(re.search(
        r"\b(how many|count|number of|total)\b",
        message,
        re.IGNORECASE,
    ))


def _requires_tool_call(message: str) -> bool:
    if re.search(r"\d{4}-\d{2}-\d{2}", message):
        return True
    return bool(re.search(
        r"\b(table|rows?|columns?|schema|database|sql|analy|analysis|plot|graph|chart|trend|average|mean|median|count|sum|min|max|category|category_name|category_id|sub_category|form|form_name|form_id|spc|measurement|measurements|value|values|production|inspect|inspection|data|records?)\b",
        message,
        re.IGNORECASE,
    ))


def _user_requested_limit(message: str) -> bool:
    if not message:
        return False
    return bool(re.search(
        r"\blimit\b|\bsample\b|\bpreview\b|\btop\s+\d+\b|\bfirst\s+\d+\b|\bhead\s+\d+\b|\brows?\s+\d+\b",
        message,
        re.IGNORECASE,
    ))


def _extract_sql_payload(text: str) -> Tuple[Optional[str], Optional[Any]]:
    if not text:
        return None, None
    for obj in _extract_json_objects(text):
        if not isinstance(obj, dict):
            continue
        key_map = {
            str(key).lower(): key
            for key in obj.keys()
            if isinstance(key, str)
        }
        sql_key = key_map.get("sql")
        if not sql_key:
            continue
        sql_value = obj.get(sql_key)
        if not isinstance(sql_value, str) or not sql_value.strip():
            continue
        params_value = None
        params_key = key_map.get("params") or key_map.get("parameters")
        if params_key:
            params_value = obj.get(params_key)
        return sql_value.strip(), params_value
    return None, None


def _extract_sql_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"```sql\s+([\s\S]+?)```", text, re.IGNORECASE)
    if match:
        sql = match.group(1).strip()
    else:
        match = re.search(r"\bselect\b[\s\S]+", text, re.IGNORECASE)
        if not match:
            return None
        sql = match.group(0).strip()
    sql = sql.split(";")[0].strip()
    return sql or None


def _schema_summary(schema: Dict[str, Any]) -> str:
    table = schema.get("table") or "unknown"
    columns = [col.get("name") for col in schema.get("columns", []) if col.get("name")]
    columns_line = ", ".join(columns[:20]) + (" ..." if len(columns) > 20 else "")
    pk = schema.get("primary_key") or []
    pk_line = ", ".join(pk) if pk else "none"
    fk_items = []
    for fk in schema.get("foreign_keys", [])[:5]:
        left_cols = fk.get("constrained_columns") or []
        right_cols = fk.get("referred_columns") or []
        pairs = ", ".join(f"{l}->{r}" for l, r in zip(left_cols, right_cols))
        fk_items.append(f"{pairs} -> {fk.get('referred_table')}")
    lines = [
        f"Table {table}: columns [{columns_line}]",
        f"PK: {pk_line}",
    ]
    if fk_items:
        lines.append("FKs: " + "; ".join(fk_items))
    return "\n".join(lines)


def _extract_dates(message: str) -> List[str]:
    dates = re.findall(r"\d{4}[/-]\d{2}[/-]\d{2}", message)
    cleaned = [d.replace("/", "-") for d in dates]
    return cleaned


def _extract_table_from_message(message: str, tables: List[str]) -> Optional[str]:
    match = re.search(r"\btable\s+([A-Za-z_][A-Za-z0-9_]*)\b", message, re.IGNORECASE)
    if match:
        try:
            return _resolve_table_name(match.group(1), tables)
        except ValueError:
            return None
    return None


def _message_mentions_table(message: str, table: str) -> bool:
    if not message or not table:
        return False
    return bool(re.search(rf"\b{re.escape(table)}\b", message, re.IGNORECASE))


def _schema_hints_from_file(message: str) -> str:
    if not SCHEMA_FILE_TABLES:
        return ""
    tables: List[str] = []
    if (
        _message_mentions_table(message, "spc_measurements")
        or _is_analysis_request(message)
    ) and "spc_measurements" in SCHEMA_FILE_TABLES:
        tables.append("spc_measurements")
        if "form_fields" in SCHEMA_FILE_TABLES:
            tables.append("form_fields")
    if re.search(r"\bcategory(_name)?\b", message, re.IGNORECASE):
        if "product_categories" in SCHEMA_FILE_TABLES:
            tables.append("product_categories")
        if "sub_categories" in SCHEMA_FILE_TABLES:
            tables.append("sub_categories")
        if "forms" in SCHEMA_FILE_TABLES:
            tables.append("forms")
    if (
        _extract_form_names(message)
        or re.search(r"\bform_name\b|\bform name\b", message, re.IGNORECASE)
    ) and "forms" in SCHEMA_FILE_TABLES:
        tables.append("forms")
    for table in SCHEMA_FILE_TABLES:
        if table in tables:
            continue
        if _message_mentions_table(message, table):
            tables.append(table)
    if not tables:
        return ""
    tables = list(dict.fromkeys(tables))
    lines: List[str] = []
    for table in tables[:4]:
        cols = SCHEMA_FILE_TABLES.get(table, [])
        if not cols:
            continue
        cols_line = ", ".join(cols[:20]) + (" ..." if len(cols) > 20 else "")
        lines.append(f"{table}: {cols_line}")
    if "spc_measurements" in tables and "form_fields" in tables:
        lines.append(
            "Join: spc_measurements.form_id = form_fields.form_id "
            "AND spc_measurements.field_index = form_fields.field_index"
        )
    if "form_fields" in tables and "forms" in tables:
        lines.append("Join: form_fields.form_id = forms.form_id")
    if "spc_measurements" in tables and "forms" in tables:
        lines.append("Join: spc_measurements.form_id = forms.form_id")
    if (
        "forms" in tables
        and "sub_categories" in tables
        and "product_categories" in tables
    ):
        lines.append(
            "Join: forms.sub_category_id = sub_categories.sub_category_id "
            "AND sub_categories.category_id = product_categories.category_id"
        )
        lines.append("Preferred path: spc_measurements -> form_fields -> forms -> sub_categories -> product_categories")
    if not lines:
        return ""
    return "Schema hints (from schema.sql):\n" + "\n".join(lines)


def _pick_datetime_column(schema: Dict[str, Any]) -> Optional[str]:
    columns = schema.get("columns", []) or []
    best = None
    best_score = -1
    for col in columns:
        name = str(col.get("name", ""))
        if not name:
            continue
        lower = name.lower()
        tokens = [t for t in re.split(r"[^a-z0-9]+", lower) if t]
        score = 0
        if "inspect_time" in lower:
            score += 100
        if "inspection" in lower:
            score += 90
        if "timestamp" in lower:
            score += 80
        if any(tok in ("timestamp", "datetime") for tok in tokens):
            score += 80
        if lower.endswith("_time") or lower.endswith("_date"):
            score += 70
        if "created" in tokens or "updated" in tokens:
            score += 60
        if any(tok in ("date", "time") for tok in tokens):
            score += 50
        col_type = str(col.get("type", "")).lower()
        if "timestamp" in col_type:
            score += 40
        if "date" in col_type:
            score += 30
        if score > best_score:
            best_score = score
            best = name
    if best_score <= 0:
        return None
    return best


def _pick_category_column(schema: Dict[str, Any]) -> Optional[str]:
    columns = schema.get("columns", []) or []
    preferred = ("category_name", "name", "category", "type")
    for pref in preferred:
        for col in columns:
            name = str(col.get("name", ""))
            if name.lower() == pref:
                return name
    for col in columns:
        name = str(col.get("name", ""))
        col_type = str(col.get("type", "")).lower()
        if "char" in col_type or "text" in col_type:
            return name
    return None


def _is_numeric_type(type_name: str) -> bool:
    lower = type_name.lower()
    return any(token in lower for token in ("int", "float", "double", "numeric", "decimal", "real"))


def _pick_metric_column(schema: Dict[str, Any]) -> Optional[str]:
    columns = schema.get("columns", []) or []
    preferred_names = ("value", "measurement", "measure", "reading", "result", "score", "amount")
    for pref in preferred_names:
        for col in columns:
            name = str(col.get("name", ""))
            if name.lower() == pref:
                return name
    for col in columns:
        name = str(col.get("name", ""))
        if not name:
            continue
        lower = name.lower()
        if any(pref in lower for pref in preferred_names):
            if _is_numeric_type(str(col.get("type", ""))):
                return name
    for col in columns:
        name = str(col.get("name", ""))
        if not name:
            continue
        if "id" in name.lower():
            continue
        if _is_numeric_type(str(col.get("type", ""))):
            return name
    return None


def _extract_category_value(message: str) -> Optional[str]:
    patterns = [
        r"category_name\s*=\s*['\"]([^'\"]+)['\"]",
        r"category name\s*=\s*['\"]([^'\"]+)['\"]",
        r"product_categories\s*=\s*['\"]([^'\"]+)['\"]",
        r"category\s*=\s*['\"]([^'\"]+)['\"]",
        r"category_name\s+['\"]([^'\"]+)['\"]",
        r"category name\s+['\"]([^'\"]+)['\"]",
        r"category\s+['\"]([^'\"]+)['\"]",
        r"\bcategory(?:_name| name)?\s+([A-Za-z0-9][A-Za-z0-9 _-]{1,40}?)(?:\s+(?:during|from|to|between|for|in|on|under)\b|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _extract_label_candidates(message: str) -> List[str]:
    if not message:
        return []
    candidates: List[str] = []
    for match in re.finditer(r"[\"']([^\"']+)[\"']", message):
        value = match.group(1).strip()
        if value:
            candidates.append(value)
    line_match = re.search(
        r"\b([A-Za-z0-9][A-Za-z0-9 _-]{1,40})\s+line\b",
        message,
        re.IGNORECASE,
    )
    if line_match:
        candidates.append(line_match.group(1).strip())
    prep_match = re.search(
        r"\b(?:in|under|for|within|on)\s+([A-Za-z0-9][A-Za-z0-9 _-]{1,40})",
        message,
        re.IGNORECASE,
    )
    if prep_match:
        candidates.append(prep_match.group(1).strip())
    stopwords = {
        "today", "yesterday", "tomorrow", "this", "last", "next",
        "month", "week", "year", "during", "from", "to", "between",
        "for", "in", "on", "at", "of", "and",
    }
    cleaned: List[str] = []
    for candidate in candidates:
        tokens = candidate.split()
        while tokens and tokens[-1].lower() in stopwords:
            tokens.pop()
        value = " ".join(tokens).strip()
        if value:
            cleaned.append(value)
        if value.lower().endswith(" line"):
            cleaned.append(value[:-5].strip())
    return list(dict.fromkeys(cleaned))


def _resolve_label_target(
    candidates: List[str],
    tables: List[str],
    query_log: Optional[List[Dict[str, Any]]] = None,
    workflow: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    checks = [
        ("product_categories", "category_name"),
        ("sub_categories", "sub_category_name"),
        ("forms", "form_name"),
    ]
    for table, column in checks:
        if table not in tables:
            continue
        for value in candidates:
            sql = f"SELECT COUNT(*) AS match_count FROM {table} WHERE {column} ILIKE :pattern"
            params = {"pattern": f"%{value}%"}
            try:
                result = run_sql(sql=sql, params=params, max_rows=None)
            except Exception:
                result = None
            if query_log is not None and result is not None:
                _log_query(query_log, sql, params, "resolver", result=result)
            if workflow is not None:
                _workflow_add(workflow, f"Resolver checked {table}.{column} for '{value}'.")
            if result and result.get("rows"):
                match_count = result["rows"][0].get("match_count")
                try:
                    if int(match_count) > 0:
                        return {
                            "table": table,
                            "column": column,
                            "value": value,
                        }
                except (TypeError, ValueError):
                    continue
    return None


def _extract_form_names(message: str) -> List[str]:
    names: List[str] = []
    category_value = _extract_category_value(message)
    for match in re.finditer(r"[\"']([^\"']+)[\"']", message):
        value = match.group(1).strip()
        if not value:
            continue
        if category_value and value.lower() == category_value.lower():
            continue
        window = message[max(0, match.start() - 40):match.end() + 40].lower()
        if re.search(r"\bcategory(_name)?\b|\bsub_category\b", window):
            continue
        if re.search(r"\bform(_name)?s?\b", window):
            names.append(value)
    if names:
        return list(dict.fromkeys(names))[:5]

    if re.search(r"\bform_name\b|\bform name\b", message, re.IGNORECASE):
        for match in re.finditer(r"[\"']([^\"']+)[\"']", message):
            value = match.group(1).strip()
            if not value:
                continue
            if category_value and value.lower() == category_value.lower():
                continue
            names.append(value)
        if names:
            return list(dict.fromkeys(names))[:5]

    match = re.search(
        r"\bforms?\s+(.+?)\s+and\s+(.+?)(?:\s+both|\s+are|\s+from|\s+between|\s+for|\s+on|\s+in|\s*$)",
        message,
        re.IGNORECASE,
    )
    if match:
        for group in match.groups():
            value = group.strip().strip(",.")
            if not value:
                continue
            if category_value and value.lower() == category_value.lower():
                continue
            names.append(value)
    return list(dict.fromkeys(names))[:5]


def _auto_query_from_message(
    message: str,
    query_log: Optional[List[Dict[str, Any]]] = None,
    workflow: Optional[List[str]] = None,
    user_lang: str = "en",
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    if not tables:
        return None, None, "No tables found in database."

    analysis_request = _is_analysis_request(message)
    measurement_request = _is_measurement_request(message)
    category_value = _extract_category_value(message)
    form_names = _extract_form_names(message)
    mentions_category = bool(re.search(r"\bcategory(_name|_id)?\b", message, re.IGNORECASE))
    mentions_sub_category = bool(re.search(r"\bsub[_ ]?category(_name|_id)?\b", message, re.IGNORECASE))
    mentions_form = bool(re.search(r"\bform(_name|_id)?s?\b", message, re.IGNORECASE))
    resolved_target = None
    if not category_value and not form_names:
        candidates = _extract_label_candidates(message)
        if candidates:
            resolved_target = _resolve_label_target(candidates, tables, query_log, workflow)
            if resolved_target:
                if resolved_target["table"] == "forms":
                    form_names = [resolved_target["value"]]
                else:
                    category_value = resolved_target["value"]
                    mentions_category = True
            else:
                return None, None, _t(user_lang, "label_no_match", value=candidates[0])
    base_table = _extract_table_from_message(message, tables)
    candidates = _table_candidates(message)
    if measurement_request and "spc_measurements" in tables:
        base_table = "spc_measurements"
    elif base_table is None and candidates:
        base_table = candidates[0]
    elif base_table is None and analysis_request and "spc_measurements" in tables:
        base_table = "spc_measurements"
    if base_table is None and mentions_sub_category and "sub_categories" in tables:
        base_table = "sub_categories"
    if base_table is None and mentions_form and "forms" in tables:
        base_table = "forms"
    if base_table is None and (category_value or mentions_category) and "product_categories" in tables:
        base_table = "product_categories"
    if base_table is None and len(tables) == 1:
        base_table = tables[0]
    if base_table is None:
        return None, None, "Could not identify a base table from the request."

    category_tables = [t for t in tables if "category" in t.lower()]
    target_table = None
    if resolved_target:
        target_table = resolved_target["table"]
    elif candidates:
        for t in candidates:
            if t != base_table:
                target_table = t
                break
    if target_table is None and category_tables:
        target_table = category_tables[0] if category_tables[0] != base_table else None
    if category_value and mentions_sub_category and "sub_categories" in tables:
        target_table = "sub_categories"
    if (
        category_value
        and mentions_category
        and not mentions_sub_category
        and "product_categories" in tables
    ):
        target_table = "product_categories"

    form_table: Optional[str] = None
    if form_names and "forms" in tables:
        form_table = "forms"
        if base_table.lower() == "forms":
            form_table = base_table
        if target_table is None:
            target_table = "forms"

    join_clauses: List[str] = []
    if target_table and target_table != base_table:
        join = find_join_path(base_table, target_table, max_hops=MAX_SCHEMA_HOPS)
        join_clauses = join.get("join_clauses") or []
        if not join_clauses and "error" in join:
            return None, None, f"No join path found between {base_table} and {target_table}."

    base_schema = get_schema(table=base_table)
    dt_col = _pick_datetime_column(base_schema)
    dates = _extract_dates(message)
    if dates and not dt_col:
        return None, None, f"No datetime column found in {base_table} for date filter."

    count_request = _is_count_request(message)
    prefer_raw = analysis_request and base_table.lower() == "spc_measurements"
    metric_col = _pick_metric_column(base_schema) if prefer_raw else None

    join_tables = set()
    for clause in join_clauses:
        match = re.search(r"\bJOIN\s+([A-Za-z_][A-Za-z0-9_]*)\b", clause, re.IGNORECASE)
        if match:
            join_tables.add(match.group(1))

    base_columns = {
        col.get("name")
        for col in base_schema.get("columns", [])
        if isinstance(col, dict) and col.get("name")
    }
    forms_schema = get_schema(table="forms") if "forms" in join_tables else None
    form_fields_schema = get_schema(table="form_fields") if "form_fields" in join_tables else None
    sub_category_schema = get_schema(table="sub_categories") if "sub_categories" in join_tables else None

    category_col = None
    category_table = None
    if category_value:
        if target_table:
            target_schema = get_schema(table=target_table)
            category_col = _pick_category_column(target_schema)
            category_table = target_table
        elif base_table in category_tables:
            category_col = _pick_category_column(base_schema)
            category_table = base_table

    select_cols = []
    params: Dict[str, Any] = {}
    where_clauses = []

    if dt_col:
        if len(dates) >= 2:
            start_date, end_date = dates[0], dates[1]
        else:
            start_date = end_date = dates[0] if dates else None
        if start_date:
            params["start_ts"] = f"{start_date} 00:00:00+00"
            params["end_ts"] = f"{end_date} 23:59:59+00"
            where_clauses.append(f"{base_table}.{dt_col} >= :start_ts")
            where_clauses.append(f"{base_table}.{dt_col} <= :end_ts")

    if category_col and category_value and category_table:
        params["category_value"] = category_value
        where_clauses.append(f"{category_table}.{category_col} = :category_value")
    if form_table and form_names:
        placeholders = []
        for idx, name in enumerate(form_names):
            key = f"form_name_{idx + 1}"
            params[key] = name
            placeholders.append(f":{key}")
        if placeholders:
            where_clauses.append(f"{form_table}.form_name IN ({', '.join(placeholders)})")
    if sub_category_schema:
        sub_cols = {
            col.get("name")
            for col in sub_category_schema.get("columns", [])
            if isinstance(col, dict) and col.get("name")
        }
        if "sub_category_name" in sub_cols and not re.search(r"封存|archived", message, re.IGNORECASE):
            where_clauses.append("sub_categories.sub_category_name <> '封存'")

    if prefer_raw:
        if dt_col:
            select_cols.append(f"{base_table}.{dt_col}")
        for col in ("form_id", "field_index", "sample_index", "line_name"):
            if col in base_columns:
                select_cols.append(f"{base_table}.{col}")
        if metric_col:
            select_cols.append(f"{base_table}.{metric_col}")
        if forms_schema:
            forms_cols = {
                col.get("name")
                for col in forms_schema.get("columns", [])
                if isinstance(col, dict) and col.get("name")
            }
            if "form_name" in forms_cols:
                select_cols.append("forms.form_name")
        if form_fields_schema:
            field_cols = {
                col.get("name")
                for col in form_fields_schema.get("columns", [])
                if isinstance(col, dict) and col.get("name")
            }
            if "product_characteristic" in field_cols:
                select_cols.append("form_fields.product_characteristic")
        if category_col and category_table:
            select_cols.append(f"{category_table}.{category_col}")
    else:
        if count_request:
            select_cols.append("COUNT(*) AS count")
        elif dt_col:
            select_cols.append(f"date_trunc('day', {base_table}.{dt_col}) AS day")
            select_cols.append("COUNT(*) AS count")
        else:
            select_cols.append("COUNT(*) AS count")

    if not select_cols:
        return None, None, "Unable to select columns for query."

    sql_parts = [
        f"SELECT {', '.join(select_cols)}",
        f"FROM {base_table}",
    ]
    if join_clauses:
        sql_parts.extend(join_clauses)
    if where_clauses:
        sql_parts.append("WHERE " + " AND ".join(where_clauses))
    if dt_col:
        if prefer_raw:
            if "sample_index" in base_columns:
                sql_parts.append(f"ORDER BY {base_table}.{dt_col}, {base_table}.sample_index")
            else:
                sql_parts.append(f"ORDER BY {base_table}.{dt_col}")
        elif not count_request:
            sql_parts.append("GROUP BY day")
            sql_parts.append("ORDER BY day")

    sql = "\n".join(sql_parts)
    return sql, params, None


def _table_candidates(message: str) -> List[str]:
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    lower = message.lower()
    matches = []
    for table in tables:
        name = table.lower()
        if re.search(rf"\b{re.escape(name)}\b", lower):
            matches.append(table)
    return matches


def _build_schema_context(message: str) -> str:
    tables = _table_candidates(message)
    if not tables:
        return ""
    parts: List[str] = []
    for table in tables[:5]:
        schema = get_schema(table=table)
        parts.append(_schema_summary(schema))
    if len(tables) >= 2:
        base = tables[0]
        for other in tables[1:3]:
            join = find_join_path(base, other, max_hops=MAX_SCHEMA_HOPS)
            clauses = join.get("join_clauses") or []
            if clauses:
                parts.append(f"Join path {base} -> {other}: " + " ".join(clauses))
    return "\n".join(parts)


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except Exception:
        pass
    if isinstance(value, bool):
        return str(value)
    try:
        num = float(value)
    except (TypeError, ValueError):
        return _safe_str(value, max_len=24)
    if num.is_integer():
        return str(int(num))
    return f"{num:.3f}"


def _detect_user_language(message: str) -> str:
    if not message:
        return "en"
    if re.search(r"[\u4e00-\u9fff]", message):
        return "zh"
    return "en"


def _t(lang: str, key: str, **kwargs: Any) -> str:
    strings = {
        "en": {
            "capabilities_intro": "I can help with data queries and analysis on: {items}. Tell me which field or filter you need.",
            "capabilities_generic": "I can help with data lookups, summaries, and charts. Tell me which table or field you need.",
            "tool_error_base": "I couldn't complete that request with the available data.",
            "no_rows": "No rows returned for that query.",
            "no_rows_saved": "No rows returned for that query.\nRaw rows saved to: {saved_to}",
            "raw_saved": "Raw rows saved to: {saved_to}",
            "found_rows": "I found {count} row(s){suffix}.",
            "columns": "Columns: {cols}",
            "plot_interpretation": "Plot interpretation:",
            "returned_rows": "Returned {count} row(s):",
            "plot_generated": "Plot generated.",
            "plot_generated_saved": "Plot generated. Saved to: {saved_to}",
            "tool_result_available": "Tool result available.",
            "sample_rows": "Sample rows:",
            "opinion": "Opinion: {opinion}",
            "insufficient_data": "The data is too limited to draw a strong conclusion.",
            "here_is": "Here is the {key}: {value}.",
            "results": "Here are the results: {pairs}.",
            "needs_db_data": (
                "I need to call a tool (get_schema or run_sql) before plotting or analysis. "
                "Try a tool-capable model or rephrase the question."
            ),
            "direct_time_date": "Today is {day}, {date}{tz}.",
            "direct_time_time": "The current time is {time}{tz}.",
            "direct_time_both": "Today is {day}, {date}. The current time is {time}{tz}.",
            "weather_no_access": "I don't have live weather access here. {capabilities}",
            "tool_loop_limit": "I couldn't finish the request within the tool loop limit. Try narrowing the question.",
            "label_no_match": "I couldn't match '{value}' to product categories, sub-categories, or forms. If you can provide the category list, I can use it.",
        },
        "zh": {
            "capabilities_intro": "我可以查詢與分析這些資料：{items}。請告訴我需要的欄位或條件。",
            "capabilities_generic": "我可以幫你做資料查詢、彙總和圖表。請告訴我需要的表或欄位。",
            "tool_error_base": "我暫時無法用現有資料完成這個請求。",
            "no_rows": "這次查詢沒有返回資料。",
            "no_rows_saved": "這次查詢沒有返回資料。\n原始資料已儲存：{saved_to}",
            "raw_saved": "原始資料已儲存：{saved_to}",
            "found_rows": "我找到了 {count} 條記錄{suffix}。",
            "columns": "欄位：{cols}",
            "plot_interpretation": "圖表解讀：",
            "returned_rows": "回傳 {count} 條記錄：",
            "plot_generated": "圖表已產生。",
            "plot_generated_saved": "圖表已產生，儲存到：{saved_to}",
            "tool_result_available": "工具結果已就緒。",
            "sample_rows": "部分樣本：",
            "opinion": "觀點：{opinion}",
            "insufficient_data": "資料量較少，暫時無法得出明確結論。",
            "here_is": "{key} 是 {value}。",
            "results": "結果為：{pairs}。",
            "needs_db_data": "我需要先呼叫工具（get_schema 或 run_sql）才能繪圖或分析。請使用可呼叫工具的模型或換個說法。",
            "direct_time_date": "今天是{day}，{date}{tz}。",
            "direct_time_time": "目前時間是 {time}{tz}。",
            "direct_time_both": "今天是{day}，{date}。目前時間是 {time}{tz}。",
            "weather_no_access": "這裡無法取得即時天氣。{capabilities}",
            "tool_loop_limit": "我沒能在工具呼叫次數限制內完成請求。可以試著把問題縮小一點。",
            "label_no_match": "我無法把「{value}」匹配到產品類別、子類別或表單名稱。如果你能提供類別清單，我就能繼續。",
        },
    }
    bundle = strings.get(lang, strings["en"])
    template = bundle.get(key, strings["en"].get(key, ""))
    return template.format(**kwargs)


def _capabilities_message(lang: str) -> str:
    if SCHEMA_FILE_TABLES:
        items: List[str] = []
        for table, cols in SCHEMA_FILE_TABLES.items():
            if not cols:
                continue
            sample = ", ".join(cols[:3])
            if lang == "zh":
                items.append(f"{table}（{sample}）")
            else:
                items.append(f"{table} ({sample})")
            if len(items) >= 4:
                break
        if items:
            return _t(lang, "capabilities_intro", items="; ".join(items))
    return _t(lang, "capabilities_generic")


def _direct_time_response(message: str, lang: str) -> Optional[str]:
    if not message:
        return None
    lower = message.lower()
    if re.search(r"\b(weather|temperature|forecast|rain|snow|wind|humidity|storm)\b", lower):
        return None
    if not re.search(r"\b(time|date|day|weekday|day of week)\b", lower):
        return None
    now = datetime.now().astimezone()
    day_str = now.strftime("%A")
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")
    tz = now.strftime("%Z")
    tz_note = f" ({tz})" if tz else ("（本地時間）" if lang == "zh" else " (local time)")

    wants_time = bool(re.search(r"\btime|clock\b", lower))
    wants_date = bool(re.search(r"\b(date|day|weekday|day of week)\b", lower))
    if not wants_time and not wants_date:
        return None
    if lang == "zh":
        day_map = {
            "Monday": "星期一",
            "Tuesday": "星期二",
            "Wednesday": "星期三",
            "Thursday": "星期四",
            "Friday": "星期五",
            "Saturday": "星期六",
            "Sunday": "星期日",
        }
        day_str = day_map.get(day_str, day_str)
    if wants_time and wants_date:
        return _t(lang, "direct_time_both", day=day_str, date=date_str, time=time_str, tz=tz_note)
    if wants_time:
        return _t(lang, "direct_time_time", time=time_str, tz=tz_note)
    if wants_date:
        return _t(lang, "direct_time_date", day=day_str, date=date_str, tz=tz_note)
    return None


def _direct_weather_response(message: str, lang: str) -> Optional[str]:
    if not message:
        return None
    lower = message.lower()
    if not re.search(r"\b(weather|temperature|forecast|rain|snow|wind|humidity|storm)\b", lower):
        return None
    return _t(lang, "weather_no_access", capabilities=_capabilities_message(lang))


def _format_simple_result(rows: List[Dict[str, Any]], lang: str) -> Optional[str]:
    if len(rows) != 1:
        return None
    row = rows[0]
    if not isinstance(row, dict) or not row:
        return None
    if len(row) == 1:
        key, value = next(iter(row.items()))
        return _t(lang, "here_is", key=key, value=_format_number(value))
    if len(row) <= 3:
        parts = [f"{key} = {_format_number(value)}" for key, value in row.items()]
        return _t(lang, "results", pairs=", ".join(parts))
    return None


def _format_sample_rows(rows: List[Dict[str, Any]], limit: int = 5) -> Optional[List[str]]:
    if not rows:
        return None
    sample = rows[:limit]
    lines: List[str] = []
    for row in sample:
        if isinstance(row, dict):
            parts = [f"{key}={_format_number(value)}" for key, value in row.items()]
            lines.append("- " + ", ".join(parts))
        else:
            lines.append("- " + _safe_str(row, max_len=80))
    return lines or None


def _analysis_summary(result: Dict[str, Any], message: Optional[str], lang: str) -> str:
    rows = result.get("rows", []) or []
    saved_to = result.get("saved_to")
    opinion = None
    plot_info = _infer_plot_config(rows)
    if plot_info:
        for line in _plot_insight_lines(plot_info, lang):
            if line.startswith("Opinion:"):
                opinion = line[len("Opinion:"):].strip()
                break
            if line.startswith("觀點："):
                opinion = line[len("觀點："):].strip()
                break
    if not opinion:
        opinion = _t(lang, "insufficient_data")

    lines = [_t(lang, "opinion", opinion=opinion)]
    sample_lines = _format_sample_rows(rows, limit=5)
    if sample_lines:
        lines.append(_t(lang, "sample_rows"))
        lines.extend(sample_lines)
    if saved_to:
        lines.append(_t(lang, "raw_saved", saved_to=saved_to))
    return "\n".join(lines)


def _workflow_add(workflow: List[str], text: str) -> None:
    workflow.append(text)


def _log_query(
    query_log: List[Dict[str, Any]],
    sql: str,
    params: Optional[Any],
    source: str,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    entry: Dict[str, Any] = {
        "sql": sql,
        "params": params,
        "source": source,
    }
    if result:
        entry["row_count"] = result.get("row_count")
        entry["truncated"] = result.get("truncated")
        entry["saved_to"] = result.get("saved_to")
    if error:
        entry["error"] = error
    query_log.append(entry)


def _friendly_tool_error(message: Optional[str], error: str, lang: str) -> str:
    base = _t(lang, "tool_error_base")
    return f"{base} {_capabilities_message(lang)}"


def _datetime_range_line(df: pd.DataFrame) -> Optional[str]:
    columns = list(df.columns)
    preferred = [
        col for col in columns
        if any(key in str(col).lower() for key in ("date", "time", "timestamp"))
    ]
    if not preferred:
        return None
    for col in preferred:
        try:
            dt = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            continue
        if not dt.notna().any():
            continue
        min_dt = dt.min()
        max_dt = dt.max()
        if pd.isna(min_dt) or pd.isna(max_dt):
            continue
        return f"{col} range: {min_dt} to {max_dt}"
    return None


def _numeric_summary_line(df: pd.DataFrame) -> Optional[str]:
    parts: List[str] = []
    for col in df.columns:
        try:
            series = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            continue
        if not series.notna().any():
            continue
        min_v = series.min()
        max_v = series.max()
        mean_v = series.mean()
        parts.append(
            f"{col} {_format_number(min_v)}/{_format_number(mean_v)}/{_format_number(max_v)}"
        )
    if not parts:
        return None
    return "Numeric min/avg/max: " + "; ".join(parts[:5])


def _categorical_summary_lines(df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    columns = list(df.columns)
    preferred = [
        col for col in columns
        if any(key in str(col).lower() for key in ("condition", "status", "category", "type", "result"))
    ]
    considered: List[str] = []
    for col in preferred + columns:
        if col in considered:
            continue
        series = df[col].astype(str)
        unique_count = series.nunique(dropna=False)
        if unique_count <= 1 or unique_count > 20:
            continue
        counts = series.value_counts(dropna=False).head(5)
        parts = [f"{_safe_str(idx)}({cnt})" for idx, cnt in counts.items()]
        lines.append(f"{col} top values: " + ", ".join(parts))
        considered.append(col)
        if len(lines) >= 3:
            break
    return lines


def _format_x_value(value: Any) -> str:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return _safe_str(value, max_len=32)


def _numeric_col_score(name: Any) -> int:
    lower = str(name).lower()
    score = 0
    if "value" in lower:
        score += 3
    if "avg" in lower or "mean" in lower:
        score += 2
    if "count" in lower:
        score += 1
    if "rate" in lower:
        score += 1
    return score


def _pick_agg_fn(name: Any) -> str:
    lower = str(name).lower()
    if any(token in lower for token in ("count", "total", "sum")):
        return "sum"
    return "mean"


def _pick_time_granularity(dt_series: pd.Series) -> Tuple[str, str]:
    if dt_series.empty:
        return "D", "Daily"
    span = (dt_series.max() - dt_series.min()).total_seconds()
    if span >= 2 * 24 * 60 * 60:
        return "D", "Daily"
    if span >= 6 * 60 * 60:
        return "H", "Hourly"
    if span >= 30 * 60:
        return "min", "Minute"
    return "S", "Second"


def _build_series_label(
    y_key: Any,
    agg_label: Optional[str],
    granularity_label: Optional[str],
) -> str:
    if agg_label and granularity_label:
        return f"{granularity_label} {agg_label} {y_key}"
    if agg_label:
        return f"{agg_label} {y_key}"
    return str(y_key)


def _prepare_time_series_plot_data(
    df: pd.DataFrame,
    x_key: Any,
    y_key: Any,
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    work = df[[x_key, y_key]].copy()
    work["_x"] = pd.to_datetime(work[x_key], errors="coerce", utc=True)
    work["_y"] = pd.to_numeric(work[y_key], errors="coerce")
    work = work.dropna(subset=["_x", "_y"])
    if work.empty:
        return None

    granularity_code, granularity_label = _pick_time_granularity(work["_x"])
    agg_fn = _pick_agg_fn(y_key)
    agg_label = "sum" if agg_fn == "sum" else "avg"

    work["_bucket"] = work["_x"].dt.floor(granularity_code)
    grouped = work.groupby("_bucket", dropna=False)["_y"].agg(agg_fn).reset_index()
    grouped = grouped.rename(columns={"_bucket": x_key, "_y": y_key})
    grouped = grouped.sort_values(x_key)

    series_label = _build_series_label(y_key, agg_label, granularity_label)
    meta = {
        "series_label": series_label,
        "aggregation": agg_label,
        "x_granularity": granularity_label,
        "x_is_datetime": True,
    }
    return grouped.to_dict("records"), meta


def _prepare_categorical_plot_data(
    df: pd.DataFrame,
    x_key: Any,
    y_key: Any,
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    work = df[[x_key, y_key]].copy()
    work[y_key] = pd.to_numeric(work[y_key], errors="coerce")
    work = work.dropna(subset=[y_key])
    if work.empty:
        return None

    agg_fn = _pick_agg_fn(y_key)
    agg_label = "sum" if agg_fn == "sum" else "avg"
    grouped = work.groupby(x_key, dropna=False)[y_key].agg(agg_fn).reset_index()
    grouped = grouped.sort_values(y_key, ascending=False)
    if len(grouped) > 20:
        grouped = grouped.head(20)

    series_label = _build_series_label(y_key, agg_label, None)
    meta = {
        "series_label": series_label,
        "aggregation": agg_label,
        "x_granularity": None,
        "x_is_datetime": False,
    }
    return grouped.to_dict("records"), meta


def _infer_plot_config(
    rows: List[Dict[str, Any]],
) -> Optional[Tuple[str, Any, Any, List[Dict[str, Any]], Dict[str, Any]]]:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if df.empty or df.shape[1] < 2:
        return None

    columns = list(df.columns)

    def _is_datetime_like(col: Any) -> bool:
        try:
            series = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            return False
        return series.notna().sum() >= max(3, int(len(df) * 0.5))

    preferred = [
        col for col in columns
        if any(key in str(col).lower() for key in ("date", "time", "timestamp", "created", "updated"))
    ]

    x_key: Any = None
    is_datetime = False
    for col in preferred + columns:
        if _is_datetime_like(col):
            x_key = col
            is_datetime = True
            break
    if x_key is None:
        x_key = columns[0]

    numeric_candidates: List[Tuple[int, int, float, Any]] = []
    for col in columns:
        if col == x_key:
            continue
        try:
            series = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            continue
        non_na = int(series.notna().sum())
        if non_na == 0:
            continue
        variance = float(series.var(skipna=True)) if non_na > 1 else 0.0
        if pd.isna(variance):
            variance = 0.0
        score = _numeric_col_score(col)
        numeric_candidates.append((score, non_na, variance, col))

    if not numeric_candidates:
        return None

    numeric_candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    y_key = numeric_candidates[0][3]

    if is_datetime:
        prepared = _prepare_time_series_plot_data(df, x_key, y_key)
        if not prepared:
            return None
        data, meta = prepared
        return ("line", x_key, y_key, data, meta)

    prepared = _prepare_categorical_plot_data(df, x_key, y_key)
    if not prepared:
        return None
    data, meta = prepared
    return ("bar", x_key, y_key, data, meta)


def _plot_insight_lines(
    plot_info: Tuple[str, Any, Any, List[Dict[str, Any]], Dict[str, Any]],
    lang: str,
) -> List[str]:
    kind, x_key, y_key, data, meta = plot_info
    series_label = meta.get("series_label") or str(y_key)
    df = pd.DataFrame(data)
    if df.empty or x_key not in df.columns or y_key not in df.columns:
        return []

    lines: List[str] = []
    if kind == "line":
        if lang == "zh":
            lines.append(f"為什麼畫這個圖：展示 {series_label} 隨 {x_key} 的變化，便於觀察趨勢和波動。")
        else:
            lines.append(
                f"Why this graph: it shows {series_label} over {x_key} to surface the trend and spikes."
            )
        df[y_key] = pd.to_numeric(df[y_key], errors="coerce")
        df = df.dropna(subset=[y_key])
        if df.empty or len(df) < 2:
            return lines

        x_dt = pd.to_datetime(df[x_key], errors="coerce")
        if x_dt.notna().sum() >= max(2, int(len(df) * 0.5)):
            df["_x"] = x_dt
        else:
            df["_x"] = pd.to_numeric(df[x_key], errors="coerce")
        df = df.dropna(subset=["_x"]).sort_values("_x")
        if df.empty or len(df) < 2:
            return lines

        y_series = df[y_key]
        window = max(1, len(y_series) // 5)
        head = y_series.iloc[:window].mean()
        tail = y_series.iloc[-window:].mean()
        delta = tail - head
        pct = None
        if head not in (0, None) and not pd.isna(head):
            pct = delta / abs(head)

        if pct is not None and abs(pct) >= 0.05:
            trend = "upward" if delta > 0 else "downward"
        elif abs(delta) > 0:
            trend = "mostly flat"
        else:
            trend = "flat"

        if pct is not None:
            change_desc = f"{_format_number(head)} -> {_format_number(tail)} ({pct * 100:.1f}%)"
        else:
            change_desc = f"{_format_number(head)} -> {_format_number(tail)}"

        mean = y_series.mean()
        std = y_series.std()
        var_desc = "low"
        if mean == 0 or pd.isna(mean):
            if std and std > 0:
                var_desc = "high"
        else:
            cv = abs(std / mean) if mean else 0
            if cv >= 0.3:
                var_desc = "high"
            elif cv >= 0.1:
                var_desc = "moderate"

        extra = ""
        max_val = y_series.max()
        min_val = y_series.min()
        if not pd.isna(max_val) and not pd.isna(min_val) and max_val != min_val:
            max_idx = y_series.idxmax()
            min_idx = y_series.idxmin()
            max_x = _format_x_value(df.loc[max_idx, "_x"])
            min_x = _format_x_value(df.loc[min_idx, "_x"])
            if lang == "zh":
                extra = (
                    f" 峰值約為 {_format_number(max_val)}（{max_x}），"
                    f"低點約為 {_format_number(min_val)}（{min_x}）。"
                )
            else:
                extra = (
                    f" Peaks at {_format_number(max_val)} around {max_x}, "
                    f"with a low of {_format_number(min_val)} around {min_x}."
                )

        if lang == "zh":
            trend_map = {
                "upward": "上升",
                "downward": "下降",
                "mostly flat": "基本持平",
                "flat": "持平",
            }
            var_map = {"low": "低", "moderate": "中", "high": "高"}
            lines.append(
                f"圖中顯示：趨勢{trend_map.get(trend, trend)}（{change_desc}）；波動程度{var_map.get(var_desc, var_desc)}。{extra}"
            )
            if trend == "upward":
                opinion = "數值有上升趨勢，如不符合預期，建議關注上升時段的變化。"
            elif trend == "downward":
                opinion = "數值呈下降趨勢，請確認是否符合預期並留意持續下降。"
            else:
                opinion = "整體較穩定，重點關注個別波動點。"
            lines.append(f"觀點：{opinion}")
        else:
            lines.append(
                f"What it shows: {trend} trend ({change_desc}); variability is {var_desc}.{extra}"
            )
            if trend == "upward":
                opinion = "Values drift upward; if that is undesirable, review changes around the rise."
            elif trend == "downward":
                opinion = "Values trend downward; confirm this matches expectations and watch for continued drift."
            else:
                opinion = "Values look stable overall; focus on isolated spikes rather than a sustained shift."
            lines.append(f"Opinion: {opinion}")
        return lines
    if lang == "zh":
        lines.append(
            f"為什麼畫這個圖：比較各 {x_key} 的 {series_label}，突出差異。"
        )
    else:
        lines.append(
            f"Why this graph: it compares {series_label} across {x_key} categories to show relative differences."
        )
    df[y_key] = pd.to_numeric(df[y_key], errors="coerce")
    df = df.dropna(subset=[y_key]).sort_values(y_key, ascending=False)
    if df.empty:
        return lines

    top = df.head(3)
    top_items = ", ".join(
        f"{_safe_str(row[x_key])}({_format_number(row[y_key])})"
        for _, row in top.iterrows()
    )
    total = df[y_key].sum()
    top_sum = top[y_key].sum()
    concentration = top_sum / total if total else None
    spread = "concentrated" if concentration and concentration >= 0.7 else "spread"
    if lang == "zh":
        spread_text = "集中" if spread == "concentrated" else "分散"
        lines.append(f"圖中顯示：排名靠前的是 {top_items}；分佈較{spread_text}。")
        if spread == "concentrated":
            opinion = "少數類別佔主導，建議優先關注這些類別。"
        else:
            opinion = "沒有單一類別佔主導，變化比較分散。"
        lines.append(f"觀點：{opinion}")
    else:
        lines.append(f"What it shows: top categories are {top_items}; distribution is {spread}.")
        if spread == "concentrated":
            opinion = "A few categories dominate; prioritize those segments for impact."
        else:
            opinion = "No single category dominates; changes are broadly distributed."
        lines.append(f"Opinion: {opinion}")
    return lines


def _plot_item_from_result(
    plot: Dict[str, Any],
    title: str,
    kind: str,
    x_key: Any,
    y_key: Any,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    item = {
        "png_base64": plot.get("png_base64"),
        "saved_to": plot.get("saved_to"),
        "title": title,
        "kind": kind,
        "x_key": x_key,
        "y_key": y_key,
    }
    if note:
        item["note"] = note
    return item


def _auto_plot_from_result(
    result: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    rows = result.get("rows", []) or []
    config = _infer_plot_config(rows)
    if not config:
        return [], None
    kind, x_key, y_key, data, meta = config
    series_label = meta.get("series_label") or str(y_key)
    title = f"{_safe_str(series_label)} by {_safe_str(x_key)}"
    items: List[Dict[str, Any]] = []
    notes: List[str] = []
    try:
        plot = make_plot(
            title=title,
            kind=kind,
            x_key=x_key,
            y_key=y_key,
            data=data,
            legend_label=series_label,
        )
    except Exception:
        return [], None
    note = f"Plot: {series_label} by {_safe_str(x_key)}."
    if meta.get("x_granularity") and meta.get("aggregation"):
        agg_note = f"Aggregated as {meta['x_granularity'].lower()} {meta['aggregation']}."
        note = f"{note}\n{agg_note}"
    items.append(_plot_item_from_result(plot, title, kind, x_key, y_key, note=note))
    notes.append(note)

    df = pd.DataFrame(rows)
    if y_key in df.columns:
        series = pd.to_numeric(df[y_key], errors="coerce").dropna()
        if len(series) >= 10:
            hist_title = f"Distribution of {_safe_str(y_key)}"
            try:
                hist_plot = make_plot(
                    title=hist_title,
                    kind="hist",
                    x_key=y_key,
                    y_key=y_key,
                    data=rows,
                    legend_label=str(y_key),
                )
            except Exception:
                hist_plot = None
            if hist_plot:
                hist_note = f"Plot: distribution of {_safe_str(y_key)}."
                items.append(_plot_item_from_result(hist_plot, hist_title, "hist", y_key, y_key, note=hist_note))
                notes.append(hist_note)
    return items, "\n".join(notes) if notes else None


def _summarize_rows(result: Dict[str, Any], message: Optional[str], lang: str) -> str:
    rows = result.get("rows", []) or []
    truncated = bool(result.get("truncated", False))
    saved_to = result.get("saved_to")
    if not rows:
        if saved_to:
            return _t(lang, "no_rows_saved", saved_to=saved_to)
        return _t(lang, "no_rows")

    simple_text = _format_simple_result(rows, lang)
    if simple_text:
        return simple_text
    if message and _is_analysis_request(message):
        return _analysis_summary(result, message, lang)

    df = pd.DataFrame(rows)
    lines: List[str] = []
    suffix = "（已截斷）" if truncated and lang == "zh" else (" (truncated)" if truncated else "")
    lines.append(_t(lang, "found_rows", count=len(rows), suffix=suffix))
    if saved_to:
        lines.append(_t(lang, "raw_saved", saved_to=saved_to))

    columns = [str(c) for c in df.columns.tolist()]
    if columns:
        if len(columns) <= 8:
            lines.append(_t(lang, "columns", cols=", ".join(columns)))
        else:
            lines.append(_t(lang, "columns", cols=", ".join(columns[:8]) + " ..."))

    date_line = _datetime_range_line(df)
    if date_line:
        lines.append(date_line)

    numeric_line = _numeric_summary_line(df)
    if numeric_line:
        lines.append(numeric_line)

    lines.extend(_categorical_summary_lines(df))

    if message and (_wants_plot(message) or _needs_opinion(message)):
        plot_info = _infer_plot_config(rows)
        if plot_info:
            lines.append(_t(lang, "plot_interpretation"))
            lines.extend(_plot_insight_lines(plot_info, lang))
    return "\n".join(lines)


def _looks_unhelpful_response(text: str, result: Optional[Dict[str, Any]] = None) -> bool:
    if not text or not text.strip():
        return True
    if result:
        rows = result.get("rows", []) or []
        if len(rows) == 1 and isinstance(rows[0], dict) and rows[0]:
            columns = [
                str(col).lower()
                for col in rows[0].keys()
                if isinstance(col, str) and len(col) >= 2
            ]
            lower = text.lower()
            if re.search(r"\d", text) or any(col in lower for col in columns):
                return False
    if len(text.strip()) < 40:
        return True
    lower = text.lower()
    generic_markers = (
        "json response",
        "json (javascript object notation)",
        "schema section",
        "database schema",
        "result of querying",
        "array of objects",
        "result set",
        "row_count",
        "truncated",
        "columns",
        "tables",
        "views",
        "indexes",
    )
    if any(marker in lower for marker in generic_markers):
        return True
    if result:
        rows = result.get("rows", []) or []
        if rows:
            columns = [
                str(col).lower()
                for col in rows[0].keys()
                if isinstance(col, str) and len(col) >= 4
            ]
            if columns and not any(col in lower for col in columns):
                if not re.search(r"\d", text):
                    return True
    return False


def _format_tool_fallback(
    name: str,
    result: Dict[str, Any],
    message: Optional[str],
    lang: str,
) -> str:
    if name == "run_sql":
        rows = result.get("rows", [])
        row_count = result.get("row_count", len(rows))
        truncated = result.get("truncated", False)
        saved_to = result.get("saved_to")
        if row_count <= 20 and not truncated:
            if message and (_wants_plot(message) or _needs_opinion(message)):
                return _summarize_rows(result, message=message, lang=lang)
            rows_json = json.dumps(rows, indent=2, ensure_ascii=True, default=str)
            suffix = f"\n\n{_t(lang, 'raw_saved', saved_to=saved_to)}" if saved_to else ""
            return f"{_t(lang, 'returned_rows', count=row_count)}\n\n{rows_json}{suffix}"
        return _summarize_rows(result, message=message, lang=lang)
    if name == "make_plot":
        saved_to = result.get("saved_to")
        if saved_to:
            return _t(lang, "plot_generated_saved", saved_to=saved_to)
        return _t(lang, "plot_generated")
    return _t(lang, "tool_result_available")


def _maybe_autoplot(
    message: str,
    last_tool_name: Optional[str],
    last_tool_result: Optional[Dict[str, Any]],
    plot_b64: Optional[str],
    plot_saved_to: Optional[str],
    plot_items: List[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if plot_b64 is not None:
        return plot_b64, None, plot_saved_to
    if last_tool_name != "run_sql" or not last_tool_result:
        return plot_b64, None, plot_saved_to
    if not _wants_plot(message):
        return plot_b64, None, plot_saved_to
    items, note = _auto_plot_from_result(last_tool_result)
    if not items:
        return plot_b64, None, plot_saved_to
    plot_items.extend(items)
    last_plot = items[-1]
    return last_plot.get("png_base64"), note, last_plot.get("saved_to")


def _needs_db_data_message(lang: str) -> str:
    return _t(lang, "needs_db_data")


def _force_json_tool_message() -> str:
    return (
        "Respond ONLY with a JSON object for a tool call, no prose. "
        "Example: {\"name\":\"describe_table\",\"parameters\":{\"table\":\"spc_measurements\"}}"
    )


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
    user_lang = _detect_user_language(req.message)
    needs_tools = _requires_tool_call(req.message)
    if user_lang == "zh":
        language_note = "最終回覆請使用繁體中文。"
    else:
        language_note = "Respond in English."
    if needs_tools:
        system_instructions = (
            "You are a helpful assistant for an internal analytics app backed by a SQL database.\n"
            "Always inspect schema first; never guess tables or columns.\n\n"
            "TOOLS:\n"
            "- get_schema(table?)\n"
            "- describe_table(table)\n"
            "- find_join_path(from_table, to_table)\n"
            "- run_sql(sql, params?)\n"
            "- make_plot(...)\n\n"
            "RULES:\n"
            "- Before any run_sql, call get_schema/describe_table for every table you will query.\n"
            "- Apply all user filters (dates, form_name, category_name). If form_name values are given, join forms and filter forms.form_name.\n"
        "- If the request is about a category itself, query product_categories directly; otherwise join via sub_categories/forms and filter product_categories.category_name.\n"
        "- If a term is unclear (e.g., \"Chain Line\"), look it up in product_categories, then sub_categories, then forms, and use the first match.\n"
        "- Join tables by ID columns (category_id, sub_category_id, form_id, field_index), not by name.\n"
        "- For measurement/production questions, start from spc_measurements and join upward: form_fields -> forms -> sub_categories -> product_categories.\n"
        "- For category measurements, include form_id, form_name, field_index, line_name, inspect_time, value; include sample_index when order matters.\n"
        "- Exclude sub_categories.sub_category_name = '封存' by default unless the user asks for archived data.\n"
        "- Use only needed columns; avoid SELECT * unless the user asked for a sample.\n"
            "- For spc_measurements analysis: fetch filtered raw rows first, then aggregate/plot.\n"
            "- If results are empty or too small, run another query (widen filters) instead of guessing.\n"
            "- Multiple small queries are OK. Use find_join_path when unsure.\n"
            "- SQL must be a single SELECT with named params. Do not use LIMIT/max_rows unless the user asks.\n"
            "- For charts: run_sql first, then make_plot. Multiple plots allowed; include legend_label.\n"
        "- Provide concise summaries plus interpretation/opinion when asked; do not describe JSON schemas.\n"
        "- Never ask for or reveal secrets.\n"
        f"{language_note}\n"
        )
    else:
        system_instructions = (
            "You are a professional, concise assistant.\n"
            "Answer common chat questions directly.\n"
            "If a request needs database data you do not have, say what data you can help with.\n"
            "Avoid guessing or speculation.\n"
            f"{language_note}\n"
        )


    # Ollama messages are the same idea: list of {role, content}
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_instructions}]
    if req.history:
        messages.extend(req.history)
    messages.append({"role": "user", "content": req.message})

    trace: List[Dict[str, Any]] = []
    _trace_event(trace, "input", {
        "message": req.message,
        "history_len": len(req.history or []),
    })
    workflow: List[str] = []
    _workflow_add(workflow, "Received request.")
    query_log: List[Dict[str, Any]] = []
    if not needs_tools:
        quick = _direct_time_response(req.message, user_lang)
        if quick:
            _trace_event(trace, "assistant", {
                "content": quick,
                "tool_calls": [],
            })
            _workflow_add(workflow, "Answered directly without database.")
            return ChatResponse(text=quick, trace=trace, plots=[], workflow=workflow, queries=query_log)
        weather = _direct_weather_response(req.message, user_lang)
        if weather:
            _trace_event(trace, "assistant", {
                "content": weather,
                "tool_calls": [],
            })
            _workflow_add(workflow, "Returned safe fallback without database.")
            return ChatResponse(text=weather, trace=trace, plots=[], workflow=workflow, queries=query_log)
        resp = ollama_chat(messages=messages, tools=None)
        assistant_msg = resp.get("message", {}) or {}
        content = assistant_msg.get("content") or ""
        if not content.strip():
            content = _capabilities_message(user_lang)
        _trace_event(trace, "assistant", {
            "content": content,
            "tool_calls": [],
        })
        _workflow_add(workflow, "Answered directly without database.")
        return ChatResponse(text=content, trace=trace, plots=[], workflow=workflow, queries=query_log)

    plot_items: List[Dict[str, Any]] = []
    plot_b64: Optional[str] = None
    plot_note: Optional[str] = None
    plot_saved_to: Optional[str] = None
    query_saved_to: Optional[str] = None
    last_tool_signature: Optional[str] = None
    last_tool_result: Optional[Dict[str, Any]] = None
    last_tool_name: Optional[str] = None
    last_tool_error: Optional[str] = None
    force_tool_attempted = False
    plot_requires_sql_prompted = False
    force_json_tool_attempted = False
    force_sql_attempted = False
    schema_checked = False
    schema_prompted = False
    schema_cache: Dict[str, Dict[str, Any]] = {}
    schema_hint = _schema_hints_from_file(req.message)
    schema_hint_sent = False
    if req.history:
        for msg in req.history:
            if (
                msg.get("role") == "tool"
                and msg.get("name") in {"get_schema", "describe_table"}
            ):
                content = msg.get("content") or ""
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict):
                    _record_schema(schema_cache, payload)
                    schema_checked = bool(schema_cache)

    def _respond(
        text: str,
        plot_png_base64: Optional[str] = None,
        plot_saved_to: Optional[str] = None,
        query_saved_to: Optional[str] = None,
    ) -> ChatResponse:
        return ChatResponse(
            text=text,
            plot_png_base64=plot_png_base64,
            plot_saved_to=plot_saved_to,
            query_saved_to=query_saved_to,
            trace=trace,
            plots=plot_items,
            workflow=workflow,
            queries=query_log,
        )

    # Agent loop: model -> tool_calls -> execute -> tool messages -> model ... :contentReference[oaicite:7]{index=7}
    for _ in range(MAX_TOOL_TURNS):  # max turns to avoid infinite loops
        if schema_hint and not schema_hint_sent and not schema_checked:
            messages.append({"role": "system", "content": schema_hint})
            schema_hint_sent = True
        resp = ollama_chat(messages=messages, tools=TOOLS)
        print("OLLAMA RAW:", json.dumps(resp, indent=2)[:4000])

        assistant_msg = resp.get("message", {}) or {}
        tool_calls = get_tool_calls(assistant_msg)
        trace_calls = []
        for tc in tool_calls:
            fn = tc.get("function") or {}
            trace_calls.append({
                "name": fn.get("name"),
                "arguments": _trace_tool_args(fn.get("name"), fn.get("arguments") or {}),
            })
        _trace_event(trace, "assistant", {
            "content": assistant_msg.get("content") or "",
            "tool_calls": trace_calls,
        })

        if tool_calls:
            signature = _tool_signature(tool_calls)
            if signature == last_tool_signature and last_tool_result and last_tool_name:
                plot_b64, plot_note, plot_saved_to = _maybe_autoplot(
                    req.message, last_tool_name, last_tool_result, plot_b64, plot_saved_to, plot_items
                )
                text = _format_tool_fallback(
                    last_tool_name,
                    last_tool_result,
                    req.message,
                    user_lang,
                )
                if plot_note:
                    text = f"{text}\n{plot_note}"
                return _respond(
                    text=text,
                    plot_png_base64=plot_b64,
                    plot_saved_to=plot_saved_to,
                    query_saved_to=query_saved_to,
                )

        # Always append the assistant message to the conversation
        # (may contain content, or tool_calls, or both depending on model)
        messages.append(assistant_msg)

        # No tool calls => final natural language answer
        if not tool_calls:
            final_text = assistant_msg.get("content", "") or ""
            if not last_tool_result and _requires_tool_call(req.message):
                category_value = _extract_category_value(req.message)
                mentions_sub_category = bool(re.search(r"\bsub[_ ]?category\b", req.message, re.IGNORECASE))
                mentions_form = bool(re.search(r"\bform(_name|_id)?s?\b", req.message, re.IGNORECASE))
                mentions_measurements = _message_mentions_table(req.message, "spc_measurements")
                needs_category_path = (
                    _is_analysis_request(req.message)
                    or mentions_sub_category
                    or mentions_form
                    or mentions_measurements
                )
                if category_value and "product_categories" not in schema_cache:
                    try:
                        _trace_event(trace, "tool_call", {
                            "name": "describe_table",
                            "arguments": {"table": "product_categories"},
                            "source": "server",
                        })
                        result = describe_table(table="product_categories")
                    except Exception as exc:
                        _trace_event(trace, "tool_error", {
                            "name": "describe_table",
                            "error": str(exc),
                            "source": "server",
                        })
                        return _respond(
                            text=_friendly_tool_error(req.message, str(exc), user_lang),
                            plot_png_base64=plot_b64,
                            plot_saved_to=plot_saved_to,
                            query_saved_to=query_saved_to,
                        )
                    _record_schema(schema_cache, result)
                    schema_checked = bool(schema_cache)
                    _workflow_add(workflow, "Schema checked: product_categories.")
                    _trace_event(trace, "tool_result", {
                        "name": "describe_table",
                        "source": "server",
                        **_trace_schema_result(result),
                    })
                    messages.append({
                        "role": "tool",
                        "name": "describe_table",
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    })
                    continue
                if category_value and needs_category_path and "sub_categories" not in schema_cache:
                    try:
                        _trace_event(trace, "tool_call", {
                            "name": "describe_table",
                            "arguments": {"table": "sub_categories"},
                            "source": "server",
                        })
                        result = describe_table(table="sub_categories")
                    except Exception as exc:
                        _trace_event(trace, "tool_error", {
                            "name": "describe_table",
                            "error": str(exc),
                            "source": "server",
                        })
                        return _respond(
                            text=_friendly_tool_error(req.message, str(exc), user_lang),
                            plot_png_base64=plot_b64,
                            plot_saved_to=plot_saved_to,
                            query_saved_to=query_saved_to,
                        )
                    _record_schema(schema_cache, result)
                    schema_checked = bool(schema_cache)
                    _workflow_add(workflow, "Schema checked: sub_categories.")
                    _trace_event(trace, "tool_result", {
                        "name": "describe_table",
                        "source": "server",
                        **_trace_schema_result(result),
                    })
                    messages.append({
                        "role": "tool",
                        "name": "describe_table",
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    })
                    continue
                if category_value and needs_category_path and "forms" not in schema_cache:
                    try:
                        _trace_event(trace, "tool_call", {
                            "name": "describe_table",
                            "arguments": {"table": "forms"},
                            "source": "server",
                        })
                        result = describe_table(table="forms")
                    except Exception as exc:
                        _trace_event(trace, "tool_error", {
                            "name": "describe_table",
                            "error": str(exc),
                            "source": "server",
                        })
                        return _respond(
                            text=_friendly_tool_error(req.message, str(exc), user_lang),
                            plot_png_base64=plot_b64,
                            plot_saved_to=plot_saved_to,
                            query_saved_to=query_saved_to,
                        )
                    _record_schema(schema_cache, result)
                    schema_checked = bool(schema_cache)
                    _workflow_add(workflow, "Schema checked: forms.")
                    _trace_event(trace, "tool_result", {
                        "name": "describe_table",
                        "source": "server",
                        **_trace_schema_result(result),
                    })
                    messages.append({
                        "role": "tool",
                        "name": "describe_table",
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    })
                    continue
                if (
                    _message_mentions_table(req.message, "spc_measurements")
                    and "spc_measurements" not in schema_cache
                ):
                    try:
                        _trace_event(trace, "tool_call", {
                            "name": "describe_table",
                            "arguments": {"table": "spc_measurements"},
                            "source": "server",
                        })
                        result = describe_table(table="spc_measurements")
                    except Exception as exc:
                        _trace_event(trace, "tool_error", {
                            "name": "describe_table",
                            "error": str(exc),
                            "source": "server",
                        })
                        return _respond(
                            text=_friendly_tool_error(req.message, str(exc), user_lang),
                            plot_png_base64=plot_b64,
                            plot_saved_to=plot_saved_to,
                            query_saved_to=query_saved_to,
                        )
                    _record_schema(schema_cache, result)
                    schema_checked = bool(schema_cache)
                    _workflow_add(workflow, "Schema checked: spc_measurements.")
                    _trace_event(trace, "tool_result", {
                        "name": "describe_table",
                        "source": "server",
                        **_trace_schema_result(result),
                    })
                    messages.append({
                        "role": "tool",
                        "name": "describe_table",
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    })
                    continue
                if _extract_form_names(req.message) and "forms" not in schema_cache:
                    try:
                        _trace_event(trace, "tool_call", {
                            "name": "describe_table",
                            "arguments": {"table": "forms"},
                            "source": "server",
                        })
                        result = describe_table(table="forms")
                    except Exception as exc:
                        _trace_event(trace, "tool_error", {
                            "name": "describe_table",
                            "error": str(exc),
                            "source": "server",
                        })
                        return _respond(
                            text=_friendly_tool_error(req.message, str(exc), user_lang),
                            plot_png_base64=plot_b64,
                            plot_saved_to=plot_saved_to,
                            query_saved_to=query_saved_to,
                        )
                    _record_schema(schema_cache, result)
                    schema_checked = bool(schema_cache)
                    _workflow_add(workflow, "Schema checked: forms.")
                    _trace_event(trace, "tool_result", {
                        "name": "describe_table",
                        "source": "server",
                        **_trace_schema_result(result),
                    })
                    messages.append({
                        "role": "tool",
                        "name": "describe_table",
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    })
                    continue
            if not last_tool_result and _requires_tool_call(req.message) and not force_tool_attempted:
                messages.append({
                    "role": "system",
                    "content": "Tool required: call get_schema or run_sql to answer. Do not respond with natural language yet.",
                })
                force_tool_attempted = True
                continue
            if not last_tool_result and _requires_tool_call(req.message) and not force_json_tool_attempted:
                messages.append({
                    "role": "system",
                    "content": _force_json_tool_message(),
                })
                force_json_tool_attempted = True
                continue
            if not last_tool_result and _requires_tool_call(req.message) and not force_sql_attempted:
                schema_context = _build_schema_context(req.message)
                if schema_context:
                    messages.append({
                        "role": "system",
                        "content": f"Schema summary:\n{schema_context}",
                    })
                messages.append({
                    "role": "system",
                    "content": "Return a single SELECT SQL statement only, no prose.",
                })
                force_sql_attempted = True
                continue
            sql_text, sql_params = _extract_sql_payload(final_text)
            if sql_text:
                if not schema_checked:
                    if not schema_prompted:
                        messages.append({
                            "role": "system",
                            "content": (
                                "Schema required before SQL. Call get_schema or describe_table "
                                "for each table you will query, then call run_sql."
                            ),
                        })
                        schema_prompted = True
                    continue
                column_error = _validate_sql_columns(sql_text, schema_cache)
                if column_error:
                    if not schema_prompted:
                        messages.append({
                            "role": "system",
                            "content": f"SQL invalid: {column_error}",
                        })
                        schema_prompted = True
                    continue
                try:
                    _trace_event(trace, "tool_call", {
                        "name": "run_sql",
                        "arguments": _trace_tool_args("run_sql", {"sql": sql_text, "params": sql_params}),
                        "source": "assistant_sql",
                    })
                    result = run_sql(sql=sql_text, params=sql_params, max_rows=None)
                except Exception as exc:
                    _trace_event(trace, "tool_error", {
                        "name": "run_sql",
                        "error": str(exc),
                        "source": "assistant_sql",
                    })
                    _log_query(query_log, sql_text, sql_params, "assistant_sql", error=str(exc))
                    _workflow_add(workflow, "SQL execution failed.")
                    return _respond(
                        text=_friendly_tool_error(req.message, str(exc), user_lang),
                        plot_png_base64=plot_b64,
                        plot_saved_to=plot_saved_to,
                        query_saved_to=query_saved_to,
                    )
                last_tool_result = result
                last_tool_name = "run_sql"
                query_saved_to = result.get("saved_to")
                _log_query(query_log, sql_text, sql_params, "assistant_sql", result=result)
                _workflow_add(workflow, "SQL executed.")
                _workflow_add(workflow, f"Rows returned: {result.get('row_count')}.")
                if result.get("row_count", 0) == 0:
                    plot_b64 = None
                    plot_saved_to = None
                    plot_items.clear()
                _trace_event(trace, "tool_result", {
                    "name": "run_sql",
                    "row_count": result.get("row_count"),
                    "truncated": result.get("truncated"),
                    "saved_to": result.get("saved_to"),
                    "source": "assistant_sql",
                })
                plot_b64, plot_note, plot_saved_to = _maybe_autoplot(
                    req.message, last_tool_name, last_tool_result, plot_b64, plot_saved_to, plot_items
                )
                text = _summarize_rows(result, message=req.message, lang=user_lang)
                if plot_note:
                    text = f"{text}\n{plot_note}"
                _workflow_add(workflow, "Answer generated from query results.")
                return _respond(
                    text=text,
                    plot_png_base64=plot_b64,
                    plot_saved_to=plot_saved_to,
                    query_saved_to=query_saved_to,
                )
            sql_text = _extract_sql_from_text(final_text)
            if sql_text and (force_sql_attempted or final_text.strip().lower().startswith("select")):
                if not schema_checked:
                    if not schema_prompted:
                        messages.append({
                            "role": "system",
                            "content": (
                                "Schema required before SQL. Call get_schema or describe_table "
                                "for each table you will query, then call run_sql."
                            ),
                        })
                        schema_prompted = True
                    continue
                column_error = _validate_sql_columns(sql_text, schema_cache)
                if column_error:
                    if not schema_prompted:
                        messages.append({
                            "role": "system",
                            "content": f"SQL invalid: {column_error}",
                        })
                        schema_prompted = True
                    continue
                try:
                    _trace_event(trace, "tool_call", {
                        "name": "run_sql",
                        "arguments": _trace_tool_args("run_sql", {"sql": sql_text}),
                        "source": "assistant_sql",
                    })
                    result = run_sql(sql=sql_text, params=None, max_rows=None)
                except Exception as exc:
                    _trace_event(trace, "tool_error", {
                        "name": "run_sql",
                        "error": str(exc),
                        "source": "assistant_sql",
                    })
                    _log_query(query_log, sql_text, None, "assistant_sql", error=str(exc))
                    _workflow_add(workflow, "SQL execution failed.")
                    return _respond(
                        text=_friendly_tool_error(req.message, str(exc), user_lang),
                        plot_png_base64=plot_b64,
                        plot_saved_to=plot_saved_to,
                        query_saved_to=query_saved_to,
                    )
                last_tool_result = result
                last_tool_name = "run_sql"
                query_saved_to = result.get("saved_to")
                _log_query(query_log, sql_text, None, "assistant_sql", result=result)
                _workflow_add(workflow, "SQL executed.")
                _workflow_add(workflow, f"Rows returned: {result.get('row_count')}.")
                if result.get("row_count", 0) == 0:
                    plot_b64 = None
                    plot_saved_to = None
                    plot_items.clear()
                _trace_event(trace, "tool_result", {
                    "name": "run_sql",
                    "row_count": result.get("row_count"),
                    "truncated": result.get("truncated"),
                    "saved_to": result.get("saved_to"),
                    "source": "assistant_sql",
                })
                plot_b64, plot_note, plot_saved_to = _maybe_autoplot(
                    req.message, last_tool_name, last_tool_result, plot_b64, plot_saved_to, plot_items
                )
                text = _summarize_rows(result, message=req.message, lang=user_lang)
                if plot_note:
                    text = f"{text}\n{plot_note}"
                _workflow_add(workflow, "Answer generated from query results.")
                return _respond(
                    text=text,
                    plot_png_base64=plot_b64,
                    plot_saved_to=plot_saved_to,
                    query_saved_to=query_saved_to,
                )
            if not last_tool_result and _requires_tool_call(req.message):
                sql, params, error = _auto_query_from_message(req.message, query_log, workflow, user_lang)
                if sql:
                    try:
                        _trace_event(trace, "tool_call", {
                            "name": "run_sql",
                            "arguments": _trace_tool_args("run_sql", {"sql": sql, "params": params}),
                            "source": "auto_query",
                        })
                        result = run_sql(sql=sql, params=params, max_rows=None)
                    except Exception as exc:
                        _trace_event(trace, "tool_error", {
                            "name": "run_sql",
                            "error": str(exc),
                            "source": "auto_query",
                        })
                        _log_query(query_log, sql, params, "auto_query", error=str(exc))
                        _workflow_add(workflow, "SQL execution failed.")
                        return _respond(
                            text=_friendly_tool_error(req.message, str(exc), user_lang),
                            plot_png_base64=plot_b64,
                            plot_saved_to=plot_saved_to,
                            query_saved_to=query_saved_to,
                        )
                    last_tool_result = result
                    last_tool_name = "run_sql"
                    query_saved_to = result.get("saved_to")
                    _log_query(query_log, sql, params, "auto_query", result=result)
                    _workflow_add(workflow, "SQL executed.")
                    _workflow_add(workflow, f"Rows returned: {result.get('row_count')}.")
                    if result.get("row_count", 0) == 0:
                        plot_b64 = None
                        plot_saved_to = None
                        plot_items.clear()
                    _trace_event(trace, "tool_result", {
                        "name": "run_sql",
                        "row_count": result.get("row_count"),
                        "truncated": result.get("truncated"),
                        "saved_to": result.get("saved_to"),
                        "source": "auto_query",
                    })
                    plot_b64, plot_note, plot_saved_to = _maybe_autoplot(
                        req.message, last_tool_name, last_tool_result, plot_b64, plot_saved_to, plot_items
                    )
                    text = _summarize_rows(result, message=req.message, lang=user_lang)
                    if plot_note:
                        text = f"{text}\n{plot_note}"
                    _workflow_add(workflow, "Answer generated from query results.")
                    return _respond(
                        text=text,
                        plot_png_base64=plot_b64,
                        plot_saved_to=plot_saved_to,
                        query_saved_to=query_saved_to,
                    )
                if error:
                    return _respond(
                        text=error,
                        plot_png_base64=plot_b64,
                        plot_saved_to=plot_saved_to,
                        query_saved_to=query_saved_to,
                    )
            plot_b64, plot_note, plot_saved_to = _maybe_autoplot(
                req.message, last_tool_name, last_tool_result, plot_b64, plot_saved_to, plot_items
            )
            if plot_b64 and last_tool_result and last_tool_name == "run_sql":
                text = _summarize_rows(last_tool_result, message=req.message, lang=user_lang)
                if plot_note:
                    text = f"{text}\n{plot_note}"
                _workflow_add(workflow, "Answer generated from query results.")
                return _respond(
                    text=text,
                    plot_png_base64=plot_b64,
                    plot_saved_to=plot_saved_to,
                    query_saved_to=query_saved_to,
                )
            if (
                last_tool_result
                and last_tool_name == "run_sql"
                and _looks_unhelpful_response(final_text, last_tool_result)
            ):
                _workflow_add(workflow, "Answer generated from query results.")
                return _respond(
                    text=_summarize_rows(last_tool_result, message=req.message, lang=user_lang),
                    plot_png_base64=plot_b64,
                    plot_saved_to=plot_saved_to,
                    query_saved_to=query_saved_to,
                )
            if not last_tool_result and _requires_tool_call(req.message):
                return _respond(
                    text=_needs_db_data_message(user_lang),
                    plot_png_base64=plot_b64,
                    plot_saved_to=plot_saved_to,
                    query_saved_to=query_saved_to,
                )
            return _respond(
                text=final_text,
                plot_png_base64=plot_b64,
                plot_saved_to=plot_saved_to,
                query_saved_to=query_saved_to,
            )

        # Execute all tool calls returned
        for tc in tool_calls:
            fn = (tc.get("function") or {})
            name = fn.get("name")
            args = fn.get("arguments") or {}
            _trace_event(trace, "tool_call", {
                "name": name,
                "arguments": _trace_tool_args(name, args),
                "source": "model",
            })

            try:
                if name == "run_sql":
                    if not schema_checked:
                        messages.append({
                            "role": "tool",
                            "name": "run_sql",
                            "content": "error: schema required before run_sql",
                        })
                        _trace_event(trace, "tool_error", {
                            "name": "run_sql",
                            "error": "schema required before run_sql",
                            "source": "model",
                        })
                        _workflow_add(workflow, "Blocked SQL execution (schema not checked).")
                        if not schema_prompted:
                            messages.append({
                                "role": "system",
                                "content": (
                                    "Schema required before run_sql. Call get_schema or describe_table "
                                    "for each table you will query, then call run_sql."
                                ),
                            })
                            schema_prompted = True
                        continue
                    column_error = _validate_sql_columns(args["sql"], schema_cache)
                    if column_error:
                        messages.append({
                            "role": "tool",
                            "name": "run_sql",
                            "content": f"error: {column_error}",
                        })
                        _trace_event(trace, "tool_error", {
                            "name": "run_sql",
                            "error": column_error,
                            "source": "model",
                        })
                        if not schema_prompted:
                            messages.append({
                                "role": "system",
                                "content": f"SQL invalid: {column_error}",
                            })
                            schema_prompted = True
                        continue
                    _validate_named_params(args["sql"], args.get("params"))
                    max_rows = args.get("max_rows")
                    if max_rows is not None and not _user_requested_limit(req.message):
                        max_rows = None
                    result = run_sql(
                        sql=args["sql"],
                        params=args.get("params"),
                        max_rows=max_rows,
                    )
                    last_tool_result = result
                    last_tool_name = "run_sql"
                    last_tool_signature = _tool_signature(tool_calls)
                    last_tool_error = None
                    query_saved_to = result.get("saved_to")
                    _log_query(query_log, args["sql"], args.get("params"), "model", result=result)
                    _workflow_add(workflow, "SQL executed.")
                    _workflow_add(workflow, f"Rows returned: {result.get('row_count')}.")
                    _trace_event(trace, "tool_result", {
                        "name": "run_sql",
                        "row_count": result.get("row_count"),
                        "truncated": result.get("truncated"),
                        "saved_to": result.get("saved_to"),
                        "source": "model",
                    })
                    if result.get("row_count", 0) == 0:
                        plot_b64 = None
                        plot_saved_to = None
                        plot_items.clear()

                    messages.append({
                        "role": "tool",
                        "name": "run_sql",
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    })


                elif name == "make_plot":
                    if last_tool_name != "run_sql" or not last_tool_result:
                        messages.append({
                            "role": "tool",
                            "name": "make_plot",
                            "content": "error: no data to plot",
                        })
                        _trace_event(trace, "tool_error", {
                            "name": "make_plot",
                            "error": "no data to plot",
                            "source": "model",
                        })
                        if not plot_requires_sql_prompted:
                            messages.append({
                                "role": "system",
                                "content": "Plot requires data. Call run_sql first, then call make_plot.",
                            })
                            plot_requires_sql_prompted = True
                        continue
                    if last_tool_result.get("row_count", 0) == 0:
                        messages.append({
                            "role": "tool",
                            "name": "make_plot",
                            "content": "error: no rows to plot",
                        })
                        _trace_event(trace, "tool_error", {
                            "name": "make_plot",
                            "error": "no rows to plot",
                            "source": "model",
                        })
                        continue
                    plot_data = args.get("data") or last_tool_result.get("rows") or []
                    result = make_plot(
                        title=args["title"],
                        kind=args["kind"],
                        x_key=args["x_key"],
                        y_key=args["y_key"],
                        data=plot_data,
                        legend_label=args.get("legend_label"),
                    )
                    plot_b64 = result.get("png_base64")
                    plot_saved_to = result.get("saved_to")
                    plot_items.append(_plot_item_from_result(
                        result,
                        title=args["title"],
                        kind=args["kind"],
                        x_key=args["x_key"],
                        y_key=args["y_key"],
                        note=f"Plot: {args['title']}.",
                    ))
                    last_tool_result = result
                    last_tool_name = "make_plot"
                    last_tool_signature = _tool_signature(tool_calls)
                    last_tool_error = None
                    last_tool_name = "make_plot"
                    _workflow_add(workflow, "Plot generated.")
                    _trace_event(trace, "tool_result", {
                        "name": "make_plot",
                        "saved_to": result.get("saved_to"),
                        "source": "model",
                    })
                    messages.append({
                        "role": "tool",
                        "name": "make_plot",
                        "content": "ok",
                    })

                elif name == "get_schema":
                    result = get_schema(
                        table=args.get("table"),
                        max_tables=args.get("max_tables"),
                    )
                    _record_schema(schema_cache, result)
                    schema_checked = bool(schema_cache)
                    last_tool_error = None
                    if result.get("table"):
                        _workflow_add(workflow, f"Schema checked: {result.get('table')}.")
                    _trace_event(trace, "tool_result", {
                        "name": "get_schema",
                        "source": "model",
                        **_trace_schema_result(result),
                    })
                    messages.append({
                        "role": "tool",
                        "name": "get_schema",
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    })

                elif name == "find_join_path":
                    _require_args(args, ["from_table", "to_table"], "find_join_path")

                    result = find_join_path(
                        from_table=args["from_table"],
                        to_table=args["to_table"],
                        max_hops=args.get("max_hops"),
                    )
                    last_tool_error = None
                    _trace_event(trace, "tool_result", {
                        "name": "find_join_path",
                        "hops": result.get("hops"),
                        "error": result.get("error"),
                        "source": "model",
                    })
                    messages.append({
                        "role": "tool",
                        "name": "find_join_path",
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    })

                elif name == "describe_table":
                    _require_args(args, ["table"], "describe_table")
                    result = describe_table(table=args["table"])
                    _record_schema(schema_cache, result)
                    schema_checked = bool(schema_cache)
                    last_tool_error = None
                    _workflow_add(workflow, f"Schema checked: {args['table']}.")
                    _trace_event(trace, "tool_result", {
                        "name": "describe_table",
                        "source": "model",
                        **_trace_schema_result(result),
                    })
                    messages.append({
                        "role": "tool",
                        "name": "describe_table",
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    })

                else:
                    messages.append({
                        "role": "tool",
                        "name": name or "unknown",
                        "content": f"error: unknown tool {name}",
                    })

            except Exception as e:
                err = str(e)
                _trace_event(trace, "tool_error", {
                    "name": name or "unknown",
                    "error": err,
                    "source": "model",
                })
                if name == "run_sql":
                    _log_query(query_log, args.get("sql", ""), args.get("params"), "model", error=err)
                    _workflow_add(workflow, "SQL execution failed.")
                messages.append({
                    "role": "tool",
                    "name": name or "unknown",
                    "content": f"error: {err}",
                })
                last_tool_error = err
                last_tool_name = name or "unknown"

                # Add a corrective system message so the model retries with required args
                messages.append({
                    "role": "system",
                    "content": (
                        f"The previous tool call failed: {name} -> {err}. "
                        "Retry the tool call with all required arguments. "
                        "Do NOT respond with natural language yet."
                    ),
                })

    # If we exit the loop, return whatever we have
    if last_tool_result and last_tool_name:
        plot_b64, plot_note, plot_saved_to = _maybe_autoplot(
            req.message, last_tool_name, last_tool_result, plot_b64, plot_saved_to, plot_items
        )
        text = _format_tool_fallback(
            last_tool_name,
            last_tool_result,
            req.message,
            user_lang,
        )
        if plot_note:
            text = f"{text}\n{plot_note}"
        _workflow_add(workflow, "Answer generated from tool results.")
        return _respond(
            text=text,
            plot_png_base64=plot_b64,
            plot_saved_to=plot_saved_to,
            query_saved_to=query_saved_to,
        )
    if last_tool_error:
        return _respond(
            text=_friendly_tool_error(req.message, last_tool_error, user_lang),
            plot_png_base64=plot_b64,
            plot_saved_to=plot_saved_to,
            query_saved_to=query_saved_to,
        )
    if not last_tool_result and _requires_tool_call(req.message):
        return _respond(
            text=_needs_db_data_message(user_lang),
            plot_png_base64=plot_b64,
            plot_saved_to=plot_saved_to,
            query_saved_to=query_saved_to,
        )
    return _respond(
        text=_t(user_lang, "tool_loop_limit"),
        plot_png_base64=plot_b64,
        plot_saved_to=plot_saved_to,
        query_saved_to=query_saved_to,
    )
