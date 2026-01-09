# test-ai-assistant

FastAPI service that turns natural language questions into read-only SQL queries
and charts against a PostgreSQL database using an Ollama model.

## Features
- Tool-driven flow for schema discovery, SQL execution, and plotting.
- Read-only SQL enforcement with optional hard row limits.
- Optional schema hints loaded from `schema.sql`.
- Saves query results and plots to disk when enabled.

## Requirements
- Python 3.10+
- Ollama server with a tool-capable model
- PostgreSQL database (via SQLAlchemy + psycopg2)

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file:
   ```
   OLLAMA_HOST=http://localhost:11434
   OLLAMA_MODEL=llama3.1:8b
   DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/dbname

   # Optional settings
   MAX_SQL_ROWS=0
   MAX_TOOL_TURNS=10
   SAVE_QUERY_RESULTS=1
   SAVE_PLOTS=1
   QUERY_RESULTS_DIR=query_results
   PLOT_RESULTS_DIR=plot_results
   MAX_SCHEMA_TABLES=50
   MAX_SCHEMA_COLUMNS=200
   MAX_SCHEMA_HOPS=6
   SCHEMA_FILE=schema.sql
   ```

## Run
```
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## API
### `POST /chat`
Request body:
```
{
  "message": "Show average measurements for form ABC this week",
  "history": [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"}
  ]
}
```

Response body (fields may be null depending on the request):
```
{
  "text": "Summary of results...",
  "plot_png_base64": null,
  "plot_saved_to": "plot_results/20250101T120000Z_abcd1234.png",
  "query_saved_to": "query_results/20250101T120000Z_efgh5678.json",
  "trace": [],
  "plots": []
}
```

## Notes
- `MAX_SQL_ROWS=0` disables the hard limit; set a positive value to enforce caps.
- The app blocks non-SELECT SQL statements.
- `schema.sql` is optional and used for faster hints; database introspection is
  still the source of truth.
