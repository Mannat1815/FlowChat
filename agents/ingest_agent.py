# agents/ingest_agent.py
import os
import hashlib
import time
import xarray as xr
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from agents.blockchain_agent import log_ingestion_event
# Used to push RAG-ready chunks into the FAISS vector store
from agents.indexing_agent import index_records

load_dotenv()

# Postgres connection: prefer POSTGRES_*, fallback to MYSQL_* from your existing .env
PG_HOST = os.getenv("POSTGRES_HOST") or os.getenv("MYSQL_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT") or os.getenv("MYSQL_PORT", "5432")
PG_USER = os.getenv("POSTGRES_USER") or os.getenv("MYSQL_USER", "postgres")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD") or os.getenv("MYSQL_PASSWORD", "")
PG_DB = os.getenv("POSTGRES_DB") or os.getenv("MYSQL_DATABASE") or "argo_db"

PG_URL = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
engine = create_engine(PG_URL, pool_pre_ping=True)

TABLE_NAME = "argo_profiles"  #for 1d
MEASUREMENTS_TABLE = "argo_measurements"  #for 2d

# RAG ingestion settings
# - AUTO_INDEX: when true, automatically create+index text chunks after DB insert
# - INGEST_CHUNK_SIZE: target rows per chunk window when summarizing
# - INGEST_MAX_CHUNKS: cap total chunks per ingest to bound latency/storage
AUTO_INDEX = (os.getenv("AUTO_INDEX", "true").lower() == "true")
INGEST_CHUNK_SIZE = int(os.getenv("INGEST_CHUNK_SIZE", "50"))
INGEST_MAX_CHUNKS = int(os.getenv("INGEST_MAX_CHUNKS", "200"))

# Optional explode of N-D arrays into a normalized long-form table
AUTO_EXPLODE_ND = (os.getenv("AUTO_EXPLODE_ND", "true").lower() == "true")
ND_TABLE = "argo_nd_values"  #for n-D
MAX_ND_DIMS = int(os.getenv("MAX_ND_DIMS", "5"))
MAX_ND_ROWS = int(os.getenv("MAX_ND_ROWS", "500000"))

# checks argo_profiles
def ensure_table_exists():
    # Create a simple flexible table if it doesn't exist. Columns will be added as needed.
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id SERIAL PRIMARY KEY,
                source_file TEXT,
                ingest_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        # In case the table already existed without these columns, ensure they are present
        conn.execute(text(f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS source_file TEXT;"))
        conn.execute(text(f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS ingest_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP;"))

#checks argo_measurements
def ensure_measurements_table():
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {MEASUREMENTS_TABLE} (
                id SERIAL PRIMARY KEY,
                source_file TEXT,
                profile_id INTEGER REFERENCES {TABLE_NAME}(id),
                level_index INTEGER,
                pres DOUBLE PRECISION,
                pres_adjusted DOUBLE PRECISION,
                temp DOUBLE PRECISION,
                temp_adjusted DOUBLE PRECISION,
                psal DOUBLE PRECISION,
                psal_adjusted DOUBLE PRECISION,
                ingest_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

def add_missing_columns(df_columns):
    with engine.begin() as conn:
        # get current columns
        res = conn.execute(text(f"""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = :table
        """), {"table": TABLE_NAME})
        # normalize to lowercase for case-insensitive comparison
        existing = {row[0].lower() for row in res.fetchall()}

        for col in df_columns:
            # sanitize column name
            col_name = "".join(c if (c.isalnum() or c == "_") else "_" for c in col)
            if col_name.lower() not in existing:
                # use double precision for numeric fields
                try:
                    conn.execute(text(f'ALTER TABLE {TABLE_NAME} ADD COLUMN "{col_name}" DOUBLE PRECISION;'))
                    print(f"Added column {col_name}")
                except Exception as e:
                    print(f"Failed to add column {col_name}: {e}")

def load_netcdf(file_path):
    """Parse NetCDF similarly to your original version, returning DataFrame."""
    data = {}
    try:
        with Dataset(file_path, "r") as ds:
            for var_name, var_obj in ds.variables.items():
                # skip QC/metadata-ish variables
                if any(x in var_name.lower() for x in ["_qc", "_adjusted", "_error", "history", "crs"]):
                    continue

                # numeric and 1D
                try:
                    if getattr(var_obj, 'ndim', 0) == 1 and np.issubdtype(var_obj.dtype, np.number):
                        data[var_name] = var_obj[:].data.tolist()
                    elif getattr(var_obj, 'ndim', 0) == 0 and np.issubdtype(var_obj.dtype, np.number):
                        data[var_name] = [float(var_obj[:].item())]
                except Exception:
                    # skip variables we can't convert
                    continue

            if not data:
                return pd.DataFrame(), "No valid numeric 1D variables found."

            lengths = [len(v) for v in data.values()]
            max_len = max(lengths)
            for k in data:
                if len(data[k]) < max_len:
                    data[k] = list(data[k]) + [None] * (max_len - len(data[k]))

            df = pd.DataFrame(data)
            return df, "Success"
    except Exception as e:
        return pd.DataFrame(), f"Failed to parse NetCDF: {e}"

def insert_dataframe(df, source_file):
    if df.empty:
        return False, "Empty dataframe", {}

    ensure_table_exists()
    add_missing_columns(df.columns)

    df_cols = [c for c in df.columns]
    cols_sql = ', '.join(f'"{c}"' for c in df_cols)
    placeholders = ', '.join(f":{c}" for c in df_cols)
    insert_sql = text(f'INSERT INTO {TABLE_NAME} (source_file, {cols_sql}) VALUES (:source_file, {placeholders})')

    rows_to_insert = []
    for _, row in df.iterrows():
        row_dict = {c: _safe_float(row[c]) for c in df_cols}
        row_dict["source_file"] = source_file
        rows_to_insert.append(row_dict)

    with engine.begin() as conn:
        for i in range(0, len(rows_to_insert), 5000):
            batch = rows_to_insert[i:i+5000]
            conn.execute(insert_sql, batch)

    # Fetch inserted IDs for mapping
    with engine.begin() as conn:
        res = conn.execute(
            text(f"SELECT id, row_number() OVER (ORDER BY id) - 1 AS profile_index FROM {TABLE_NAME} WHERE source_file=:sf"),
            {"sf": source_file}
        ).fetchall()
        profile_map = {r.profile_index: r.id for r in res}

    return True, f"Inserted {len(rows_to_insert)} rows", profile_map

def insert_measurements_2d(ds, source_file, profile_map):
    ensure_measurements_table()
    two_d_vars = ["pres","pres_adjusted","temp","temp_adjusted","psal","psal_adjusted"]
    # Skip if 2D vars are missing
    if not all(v in ds.variables for v in two_d_vars):
        return 0

    n_profiles = ds["pres"].shape[0]
    rows = []
    for i in range(n_profiles):
        profile_id = profile_map.get(i)  # map 0-based index
        if profile_id is None:
            continue
        for level in range(ds["pres"].shape[1]):
            row = {"source_file": source_file, "profile_id": profile_id, "level_index": level}
            for var in two_d_vars:
                try:
                    row[var] = float(ds[var][i, level])
                except Exception:
                    row[var] = None
            rows.append(row)

    # Bulk insert
    if not rows:
        return 0
    cols = ["source_file","profile_id","level_index"] + two_d_vars
    placeholders = ", ".join(f":{c}" for c in cols)
    insert_sql = text(f"INSERT INTO {MEASUREMENTS_TABLE} ({', '.join(cols)}) VALUES ({placeholders})")
    batch = 5000
    inserted = 0
    with engine.begin() as conn:
        for i in range(0, len(rows), batch):
            conn.execute(insert_sql, rows[i:i+batch])
            inserted += len(rows[i:i+batch])
    return inserted

def _safe_float(v):
    """Convert value to float or return None when not possible/NaN."""
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except Exception:
        return None

def _summ_stats(series):
    """Compute min/max/mean over a numeric pandas Series, skipping nulls/NaNs."""
    vals = [x for x in (series.tolist() if hasattr(series, 'tolist') else list(series)) if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not vals:
        return None
    try:
        arr = np.array(vals, dtype=float)
        return {
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr))
        }
    except Exception:
        return None

def _format_stats(name, stats):
    """Pretty-print a stats dict for inclusion in chunk text."""
    if not stats:
        return None
    return f"{name}: min {stats['min']:.3f}, max {stats['max']:.3f}, mean {stats['mean']:.3f}"

def build_text_chunks(df, source_file, chunk_size=INGEST_CHUNK_SIZE, max_chunks=INGEST_MAX_CHUNKS):
    """
    Convert numeric table rows into coarse summaries suitable for RAG retrieval.
    Produces aggregated window summaries with basic stats for key variables.
    Returns list[{'id': str, 'text': str, 'source': str}].
    """
    if df.empty:
        return []

    # Identify commonly relevant columns if present (best-effort aliases)
    cols = set(df.columns)
    col_lat = next((c for c in ["latitude", "lat", "LATITUDE", "Lat"] if c in cols), None)
    col_lon = next((c for c in ["longitude", "lon", "LONGITUDE", "Lon"] if c in cols), None)
    col_depth = next((c for c in ["depth", "pres", "pressure", "DEPTH", "PRES"] if c in cols), None)
    col_temp = next((c for c in ["temperature", "temp", "TEMP", "TEMP_ADJUSTED"] if c in cols), None)
    col_salt = next((c for c in ["salinity", "psal", "PSAL", "sal"] if c in cols), None)
    col_time = next((c for c in ["time", "juld", "JULD", "date", "datetime"] if c in cols), None)

    chunks = []
    total_rows = len(df)
    # Compute number of chunks, then rows per chunk, bounded by max_chunks
    num_chunks = min(max(1, (total_rows + chunk_size - 1) // chunk_size), max_chunks)
    rows_per_chunk = max(1, total_rows // num_chunks)

    for i in range(0, total_rows, rows_per_chunk):
        if len(chunks) >= max_chunks:
            break
        window = df.iloc[i:i+rows_per_chunk]

        lines = [f"Source file: {source_file}"]
        lines.append(f"Rows: {len(window)} (subset of {total_rows})")

        # Summaries for key signals
        for name, col in [("Latitude", col_lat), ("Longitude", col_lon), ("Depth", col_depth), ("Temperature", col_temp), ("Salinity", col_salt)]:
            if col:
                s = _summ_stats(window[col])
                formatted = _format_stats(name, s)
                if formatted:
                    lines.append(formatted)

        # Time range if present
        if col_time and col_time in window:
            try:
                tvals = [v for v in window[col_time].tolist() if v is not None]
                if tvals:
                    t_min = min(tvals)
                    t_max = max(tvals)
                    lines.append(f"Time range: {t_min} to {t_max}")
            except Exception:
                pass

        # Mention available numeric columns for discoverability
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if num_cols:
            lines.append("Numeric columns: " + ", ".join(num_cols[:20]) + (" ..." if len(num_cols) > 20 else ""))

        text = "\n".join(lines)
        chunk_id = f"{source_file}:{i}-{i+len(window)-1}"
        chunks.append({
            "id": chunk_id,
            "text": text,
            "source": source_file
        })

    return chunks

def ingest_file(file_path):
    print("Ingesting", file_path)
    df, msg = load_netcdf(file_path)
    if df.empty:
        # If no 1D numeric variables were parsed, still try ND explode
        if AUTO_EXPLODE_ND:
            try:
                inserted = explode_nd_to_long(file_path, os.path.basename(file_path))
                if inserted > 0:
                    # log to "blockchain" simulator
                    log_ingestion_event(file_path)
                    print(f"ND explode rows inserted (no 1D data): {inserted}")
                    return True, f"ND rows inserted: {inserted}"
            except Exception as e:
                print(f"ND explode failed on empty 1D parse: {e}")
        return False, msg

    success, imsg, profile_map = insert_dataframe(df, os.path.basename(file_path))
    if success:
        # log to "blockchain" simulator
        log_ingestion_event(file_path)

        # Insert 2D measurements linked to profiles
        with Dataset(file_path, "r") as ds:
            inserted_2d = insert_measurements_2d(ds, os.path.basename(file_path), profile_map) 
            print(f"Inserted {inserted_2d} 2D measurement rows")

        # Optionally explode N-D numeric arrays into normalized long-form table
        if AUTO_EXPLODE_ND:
            try:
                inserted = explode_nd_to_long(file_path, os.path.basename(file_path))
                print(f"ND explode rows inserted: {inserted}")
            except Exception as e:
                print(f"ND explode failed: {e}")
        # Auto-index for RAG if enabled
        if AUTO_INDEX:
            try:
                # Summarize rows into human-readable context chunks
                chunks = build_text_chunks(df, os.path.basename(file_path))
                if chunks:
                    # Push chunks into FAISS via indexing_agent
                    indexed = index_records(chunks)
                    print(f"Indexed {len(chunks)} chunks for {os.path.basename(file_path)} (success={indexed})")
                else:
                    print("No chunks produced for indexing.")
            except Exception as e:
                print(f"Indexing failed: {e}")
        return True, imsg
    return False, imsg

def ensure_nd_table_exists():
    """Create normalized ND values table if missing (supports up to MAX_ND_DIMS)."""
    dim_cols = ", ".join([f"dim{i} INTEGER" for i in range(MAX_ND_DIMS)])
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {ND_TABLE} (
                id SERIAL PRIMARY KEY,
                source_file TEXT,
                var_name TEXT,
                value DOUBLE PRECISION,
                {dim_cols},
                ingest_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

def _pick_first_existing(ds, candidates):
    for name in candidates:
        if name in ds.variables:
            return name
    return None

def _to_numpy(var):
    try:
        arr = var[:]
        # Handle masked arrays
        try:
            import numpy.ma as ma
            if isinstance(arr, ma.MaskedArray):
                arr = arr.filled(np.nan)
        except Exception:
            pass
        return np.array(arr)
    except Exception:
        return None

def explode_nd_to_long(file_path, source_file):
    ensure_nd_table_exists()
    rows = []
    total = 0
    try:
        from netCDF4 import Dataset
        with Dataset(file_path, 'r') as ds:
            for var_name, var_obj in ds.variables.items():
                # skip QC/metadata-ish variables
                if any(x in var_name.lower() for x in ["_qc", "_error", "history", "crs"]):
                    continue
                try:
                    ndim = getattr(var_obj, 'ndim', 0)
                    if ndim < 1 or ndim > MAX_ND_DIMS:
                        continue
                    if not np.issubdtype(var_obj.dtype, np.number):
                        continue
                except Exception:
                    continue

                arr = _to_numpy(var_obj)
                if arr is None:
                    continue

                # Debug log (safe now)
                print("ND var include:", var_name, ndim, var_obj.dtype, arr.shape)

                for idx in np.ndindex(arr.shape):
                    v = arr[idx]
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if np.isnan(fv):
                        continue
                    dims = list(idx) + [None] * (MAX_ND_DIMS - len(idx))
                    rows.append({
                        "source_file": source_file,
                        "var_name": var_name,
                        "value": fv,
                        **{f"dim{i}": int(dims[i]) if dims[i] is not None else None for i in range(MAX_ND_DIMS)}
                    })
                    total += 1
                    if total >= MAX_ND_ROWS:
                        break
                if total >= MAX_ND_ROWS:
                    break
    except Exception as e:
        raise e
    if not rows:
        return 0

    # Bulk insert in batches to avoid parameter limits
    dim_cols = [f"dim{i}" for i in range(MAX_ND_DIMS)]
    cols = ["source_file", "var_name", "value"] + dim_cols
    placeholders = ", ".join(f":{c}" for c in cols)
    insert_sql = text(f"INSERT INTO {ND_TABLE} ({', '.join(cols)}) VALUES ({placeholders})")
    batch = 5000
    inserted = 0
    with engine.begin() as conn:
        for i in range(0, len(rows), batch):
            chunk = rows[i:i+batch]
            conn.execute(insert_sql, chunk)
            inserted += len(chunk)
    return inserted