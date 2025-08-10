"""
- Asks for the JSONL file path (supports drag & drop).
- Asks for Qdrant host; defaults to http://localhost:6333.
- Lists existing collections and lets you pick one.
- Validates the selected collection and (best effort) the vector dimension.
- Streams points in batches with a progress bar.
- Clear, colorized messages and helpful errors.
    Expected JSONL format per line:
    {"id":"<uuid or string>", "vector":[...floats...], "payload":{...original row...}}
    This matches the output of Convector.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

from colorama import Fore, Style, init as color_init
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from tqdm import tqdm

DEFAULT_QDRANT_URL: str = "http://localhost:6333"
BATCH_SIZE: int = 100          # Adjust if you want larger/smaller upserts
DEFAULT_INPUT_FILE: str = "output.jsonl"  # Just a hint in prompts
EXPECTED_DIM: Optional[int] = 384         # Best-effort check; set None to skip

# Stable UUID namespace for generated IDs (only used if an item has no id)
ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "qdrantimport.local")

# Initialize terminal colors cross-platform
color_init(autoreset=True)


# --------------------------------- Data Model ---------------------------------

@dataclass(frozen=True)
class Settings:
    qdrant_url: str = DEFAULT_QDRANT_URL
    batch_size: int = BATCH_SIZE
    expected_dim: Optional[int] = EXPECTED_DIM


# ------------------------------- Helper Functions ------------------------------

def _clean_dropped_path(raw: str) -> str:
    """Trim quotes some terminals add when drag-dropping a file path."""
    raw = raw.strip()
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    return raw


def ask_for_jsonl_path() -> Path:
    """Ask user for the JSONL file path and validate it exists."""
    print(Fore.GREEN + f"Please paste your JSONL path (e.g., {DEFAULT_INPUT_FILE}) and press Enter:")
    raw = input().strip()
    path_str = _clean_dropped_path(raw)
    if not path_str:
        print(Fore.RED + "No file provided. Exiting.")
        sys.exit(1)

    path = Path(path_str)
    if not path.is_file():
        print(Fore.RED + f"File not found: {path}")
        print(Fore.YELLOW + "Tip: Put the JSONL in this folder or provide an absolute path.")
        sys.exit(1)
    if path.suffix.lower() != ".jsonl":
        print(Fore.RED + "The selected file is not a .jsonl. Please choose a JSONL file.")
        sys.exit(1)
    return path


def ask_for_qdrant_url(default_url: str = DEFAULT_QDRANT_URL) -> str:
    """Qdrant URL; default to http://localhost:6333."""
    print(Fore.YELLOW + f"Enter your Qdrant URL (press Enter for default: {default_url}):")
    raw = input().strip()
    if not raw:
        return default_url
    return raw


def make_qdrant_client(url: str) -> QdrantClient:
    """
    Build QdrantClient accepting either a full URL or host:port.
    qdrant-client prefers `url="http://host:port"`.
    """
    try:
        if url.startswith("http://") or url.startswith("https://"):
            return QdrantClient(url=url)
        # Accept "localhost:6333" style
        if ":" in url:
            host, port_str = url.split(":", 1)
            return QdrantClient(host=host, port=int(port_str))
        # Accept "localhost" → default port 6333
        return QdrantClient(host=url, port=6333)
    except Exception as e:
        print(Fore.RED + f"Failed to create Qdrant client: {e}")
        sys.exit(1)


def ensure_qdrant_is_up(client: QdrantClient) -> None:
    """Ping Qdrant by listing collections."""
    try:
        resp = client.get_collections()
        # CollectionsResponse object; if no exception, we are good
        if resp is None:
            print(Fore.RED + "Could not query collections. Is Qdrant running?")
            sys.exit(1)
        print(Fore.GREEN + "Qdrant is reachable ✔" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Connection error: {e}")
        print(Fore.YELLOW + "Make sure Qdrant is running, e.g.:")
        print(Fore.WHITE + "docker run -p 6333:6333 --name qdrant --rm qdrant/qdrant")
        sys.exit(1)


def list_collections(client: QdrantClient) -> List[str]:
    """Return a list of collection names and print them enumerated."""
    try:
        resp = client.get_collections()
        # resp.collections is a list[CollectionDescription], each has .name
        items = getattr(resp, "collections", None) or []
        names = [getattr(c, "name", None) for c in items if getattr(c, "name", None)]
        if not names:
            print(Fore.RED + "No collections found in Qdrant.")
            return []
        print(Fore.YELLOW + "Available collections:" + Style.RESET_ALL)
        for idx, name in enumerate(names, start=1):
            print(Fore.CYAN + f"{idx}: {name}" + Style.RESET_ALL)
        return names
    except Exception as e:
        print(Fore.RED + f"Error listing collections: {e}")
        sys.exit(1)


def ask_for_collection_name(names: List[str]) -> str:
    """Let user select a collection by index."""
    print(Fore.YELLOW + "\nType the number of the collection to import into:" + Style.RESET_ALL)
    try:
        sel = int(input().strip())
        if sel < 1 or sel > len(names):
            print(Fore.RED + "Invalid collection number.")
            sys.exit(1)
        chosen = names[sel - 1]
        print(Fore.GREEN + f"\nYou selected: {chosen}. Proceed? (Y/N)" + Style.RESET_ALL)
        confirm = input().strip().lower()
        if confirm not in {"y", "yes", "s", "sim"}:
            print(Fore.RED + "Canceled by user.")
            sys.exit(0)
        return chosen
    except ValueError:
        print(Fore.RED + "Invalid input (not a number).")
        sys.exit(1)


def get_collection_vector_size(client: QdrantClient, collection: str) -> Optional[int]:
    """
    Best-effort: fetch collection config and extract vector size.
    Works for single-vector collections.
    """
    try:
        info = client.get_collection(collection_name=collection)
        # qdrant-client returns different shapes depending on version.
        # Try common paths:
        # info.config.params.vectors.size  OR
        # info.config.params.vectors["size"] for dict-like
        vectors = None
        if hasattr(info, "config") and hasattr(info.config, "params"):
            vectors = getattr(info.config.params, "vectors", None)
        if vectors is None and hasattr(info, "result"):
            # Older clients sometimes nest under .result
            cfg = getattr(info.result, "config", None)
            if cfg and hasattr(cfg, "params"):
                vectors = getattr(cfg.params, "vectors", None)

        if vectors is None:
            return None

        # Single vector config can be: {"size": 384, "distance": "Cosine"} OR a model with .size
        size = getattr(vectors, "size", None)
        if size is not None:
            return int(size)

        if isinstance(vectors, dict) and "size" in vectors:
            return int(vectors["size"])

        # If it's a dict of named vectors, skip strict checking (advanced use)
        return None
    except Exception:
        return None


def generate_stable_id(record: dict, index: int) -> str:
    """Generate a stable UUIDv5 if no 'id' is present."""
    key = json.dumps({"i": index, "p": record.get("payload"), "v": record.get("vector")}, sort_keys=True, ensure_ascii=False)
    return str(uuid.uuid5(ID_NAMESPACE, key))


def count_lines(path: Path) -> int:
    """Count lines in a file for progress bar total."""
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def iter_points_from_jsonl(path: Path) -> Iterable[PointStruct]:
    """
    Accepts per-line objects containing at minimum: vector (list[float]) and payload (dict).
    Uses existing 'id' if present; otherwise generates a stable id.
    """
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Be tolerant with key variations: 'vector' or 'embedding'
            vector = obj.get("vector", obj.get("embedding"))
            payload = obj.get("payload", {})

            if vector is None:
                raise ValueError(f"Missing 'vector' for item at line {idx+1}")

            pid = obj.get("id")
            if pid is None:
                pid = generate_stable_id(obj, idx)

            yield PointStruct(id=pid, vector=vector, payload=payload)


def upsert_in_batches(
    client: QdrantClient,
    collection: str,
    points_iter: Iterable[PointStruct],
    total: int,
    batch_size: int,
) -> Tuple[int, Optional[str]]:
    """
    Upsert points in batches. Returns (imported_count, error_message_if_any).
    """
    batch: List[PointStruct] = []
    imported = 0
    try:
        with tqdm(total=total, desc="Importing", unit="item") as pbar:
            for p in points_iter:
                batch.append(p)
                if len(batch) >= batch_size:
                    client.upsert(collection_name=collection, points=batch)
                    imported += len(batch)
                    pbar.update(len(batch))
                    batch.clear()
            if batch:
                client.upsert(collection_name=collection, points=batch)
                imported += len(batch)
                pbar.update(len(batch))
                batch.clear()
        return imported, None
    except Exception as e:
        return imported, str(e)


# -------------------------------------- Main --------------------------------------

def main(settings: Settings = Settings()) -> None:
    # 1) Ask for JSONL file
    jsonl_path = ask_for_jsonl_path()

    # 2) Ask for Qdrant URL (default localhost)
    qdrant_url = ask_for_qdrant_url(settings.qdrant_url)

    # 3) Build client and ensure Qdrant is reachable
    client = make_qdrant_client(qdrant_url)
    ensure_qdrant_is_up(client)

    # 4) List collections and let user choose one
    names = list_collections(client)
    if not names:
        print(Fore.RED + "No collections available to import into.")
        sys.exit(1)
    collection = ask_for_collection_name(names)

    # 5) (Optional) best-effort dimension check
    coll_dim = get_collection_vector_size(client, collection)
    if coll_dim is not None and settings.expected_dim is not None:
        if coll_dim != settings.expected_dim:
            print(
                Fore.RED
                + f"\nVector dimension mismatch: collection '{collection}' expects {coll_dim}, "
                  f"but you configured {settings.expected_dim}."
            )
            print(Fore.YELLOW + "Tip: Recreate the collection with the correct size (e.g., 384) or change your embedding model.")
            sys.exit(1)

    # 6) Count lines for progress bar total
    print(Fore.YELLOW + f"\nLoading file: {jsonl_path}")
    try:
        total_lines = count_lines(jsonl_path)
        if total_lines == 0:
            print(Fore.RED + "The JSONL file is empty.")
            sys.exit(1)
    except Exception as e:
        print(Fore.RED + f"Failed to read file: {e}")
        sys.exit(1)

    # 7) Iterate points and import
    try:
        points_iter = iter_points_from_jsonl(jsonl_path)
    except Exception as e:
        print(Fore.RED + f"Invalid JSONL format: {e}")
        sys.exit(1)

    print(Fore.YELLOW + f"Starting import into collection: {collection}")
    imported, err = upsert_in_batches(
        client=client,
        collection=collection,
        points_iter=points_iter,
        total=total_lines,
        batch_size=settings.batch_size,
    )

    # 8) Wrap-up
    if err is None:
        print(Fore.GREEN + "\n✅ Import completed successfully!")
        print(Fore.GREEN + f"Total items imported: {imported}")
    else:
        print(Fore.RED + "\n❌ Import failed.")
        print(Fore.RED + f"Imported before error: {imported}")
        print(Fore.RED + f"Error: {err}")

    print(Fore.GREEN + "Press Enter to exit.")
    input()


if __name__ == "__main__":
    main()