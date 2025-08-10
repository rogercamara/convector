#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
- Ask for CSV path (supports drag & drop).
- Auto-detect columns; no code edits needed.
- Build one text per row and embed with a free 384dim model.
- Stream write newline delimited JSON to ./output.jsonl (always).
- Batch embeddings for speed; graceful errors; friendly prompts.
Note: The default model outputs 384dim vectors. Make sure your vector DB
collection uses the same dimension (e.g., Qdrant size=384).
"""

from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd
from colorama import Fore, init as color_init
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME: str = "paraphrase-multilingual-MiniLM-L12-v2"  # 384-dim model
EXPECTED_DIM: int = 384
BATCH_SIZE: int = 64
OUTPUT_FILENAME: str = "output.jsonl"
NORMALIZE_EMBEDDINGS: bool = True  # alinhado ao uso típico com cosine-sim

# Stable UUID namespace for deterministic IDs (do not change in the same project)
ID_NAMESPACE: uuid.UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "convector.local")

# Initialize color output across platforms
color_init(autoreset=True)

# --------------------------------- Data Structures ---------------------------------

@dataclass(frozen=True)
class Settings:
    model_name: str = MODEL_NAME
    expected_dim: int = EXPECTED_DIM
    batch_size: int = BATCH_SIZE
    output_filename: str = OUTPUT_FILENAME
    normalize_embeddings: bool = NORMALIZE_EMBEDDINGS

# -------------------------------- Helper Functions ---------------------------------

def _clean_dropped_path(raw: str) -> str:
    raw = raw.strip()
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    return raw

def ask_for_csv_path() -> Path:
    print(Fore.GREEN + "Please paste your CSV path (or drag & drop it here) and press Enter:")
    raw = input().strip()
    path_str = _clean_dropped_path(raw)
    if not path_str:
        print(Fore.RED + "No file provided. Exiting.")
        sys.exit(1)

    path = Path(path_str)
    if not path.is_file():
        print(Fore.RED + f"File not found: {path}")
        print(Fore.YELLOW + "Tip: put the CSV in this folder or provide an absolute path.")
        sys.exit(1)
    if path.suffix.lower() != ".csv":
        print(Fore.RED + "The selected file is not a .csv. Please choose a CSV file.")
        sys.exit(1)
    return path

def load_csv(path: Path) -> pd.DataFrame:
    print(Fore.YELLOW + f"Loading CSV: {path}")
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            try:
                df = pd.read_csv(path, sep=None, engine="python", encoding="latin-1")
            except Exception as e:
                print(Fore.RED + f"Error loading CSV: {e}")
                sys.exit(1)
    if df.empty:
        print(Fore.RED + "The CSV is empty. Nothing to convert.")
        sys.exit(1)
    return df

def preview_columns_and_confirm(df: pd.DataFrame) -> None:
    cols = list(map(str, df.columns))
    print(Fore.CYAN + "\nDetected columns:")
    print(Fore.CYAN + ", ".join(cols))
    print(Fore.GREEN + "\nThese columns will be concatenated per row and embedded with a 384‑dim model.")
    print(Fore.GREEN + "Output will be saved as: " + Fore.WHITE + OUTPUT_FILENAME)
    print(Fore.YELLOW + "\nContinue? (Y/N)")
    choice = input().strip().lower()
    if choice not in {"y", "yes", "s", "sim"}:
        print(Fore.RED + "Canceled by user.")
        sys.exit(0)

def row_to_text(row: pd.Series) -> str:
    parts: List[str] = []
    for val in row.values:
        if pd.isna(val):
            continue
        s = str(val).strip()
        if s:
            parts.append(s)
    return " ".join(parts)

def df_to_texts(df: pd.DataFrame) -> List[str]:
    return [row_to_text(row) for _, row in df.iterrows()]

def make_stable_id(payload: dict, idx: int) -> str:
    key = json.dumps({"i": idx, "p": payload}, sort_keys=True, ensure_ascii=False)
    return str(uuid.uuid5(ID_NAMESPACE, key))

def detect_device() -> str:
    try:
        import torch  # noqa
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def embed_in_batches(
    model: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int,
    normalize: bool,
) -> Iterable[Tuple[int, List[float]]]:
    total = len(texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = texts[start:end]
        vecs = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
        )
        if hasattr(vecs, "tolist"):
            vecs = vecs.tolist()
        for i, vec in enumerate(vecs, start=start):
            yield i, vec

def write_jsonl_stream(
    df: pd.DataFrame,
    texts: Sequence[str],
    vectors_iter: Iterable[Tuple[int, List[float]]],
    out_path: Path,
    expected_dim: int,
) -> int:
    tmp_path = out_path.with_suffix(".jsonl.tmp")
    written = 0
    try:
        with tmp_path.open("w", encoding="utf-8") as fout, tqdm(
            total=len(texts), desc="Converting", unit="row"
        ) as pbar:
            for abs_idx, vec in vectors_iter:
                if len(vec) != expected_dim:
                    print(
                        Fore.RED
                        + f"\nUnexpected vector size at row {abs_idx}: {len(vec)} (expected {expected_dim}). Skipping."
                    )
                    pbar.update(1)
                    continue

                payload = df.iloc[abs_idx].to_dict()
                point = {
                    "id": make_stable_id(payload, abs_idx),
                    "text": texts[abs_idx],         # <<< incluído
                    "vector": vec,
                    "payload": payload,
                }
                fout.write(json.dumps(point, ensure_ascii=False) + "\n")
                written += 1
                pbar.update(1)

        tmp_path.replace(out_path)
    except Exception as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        print(Fore.RED + f"\nFailed to write output: {e}")
        sys.exit(1)

    return written

# -------------------------------------- Main --------------------------------------

def main(settings: Settings = Settings()) -> None:
    # 1) Get CSV path from user
    csv_path = ask_for_csv_path()

    # 2) Load CSV
    df = load_csv(csv_path)

    # 3) Preview + confirm
    preview_columns_and_confirm(df)

    # 4) Prepare texts
    texts = df_to_texts(df)

    # 5) Load embedding model
    device = detect_device()
    print(Fore.YELLOW + f"\nLoading embedding model: {settings.model_name} ({settings.expected_dim}-dim)")
    print(Fore.YELLOW + f"Device: {device.upper()} | normalize_embeddings={settings.normalize_embeddings}")
    try:
        model = SentenceTransformer(settings.model_name)
        try:
            model = model.to(device)
        except Exception:
            pass  # versões antigas já sobem no device correto
    except Exception as e:
        print(Fore.RED + f"Failed to load embedding model: {e}")
        sys.exit(1)

    # 6) Prepare output path (always ./output.jsonl)
    out_path = Path.cwd() / settings.output_filename
    print(Fore.YELLOW + f"\nStarting conversion. Writing to: {out_path}")

    # 7) Embed in batches and stream-write JSONL
    vectors_iter = embed_in_batches(model, texts, settings.batch_size, settings.normalize_embeddings)
    written = write_jsonl_stream(df, texts, vectors_iter, out_path, settings.expected_dim)

    # 8) Wrap-up
    print(Fore.GREEN + "\n✅ Conversion finished!")
    print(Fore.GREEN + f"Total rows processed: {len(df)}")
    print(Fore.GREEN + f"Total records written: {written}")
    print(Fore.GREEN + f"Output file: {out_path}")
    print(Fore.GREEN + "Press Enter to exit.")
    input()

if __name__ == "__main__":
    main()