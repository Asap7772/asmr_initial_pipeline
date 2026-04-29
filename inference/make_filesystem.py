from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from tqdm import tqdm


SOURCE_FILE = Path("/iris/u/asap7772/asmr_private/decrypted.jsonl")
AGENT_SAVE_DIR = Path("/iris/u/asap7772/asmr_private/data/train")
PRIVILEGED_SAVE_DIR = Path("/iris/u/asap7772/asmr_private/data/train_privileged")

MAX_URL_SLUG_LENGTH = 96
MAX_DOCID_SLUG_LENGTH = 64
HASH_LENGTH = 12
DOC_GROUPS = (
    ("evidence_docs", "evidence_docs"),
    ("gold_docs", "gold_docs"),
    ("negative_docs", "negative_docs"),
)


def _slugify(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return re.sub(r"_+", "_", slug) or "unknown"


def _shorten_slug(slug: str, hash_source: str, max_length: int) -> str:
    if len(slug) <= max_length:
        return slug

    digest = hashlib.sha256(hash_source.encode("utf-8")).hexdigest()[:HASH_LENGTH]
    prefix_length = max_length - HASH_LENGTH - 1
    return f"{slug[:prefix_length].rstrip('_')}_{digest}"


def _sanitize_docid(docid: object) -> str:
    raw_docid = str(docid or "unknown")
    return _shorten_slug(_slugify(raw_docid), raw_docid, MAX_DOCID_SLUG_LENGTH)


def _sanitize_url(url: object, max_length: int = MAX_URL_SLUG_LENGTH) -> str:
    raw_url = str(url or "")
    parsed = urlparse(raw_url)
    parts = [parsed.netloc, parsed.path, parsed.query, parsed.fragment]
    slug = _slugify("_".join(part for part in parts if part))
    return _shorten_slug(slug, raw_url, max_length)


def _doc_filename(doc: dict, used_names: set[str]) -> str:
    docid = _sanitize_docid(doc.get("docid", "unknown"))
    url_slug = _sanitize_url(doc.get("url", ""))
    base_name = f"{docid}_{url_slug}"
    filename = f"{base_name}.txt"

    suffix = 2
    while filename in used_names:
        filename = f"{base_name}_{suffix}.txt"
        suffix += 1

    used_names.add(filename)
    return filename


def _write_text(path: Path, text: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text or ""), encoding="utf-8")


def _remove_stale_agent_metadata(agent_query_dir: Path) -> None:
    for filename in ("query.txt", "answer.txt"):
        stale_path = agent_query_dir / filename
        if stale_path.exists():
            stale_path.unlink()


def main() -> None:
    df = pd.read_json(SOURCE_FILE, lines=True)
    AGENT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    PRIVILEGED_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing queries"):
        query_id = _slugify(row["query_id"])
        agent_query_dir = AGENT_SAVE_DIR / query_id
        privileged_query_dir = PRIVILEGED_SAVE_DIR / query_id
        agent_query_dir.mkdir(parents=True, exist_ok=True)
        privileged_query_dir.mkdir(parents=True, exist_ok=True)
        _remove_stale_agent_metadata(agent_query_dir)

        _write_text(privileged_query_dir / "query.txt", row["query"])
        _write_text(privileged_query_dir / "answer.txt", row["answer"])

        used_agent_names: set[str] = set()
        manifest = {
            "query_id": row["query_id"],
            "query_file": "query.txt",
            "answer_file": "answer.txt",
            "documents": [],
        }

        for source_key, label in DOC_GROUPS:
            label_dir = privileged_query_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)

            for doc in row[source_key]:
                filename = _doc_filename(doc, used_agent_names)
                text = doc.get("text", "")

                _write_text(agent_query_dir / filename, text)
                _write_text(label_dir / filename, text)

                manifest["documents"].append(
                    {
                        "label": label,
                        "docid": doc.get("docid"),
                        "url": doc.get("url"),
                        "agent_file": str(Path(query_id) / filename),
                        "privileged_file": str(Path(query_id) / label / filename),
                    }
                )

        manifest_path = privileged_query_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
