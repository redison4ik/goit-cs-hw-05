from __future__ import annotations

import argparse
import string
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import requests


def get_text(url: str, timeout: int = 20) -> str | None:
    try:
        headers = {"User-Agent": "WordFreq-MapReduce/1.0"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text
    except requests.RequestException:
        return None


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def normalize(text: str) -> List[str]:
    text = text.replace("—", " ").replace("–", " ").replace("…", " ")
    text = text.translate(_PUNCT_TABLE).lower()
    return [w for w in text.split() if w]


def map_function(word: str) -> Tuple[str, int]:
    return word, 1

def shuffle_function(mapped_values: Iterable[Tuple[str, int]]):
    shuffled: Dict[str, List[int]] = defaultdict(list)
    for k, v in mapped_values:
        shuffled[k].append(v)
    return shuffled.items()

def reduce_function(key_values: Tuple[str, List[int]]) -> Tuple[str, int]:
    key, values = key_values
    return key, sum(values)

def map_reduce(text: str, search_words: Iterable[str] | None = None) -> Dict[str, int]:
    words = normalize(text)
    if search_words:
        sw = {w.lower() for w in search_words}
        words = [w for w in words if w in sw]

    with ThreadPoolExecutor() as ex:
        mapped = list(ex.map(map_function, words))

    shuffled = shuffle_function(mapped)

    with ThreadPoolExecutor() as ex:
        reduced = dict(ex.map(reduce_function, shuffled))

    return reduced

def visualize_top_words(freqs: Dict[str, int], top_n: int = 20, title: str = "Top words"):
    if not freqs:
        print("no data to visualize")
        return

    items = sorted(freqs.items(), key=lambda kv: kv[1], reverse=True)[:max(1, top_n)]
    labels, values = zip(*items)

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# ------------------------- CLI --------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="text upload via URL, words calculation (MapReduce) and visualization."
    )
    p.add_argument("--url", "-u", required=True, help="URL (plain text)")
    p.add_argument("--top", "-n", type=int, default=20, help="number of words (by default 20)")
    p.add_argument("--filter", "-f", nargs="*", help="filter words")
    p.add_argument("--save", "-o", type=Path, help="save to CSV (optional)")
    return p


def save_csv(freqs: Dict[str, int], path: Path) -> None:
    lines = ["word,frequency"] + [f"{w},{c}" for w, c in freqs.items()]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"CSV saved: {path.resolve()}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    text = get_text(args.url)
    if text is None:
        print("Error: no text loaded from the URL")
        raise SystemExit(1)

    print("DEBUG: lenth =", len(text) if text else "no text")
    print("first 500 symbols:")
    print(text[:500] if text else "Upload failed")

    freqs = map_reduce(text, search_words=args.filter)

    if args.save:
        save_csv(freqs, args.save)

    title = "Top words" if not args.filter else f"Top filtered words ({', '.join(args.filter)})"
    visualize_top_words(freqs, top_n=args.top, title=title)


if __name__ == "__main__":
    main()
