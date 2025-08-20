# wordfreq_mapreduce.py
from __future__ import annotations

import argparse
import string
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import requests


# -------------------------- I/O --------------------------
def get_text(url: str, timeout: int = 20) -> str | None:
    """Завантажує сирий текст з URL. Повертає None у разі помилки."""
    try:
        headers = {"User-Agent": "WordFreq-MapReduce/1.0"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text
    except requests.RequestException:
        return None


# ---------------------- Preprocessing --------------------
# Видалення пунктуації та нормалізація регістру
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def normalize(text: str) -> List[str]:
    """Зводить до нижнього регістру, прибирає пунктуацію, розбиває на слова."""
    # інколи в текстах трапляються типові «сміттєві» символи — почистимо мінімально
    text = text.replace("—", " ").replace("–", " ").replace("…", " ")
    text = text.translate(_PUNCT_TABLE).lower()
    # прибираємо порожні токени
    return [w for w in text.split() if w]


# ----------------------- MapReduce -----------------------
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
    """
    Парадигма MapReduce із ThreadPoolExecutor:
      - Map: (слово -> 1) паралельно
      - Shuffle: групування
      - Reduce: сумування паралельно
    Якщо задано search_words — рахуємо частоти лише цих слів.
    """
    words = normalize(text)
    if search_words:
        sw = {w.lower() for w in search_words}
        words = [w for w in words if w in sw]

    # Мапінг паралельно
    with ThreadPoolExecutor() as ex:
        mapped = list(ex.map(map_function, words))

    # Shuffle
    shuffled = shuffle_function(mapped)

    # Редукція паралельно
    with ThreadPoolExecutor() as ex:
        reduced = dict(ex.map(reduce_function, shuffled))

    return reduced


# -------------------- Visualization ----------------------
def visualize_top_words(freqs: Dict[str, int], top_n: int = 20, title: str = "Top words"):
    """
    Малює bar chart топ-N слів.
    Вимога: matplotlib, один графік, без стилів/кольорів, як просили.
    """
    if not freqs:
        print("Немає даних для візуалізації.")
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
        description="Завантаження тексту за URL, підрахунок частот слів (MapReduce) і візуалізація топ-слів."
    )
    p.add_argument("--url", "-u", required=True, help="URL із текстом (plain text)")
    p.add_argument("--top", "-n", type=int, default=20, help="Скільки топ-слів показати (за замовч. 20)")
    p.add_argument("--filter", "-f", nargs="*", help="Список слів для фільтру (рахувати тільки їх)")
    p.add_argument("--save", "-o", type=Path, help="Куди зберегти частоти у CSV (необов’язково)")
    return p


def save_csv(freqs: Dict[str, int], path: Path) -> None:
    lines = ["word,frequency"] + [f"{w},{c}" for w, c in freqs.items()]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Збережено CSV: {path.resolve()}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    text = get_text(args.url)
    if text is None:
        print("Помилка: не вдалося отримати текст за вказаною URL-адресою.")
        raise SystemExit(1)

    print("DEBUG: довжина тексту =", len(text) if text else "немає тексту")
    print("Перші 500 символів:")
    print(text[:500] if text else "Нічого не завантажилось")

    freqs = map_reduce(text, search_words=args.filter)

    # опціонально — зберегти частоти у CSV
    if args.save:
        save_csv(freqs, args.save)

    title = "Top words" if not args.filter else f"Top filtered words ({', '.join(args.filter)})"
    visualize_top_words(freqs, top_n=args.top, title=title)


if __name__ == "__main__":
    main()
