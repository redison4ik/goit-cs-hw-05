# sorter_async.py
from __future__ import annotations
import asyncio
import argparse
import logging
from typing import AsyncIterator
from aiopath import AsyncPath
import aiofiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("sorter.log", encoding="utf-8"),
        logging.StreamHandler()
    ],
)
log = logging.getLogger("sorter")


async def awalk(root: AsyncPath) -> AsyncIterator[AsyncPath]:
    try:
        async for entry in root.iterdir():
            try:
                if await entry.is_dir():
                    async for sub in awalk(entry):
                        yield sub
                elif await entry.is_file():
                    yield entry
            except Exception as e:
                log.error("access error %s: %s", entry, e)
    except FileNotFoundError:
        log.error("folder %s not exist", root)
    except PermissionError as e:
        log.error("no access %s: %s", root, e)


async def copy_file(src: AsyncPath, out_root: AsyncPath, sem: asyncio.Semaphore, chunk: int = 1 << 20) -> None:
    
    ext = (src.suffix.lower().lstrip(".") or "_noext")
    dest_dir = out_root / ext
    dest = dest_dir / src.name

    async with sem:  
        try:
            if not await dest_dir.exists():
                await dest_dir.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(src, "rb") as rf, aiofiles.open(dest, "wb") as wf:
                while True:
                    data = await rf.read(chunk)
                    if not data:
                        break
                    await wf.write(data)

            log.info("OK: %s -> %s", src, dest)
        except Exception as e:
            log.exception("copy error %s -> %s: %s", src, dest, e)


async def read_folder(src_root: AsyncPath, out_root: AsyncPath, limit: int = 64) -> None:
    
    if not await src_root.exists():
        log.error("folder not exist: %s", src_root)
        return
    if not await src_root.is_dir():
        log.error("not a folder: %s", src_root)
        return

    if not await out_root.exists():
        await out_root.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(limit)
    tasks: list[asyncio.Task] = []

    async for file_path in awalk(src_root):
        tasks.append(asyncio.create_task(copy_file(file_path, out_root, sem)))

    if not tasks:
        log.warning("no files in the folder %s", src_root)
        return

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            log.error("copy error: %s", r)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="async sort."
    )
    p.add_argument("--source", "-s", required=True, help="source folder")
    p.add_argument("--output", "-o", required=True, help="output folder")
    p.add_argument("--limit", "-l", type=int, default=64, help="limit of concurrent copy operations (default: 64)")
    return p


async def amain(args: argparse.Namespace) -> None:
    src = AsyncPath(args.source)
    out = AsyncPath(args.output)
    await read_folder(src, out, limit=args.limit)


if __name__ == "__main__":
    parser = build_parser()

    # якщо скрипт запущено без аргументів — підставимо значення за замовчуванням
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend([
            "--source", r"C:\Users\1028581\OneDrive - IMPERIAL TOBACCO LTD\Desktop\GoIT\Comp_gen\HW_5\source folder",
            "--output", r"C:\Users\1028581\OneDrive - IMPERIAL TOBACCO LTD\Desktop\GoIT\Comp_gen\HW_5\output folder",
            "--limit", "32"
        ])

    ns = parser.parse_args()
    try:
        asyncio.run(amain(ns))
    except KeyboardInterrupt:
        log.warning("cancelled by user (Ctrl+C)")


