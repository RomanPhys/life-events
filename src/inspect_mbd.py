"""Скачивает MBD-mini с Hugging Face, распаковывает и печатает таблицы, колонки и даты.

    python -m src.inspect_mbd --download --root data/mbd
"""
import argparse
import tarfile
from pathlib import Path

import pandas as pd

FILES = ["client_split.tar.gz", "targets.tar.gz", "detail.tar.gz"]  # ptls.tar.gz (1.4 ГБ) не нужен


def download(root: Path):
    from huggingface_hub import hf_hub_download
    root.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        path = hf_hub_download(repo_id="ai-lab/MBD-mini", filename=f, repo_type="dataset", local_dir=root)
        print("скачано:", path)
        with tarfile.open(path) as t:
            t.extractall(root, filter="data") if hasattr(tarfile, "data_filter") else t.extractall(root)
        Path(path).unlink()  # экономим место


def inspect(root: Path):
    dirs = {}
    for p in root.rglob("*"):
        if p.is_file():
            dirs.setdefault(p.parent, []).append(p)
    for d, files in sorted(dirs.items()):
        size = sum(f.stat().st_size for f in files) / 2 ** 20
        print(f"\n{d.relative_to(root)}: {len(files)} файлов, {size:,.0f} МБ, напр. {files[0].name}")
        data = [f for f in files if f.suffix in (".parquet", ".csv")]
        if not data:
            continue
        f = data[0]
        df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f, nrows=1000)
        print("  колонки:", dict(df.dtypes.astype(str)))
        print(df.head(3).to_string(max_colwidth=40))
        for c in ("event_time", "mon"):
            if c in df.columns:
                print(f"  {c}: {df[c].min()} .. {df[c].max()} (в первом файле)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/mbd")
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    if a.download:
        download(root)
    inspect(root)


if __name__ == "__main__":
    main()
