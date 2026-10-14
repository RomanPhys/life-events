import tempfile
from functools import lru_cache
from pathlib import Path

from src.prepare import prepare
from src.synthetic import generate


@lru_cache(maxsize=1)
def tiny_prepared() -> str:
    """Маленький синтетический датасет, подготовленный тем же prepare, что и MBD."""
    root = Path(tempfile.mkdtemp(prefix="life_events_"))
    trx, targets, split = generate(150, seed=1)
    for name, df in [("trx", trx), ("targets", targets), ("client_split", split)]:
        (root / "raw" / name).mkdir(parents=True)
        df.to_csv(root / "raw" / name / "part-00000.csv", index=False)
    prepare(root / "raw", root / "prep", max_clients=None, last_months=None, max_vocab=50, n_bins=16,
            min_events=1, seed=0)
    return str(root / "prep")
