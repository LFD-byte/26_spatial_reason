"""日志与实验记录:控制台打印 + JSONL 指标文件。"""

from __future__ import annotations

import json
import time
from pathlib import Path


class JsonlLogger:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def log(self, **record) -> None:
        record = {"time": round(time.time() - self.t0, 1), **record}
        with open(self.path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in record.items()), flush=True)
