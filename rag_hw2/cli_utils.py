from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import yaml


def load_config_file(path) :
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return json.loads(p.read_text(encoding="utf-8"))


def add_common_logging_args(parser) :
    parser.add_argument("--verbose", action="store_true", help="Print progress details.")
