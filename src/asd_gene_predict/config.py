"""Leitura da configuração em YAML."""

from pathlib import Path
from typing import Any

import yaml

from asd_gene_predict.paths import CONFIGS


def load_config(path: Path | str = CONFIGS / "default.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
