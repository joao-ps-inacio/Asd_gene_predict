"""Caminhos centrais do projeto. Todo o código usa estes caminhos em vez de paths relativos."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
