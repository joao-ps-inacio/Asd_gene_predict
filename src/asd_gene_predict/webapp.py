"""Data for the web app: the thesis rankings joined with labels and gene annotations.

Output, in ``app/data/``:

- ``genes.json``: one row per ranked gene (ranks, SFARI status, gene name, LOEUF), stored as
  columns + rows to keep the file small;
- ``annotations/NN.json``: UniProt function, disease involvement and keywords, split into 100
  files by the last two digits of the Ensembl ID, so the app downloads only what it shows;
- ``meta.json``: sources, versions and the temporal-validation summary shown in the app.

Gene roles (what the model knew about each gene in January 2024):

- ``train_pos``: SFARI category 1 or syndromic without a score, the thesis training positives;
- ``train_neg``: Krishnan et al. (2016) negatives;
- ``sfari_2024``: SFARI categories 2 and 3, known but not used for training;
- ``new``: added to SFARI after January 2024;
- ``""``: no label.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from asd_gene_predict.paths import ROOT

APP_DATA = ROOT / "app" / "data"
ID = "ensembl_gene_id"
TOPS = (0.01, 0.05, 0.10)

# UniProt keywords that describe structure, chemistry or annotation status rather than what the
# gene does. They are dropped so the remaining keywords are worth reading.
_GENERIC = """
    Reference proteome; Proteomics identification; Alternative splicing; 3D-structure;
    Phosphoprotein; Repeat; Glycoprotein; Disulfide bond; Signal; Acetylation;
    Direct protein sequencing; Ubl conjugation; Isopeptide bond; Methylation; Lipoprotein;
    Coiled coil; Transmembrane; Transmembrane helix; Metal-binding; Zinc; Magnesium; Calcium;
    Iron; Manganese; Copper; Potassium; Sodium; Disease variant; Polymorphism; Palmitate;
    Myristate; Prenylation; Zymogen; Cleavage on pair of basic residues; Signal-anchor;
    Transit peptide; Leucine-rich repeat; WD repeat; ANK repeat; SH3 domain; SH2 domain;
    EGF-like domain; Immunoglobulin domain; Zinc-finger; Nucleotide-binding; ATP-binding;
    GTP-binding; Hydroxylation; Sulfation; ADP-ribosylation; Citrullination; Amidation;
    Pyrrolidone carboxylic acid; GPI-anchor; Chromosomal rearrangement; Lipid-binding;
    Kelch repeat; TPR repeat; Collagen; Selenocysteine; Phosphopantetheine; Pyridoxal phosphate;
    FAD; Flavoprotein; NAD; NADP; Heme; Iron-sulfur; 4Fe-4S; 2Fe-2S; Cobalt; Nickel; Cadmium
"""
GENERIC_KEYWORDS = frozenset(k.strip() for k in _GENERIC.split(";") if k.strip())

# How each input is named and linked in the app.
THESIS_REPO = "https://github.com/a59490/Tese_ASD_Gene_Pred"
SOURCE_INFO = {
    "legacy_rank_protein": ("Thesis ranking: protein embeddings (ProtT5) + XGBoost", THESIS_REPO),
    "legacy_rank_graph": (
        "Thesis ranking: STRING network embeddings (DeepWalk) + logistic regression",
        THESIS_REPO,
    ),
    "sfari": ("SFARI Gene, release used for training", "https://gene.sfari.org/"),
    "krishnan_2016": (
        "Krishnan et al. (2016), Nature Neuroscience: negative genes",
        "https://doi.org/10.1038/nn.4353",
    ),
    "sfari_2026q2": ("SFARI Gene, release used for validation", "https://gene.sfari.org/"),
    "legacy_peptides": ("Protein sequences used in the thesis", THESIS_REPO),
    "hgnc": ("HGNC gene symbols, names and groups", "https://www.genenames.org/"),
    "uniprot_human": (
        "UniProtKB/Swiss-Prot function, disease involvement and keywords",
        "https://www.uniprot.org/",
    ),
    "gnomad_constraint": (
        "gnomAD constraint metrics (LOEUF)",
        "https://gnomad.broadinstitute.org/",
    ),
}

_ECO = re.compile(r"\s*\{ECO:[^}]*\}")
_PUBMED = re.compile(r"\s*\((?:PubMed:\d+(?:,\s*)?)+\)")
_DISEASE = re.compile(r"\s*(.+?)\s*(?:\[MIM:(\d+)\])?\s*:")


# --------------------------------------------------------------------------- rankings


def read_ranking(path: Path) -> pd.DataFrame:
    """Thesis ranked list → ``symbol | score`` indexed by Ensembl gene ID."""
    df = pd.read_csv(path, usecols=["Gene", "Ensembl_ID", "Probability_Class_1"])
    df = df.drop_duplicates("Ensembl_ID").set_index("Ensembl_ID").rename_axis(ID)
    return df.rename(columns={"Gene": "symbol", "Probability_Class_1": "score"})


def rank_scores(scores: pd.Series) -> pd.Series:
    """Rank 1 = highest score. Ties are broken by gene ID so the order is reproducible."""
    df = scores.dropna().rename("s").rename_axis(ID).reset_index()
    df = df.sort_values(["s", ID], ascending=[False, True], kind="mergesort")
    return pd.Series(np.arange(1, len(df) + 1), index=df[ID].to_numpy(), name="rank")


# --------------------------------------------------------------------------- labels


def sfari_label(sfari: pd.DataFrame) -> pd.Series:
    """SFARI status as shown in the app: ``"1"``, ``"2"``, ``"3"`` or ``"S"`` (syndromic gene
    without a score)."""
    label = sfari["sfari_score"].astype("string")
    return label.fillna("S").set_axis(sfari[ID]).rename("sfari")


def gene_roles(
    genes: Iterable[str],
    sfari_old: pd.DataFrame,
    negatives: Iterable[str],
    added: Iterable[str],
) -> pd.Series:
    """What the thesis model knew about each gene (see the module docstring)."""
    genes = pd.Index(genes)
    score = sfari_old.set_index(ID)["sfari_score"]
    train_pos = set(score.index[score.isna() | score.le(1).fillna(False)])
    known = set(score.index)
    role = pd.Series("", index=genes, name="role")
    role[genes.isin(list(added))] = "new"
    role[genes.isin(list(set(negatives)))] = "train_neg"
    role[genes.isin(list(known - train_pos))] = "sfari_2024"
    role[genes.isin(list(train_pos))] = "train_pos"
    return role


def candidate_pool(rankings: Iterable[pd.Series], exclude: Iterable[str]) -> set[str]:
    """Genes scored by every ranking and unlabelled in January 2024: the pool used by the
    temporal validation, so the app reproduces its numbers."""
    shared = set.intersection(*(set(s.dropna().index) for s in rankings))
    return shared - set(exclude)


def pool_summary(ranks: pd.Series, pool: set[str], positives: set[str]) -> dict:
    """Share of the newly added genes found in the top 1/5/10% of the pool."""
    r = ranks[ranks.index.isin(pool)].sort_values()
    n = len(r)
    hits = r.index.isin(positives)
    out = {"n_pool": n, "n_new": int(hits.sum())}
    for top in TOPS:
        k = int(np.ceil(top * n))
        out[f"top{round(top * 100)}"] = round(float(hits[:k].sum() / max(hits.sum(), 1)), 4)
    return out


# --------------------------------------------------------------------------- annotations


def read_hgnc(path: Path) -> pd.DataFrame:
    """HGNC complete set → one row per Ensembl gene: symbol, name, location, locus group,
    gene group, other symbols, HGNC and OMIM IDs."""
    cols = [
        "hgnc_id",
        "symbol",
        "name",
        "locus_group",
        "location",
        "gene_group",
        "prev_symbol",
        "alias_symbol",
        "ensembl_gene_id",
        "omim_id",
    ]
    df = pd.read_csv(path, sep="\t", usecols=lambda c: c in cols, dtype=str)
    df = df.dropna(subset=[ID]).drop_duplicates(ID).set_index(ID)
    other = df.reindex(columns=["prev_symbol", "alias_symbol"]).fillna("")
    df["other_symbols"] = (other["prev_symbol"] + "|" + other["alias_symbol"]).str.strip("|")
    return df.drop(columns=["prev_symbol", "alias_symbol"], errors="ignore")


def clean_function(text: object) -> list[str]:
    """UniProt ``Function [CC]`` → paragraphs without evidence codes or PubMed references."""
    if not isinstance(text, str):
        return []
    paragraphs = []
    for part in text.split("FUNCTION:"):
        part = _PUBMED.sub("", _ECO.sub("", part))
        part = re.sub(r"\s+([.,;])", r"\1", part)
        part = re.sub(r"\.(\s*\.)+", ".", part)  # "text. {ECO:...}." leaves ".."
        part = re.sub(r"\s{2,}", " ", part).strip()
        if part:
            paragraphs.append(part)
    return paragraphs


def shorten(paragraphs: list[str], limit: int = 800) -> tuple[str, bool]:
    """First paragraph, cut at the last full sentence before ``limit`` characters. Returns the
    text and whether anything was left out (the app then links to the full UniProt entry)."""
    if not paragraphs:
        return "", False
    text = paragraphs[0]
    if len(text) <= limit:
        return text, len(paragraphs) > 1
    cut = text[:limit].rsplit(". ", 1)[0]
    return (cut if cut.endswith(".") else cut + "."), True


def parse_diseases(text: object) -> list[list[str | None]]:
    """UniProt ``Involvement in disease`` → ``[[name, mim_id], ...]``."""
    if not isinstance(text, str):
        return []
    out = []
    for chunk in text.split("DISEASE:")[1:]:
        if chunk.lstrip().startswith("Note="):  # a remark, not a named disease
            continue
        m = _DISEASE.match(chunk)
        if m:
            out.append([m.group(1).strip(), m.group(2)])
    return out


def select_keywords(text: object) -> list[str]:
    if not isinstance(text, str):
        return []
    return [k for k in text.split(";") if k and k not in GENERIC_KEYWORDS]


def read_uniprot(path: Path) -> pd.DataFrame:
    """UniProt TSV (reviewed human entries) → one entry per HGNC ID: accession, protein name,
    function paragraphs, diseases and keywords. When a gene has several entries, the one with
    a function and the longest sequence wins."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["hgnc_id"] = df["HGNC"].str.extract(r"(HGNC:\d+)", expand=False)
    df["length"] = pd.to_numeric(df["Length"], errors="coerce")
    df["has_function"] = df["Function [CC]"].notna()
    df = df.dropna(subset=["hgnc_id"]).sort_values(
        ["has_function", "length"], ascending=False, kind="mergesort"
    )
    df = df.drop_duplicates("hgnc_id").set_index("hgnc_id")
    return pd.DataFrame(
        {
            "accession": df["Entry"],
            "protein_name": df["Protein names"].str.replace(r"\s*\(.*$", "", regex=True),
            "function": df["Function [CC]"].map(clean_function),
            "diseases": df["Involvement in disease"].map(parse_diseases),
            "keywords": df["Keywords"].map(select_keywords),
        }
    )


def read_loeuf(path: Path) -> pd.Series:
    """gnomAD constraint → LOEUF per gene, from the MANE Select transcript (or the canonical
    one when a gene has no MANE transcript)."""
    df = pd.read_csv(path, sep="\t", low_memory=False, dtype={"mane_select": str, "canonical": str})
    df = df[df["gene_id"].astype(str).str.startswith("ENSG")]
    df["mane"] = df["mane_select"].str.lower().eq("true")
    df["canon"] = df["canonical"].str.lower().eq("true")
    df = df[df["mane"] | df["canon"]].dropna(subset=["lof.oe_ci.upper"])
    df = df.sort_values(["mane", "canon"], ascending=False, kind="mergesort")
    return df.drop_duplicates("gene_id").set_index("gene_id")["lof.oe_ci.upper"].rename_axis(ID)


# --------------------------------------------------------------------------- assembly

LOCUS_CODES = {
    "protein-coding gene": "pc",
    "non-coding RNA": "nc",
    "pseudogene": "ps",
    "other": "o",
}

COLUMNS = [
    "id",  # Ensembl gene ID
    "sym",  # HGNC symbol (current), else the symbol used in the thesis
    "alt",  # other symbols, "|"-separated, for search
    "name",  # HGNC gene name
    "loc",  # cytogenetic location
    "type",  # HGNC locus group, coded with LOCUS_CODES
    "pr",  # rank in the protein (ProtT5) ranking, 1 = most likely ASD gene
    "gr",  # rank in the graph (DeepWalk) ranking
    "s24",  # SFARI status in the January 2024 release (training data)
    "s26",  # SFARI status in the July 2026 release
    "role",  # see module docstring
    "pool",  # 1 if the gene is in the temporal-validation candidate pool
    "loeuf",  # gnomAD LOEUF (lower = less tolerant to loss of function)
    "acc",  # UniProt accession
]


def build_genes(
    protein: pd.DataFrame,
    graph: pd.DataFrame,
    roles: pd.Series,
    pool: set[str],
    s24: pd.Series,
    s26: pd.Series | None = None,
    hgnc: pd.DataFrame | None = None,
    uniprot: pd.DataFrame | None = None,
    loeuf: pd.Series | None = None,
) -> pd.DataFrame:
    """One row per gene in either ranking, with the columns in ``COLUMNS``."""
    ids = protein.index.union(graph.index)
    out = pd.DataFrame(index=ids)
    thesis_symbol = protein["symbol"].combine_first(graph["symbol"]).reindex(ids)
    hg = (hgnc if hgnc is not None else pd.DataFrame(columns=["symbol"])).reindex(ids)

    # A few genes have no symbol anywhere; they are shown by their Ensembl ID.
    out["sym"] = hg["symbol"].fillna(thesis_symbol).fillna(pd.Series(ids, index=ids))
    other = hg.get("other_symbols", pd.Series("", index=ids)).fillna("")
    out["alt"] = [
        "|".join(dict.fromkeys(s for s in f"{old}|{alt}".split("|") if s and s != sym))
        for sym, old, alt in zip(out["sym"], thesis_symbol.fillna(""), other, strict=True)
    ]
    out["name"] = hg.get("name")
    out["loc"] = hg.get("location")
    out["type"] = hg.get("locus_group", pd.Series(index=ids, dtype=str)).map(LOCUS_CODES)
    out["pr"] = rank_scores(protein["score"]).reindex(ids).astype("Int64")
    out["gr"] = rank_scores(graph["score"]).reindex(ids).astype("Int64")
    out["s24"] = s24.reindex(ids)
    out["s26"] = s26.reindex(ids) if s26 is not None else None
    out["role"] = roles.reindex(ids).fillna("")
    out["pool"] = ids.isin(list(pool)).astype(int)
    out["loeuf"] = loeuf.reindex(ids).round(3) if loeuf is not None else None
    if uniprot is not None and "hgnc_id" in hg:
        out["acc"] = uniprot["accession"].reindex(hg["hgnc_id"]).to_numpy()
    else:
        out["acc"] = None
    out = out.rename_axis("id").reset_index()
    order = out["pr"].fillna(out["gr"] + len(out)).sort_values(kind="mergesort").index
    return out.loc[order, COLUMNS].reset_index(drop=True)


def build_annotations(
    genes: pd.DataFrame, hgnc: pd.DataFrame, uniprot: pd.DataFrame
) -> dict[str, dict]:
    """Per-gene detail for the app: protein name, function, diseases, keywords, gene group,
    OMIM ID. Genes with nothing to show are left out."""
    hg = hgnc.reindex(genes["id"])
    up = uniprot.reindex(hg["hgnc_id"])
    out = {}
    for gid, (_, h), (_, u) in zip(genes["id"], hg.iterrows(), up.iterrows(), strict=True):
        function, more = shorten(u["function"] if isinstance(u["function"], list) else [])
        entry = {
            "protein": u["protein_name"] if isinstance(u["protein_name"], str) else None,
            "function": function,
            "more": more,
            "diseases": u["diseases"] if isinstance(u["diseases"], list) else [],
            "keywords": u["keywords"] if isinstance(u["keywords"], list) else [],
            "group": h["gene_group"] if isinstance(h["gene_group"], str) else None,
            "omim": h["omim_id"] if isinstance(h["omim_id"], str) else None,
        }
        entry = {k: v for k, v in entry.items() if v}
        if entry:
            out[gid] = entry
    return out


def shard(gene_id: str) -> str:
    """Annotation file for a gene: the last two digits of its Ensembl ID."""
    return gene_id[-2:]


def _json_value(v):
    if v is None or v is pd.NA:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def write_app_data(
    genes: pd.DataFrame,
    annotations: dict[str, dict],
    meta: dict,
    out_dir: Path = APP_DATA,
) -> list[Path]:
    """Write ``genes.json``, ``meta.json`` and the annotation shards; old shards are removed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [[_json_value(v) for v in row] for row in genes[COLUMNS].itertuples(index=False)]
    written = [out_dir / "genes.json", out_dir / "meta.json"]
    written[0].write_text(
        json.dumps({"columns": COLUMNS, "rows": rows}, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    written[1].write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    ann_dir = out_dir / "annotations"
    shutil.rmtree(ann_dir, ignore_errors=True)
    ann_dir.mkdir()
    shards: dict[str, dict] = {}
    for gid, entry in annotations.items():
        shards.setdefault(shard(gid), {})[gid] = entry
    for key, entries in sorted(shards.items()):
        path = ann_dir / f"{key}.json"
        path.write_text(
            json.dumps(entries, separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
        )
        written.append(path)
    return written
