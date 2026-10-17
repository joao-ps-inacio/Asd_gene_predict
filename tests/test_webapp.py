import json

import pandas as pd
import pytest

from asd_gene_predict import webapp as wa

UNIPROT_TSV = (
    "Entry\tGene Names (primary)\tGene Names (synonym)\tProtein names\tFunction [CC]\t"
    "Involvement in disease\tKeywords\tLength\tHGNC\n"
    "Q9HCK8\tCHD8\t\tChromodomain-helicase-DNA-binding protein 8 (CHD-8) (EC 3.6.4.12)\t"
    "FUNCTION: Slides nucleosomes along DNA (PubMed:28533432). Represses transcription "
    "{ECO:0000269|PubMed:1}. FUNCTION: [Isoform 2]: Binds CTCF.\t"
    "DISEASE: Intellectual developmental disorder with autism and macrocephaly (IDDAM) "
    "[MIM:615032]: An autosomal dominant disorder. {ECO:0000269|PubMed:23160955}.\t"
    "Autism;Phosphoprotein;Chromatin regulator;Reference proteome\t2581\tHGNC:20153;\n"
    "X0FRAG\tCHD8\t\tCHD8 fragment\t\t\t\t120\tHGNC:20153;\n"
    "P00001\tGENEX\t\tProtein X\t\t\tMembrane\t300\tHGNC:1;\n"
)


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_rank_scores_breaks_ties_by_gene_id():
    ranks = wa.rank_scores(pd.Series({"G3": 0.9, "G1": 0.9, "G2": 0.1, "G4": None}))
    assert ranks.to_dict() == {"G1": 1, "G3": 2, "G2": 3}


def test_gene_roles_follow_what_the_model_knew():
    old = pd.DataFrame(
        {
            "ensembl_gene_id": ["P1", "P2", "S1"],
            "sfari_score": pd.array([1, None, 2], dtype="Int64"),
        }
    )
    roles = wa.gene_roles(["P1", "P2", "S1", "N1", "A1", "U1"], old, ["N1"], added=["A1"])
    assert roles.to_dict() == {
        "P1": "train_pos",  # category 1
        "P2": "train_pos",  # syndromic, no score
        "S1": "sfari_2024",  # known, not used for training
        "N1": "train_neg",
        "A1": "new",
        "U1": "",
    }


def test_pool_summary_counts_new_genes_near_the_top():
    ranks = pd.Series(range(1, 101), index=[f"G{i}" for i in range(1, 101)])
    scores = pd.Series(1.0, index=ranks.index)
    pool = wa.candidate_pool([scores, scores.iloc[:-1]], exclude={"G1"})  # G100 lacks a score
    assert len(pool) == 98 and "G1" not in pool
    summary = wa.pool_summary(ranks, pool, positives={"G2", "G3", "G60"})
    assert summary["n_pool"] == 98 and summary["n_new"] == 3
    assert summary["top1"] == pytest.approx(1 / 3, abs=1e-3)  # G2 is the best pool gene
    assert summary["top10"] == pytest.approx(2 / 3, abs=1e-3)


def test_clean_function_drops_references_and_splits_isoforms():
    paragraphs = wa.clean_function(
        "FUNCTION: Slides nucleosomes (PubMed:1, PubMed:22). Represses {ECO:0000269|PubMed:1}. "
        "FUNCTION: [Isoform 2]: Binds CTCF (By similarity). {ECO:0000250|UniProtKB:Q1}."
    )
    assert paragraphs == [
        "Slides nucleosomes. Represses.",
        "[Isoform 2]: Binds CTCF (By similarity).",
    ]
    assert wa.clean_function(None) == []


def test_shorten_cuts_at_a_sentence():
    text, more = wa.shorten(["First sentence. Second sentence is long."], limit=25)
    assert (text, more) == ("First sentence.", True)
    assert wa.shorten(["Short."], limit=25) == ("Short.", False)
    assert wa.shorten(["Short.", "Isoform text."], limit=25) == ("Short.", True)


def test_parse_diseases_with_and_without_mim():
    text = (
        "DISEASE: Rett syndrome (RTT) [MIM:312750]: A disorder. "
        "DISEASE: Autism, X-linked 3 (AUTSX3): A condition. Note=Something. "
        "DISEASE: Note=A chromosomal aberration involving the gene is found in patients."
    )
    assert wa.parse_diseases(text) == [
        ["Rett syndrome (RTT)", "312750"],
        ["Autism, X-linked 3 (AUTSX3)", None],
    ]


def test_select_keywords_drops_generic_ones():
    assert wa.select_keywords("Autism;Phosphoprotein;Synapse;3D-structure") == [
        "Autism",
        "Synapse",
    ]


def test_read_uniprot_keeps_the_entry_with_a_function(tmp_path):
    up = wa.read_uniprot(write(tmp_path, "up.tsv", UNIPROT_TSV))
    chd8 = up.loc["HGNC:20153"]
    assert chd8["accession"] == "Q9HCK8"
    assert chd8["protein_name"] == "Chromodomain-helicase-DNA-binding protein 8"
    assert chd8["function"][0].startswith("Slides nucleosomes along DNA.")
    assert chd8["diseases"] == [
        ["Intellectual developmental disorder with autism and macrocephaly (IDDAM)", "615032"]
    ]
    assert chd8["keywords"] == ["Autism", "Chromatin regulator"]


def test_read_loeuf_prefers_mane_transcript(tmp_path):
    tsv = (
        "gene\tgene_id\ttranscript\tcanonical\tmane_select\tlof.oe_ci.upper\n"
        "CHD8\tENSG00000100888\tT1\ttrue\tfalse\t0.5\n"
        "CHD8\tENSG00000100888\tT2\tfalse\ttrue\t0.1\n"
        "ABC\tENSG00000000001\tT3\ttrue\tfalse\t1.2\n"
    )
    loeuf = wa.read_loeuf(write(tmp_path, "gnomad.tsv", tsv))
    assert loeuf.to_dict() == {"ENSG00000100888": 0.1, "ENSG00000000001": 1.2}


def test_build_and_write_app_data(tmp_path):
    protein = pd.DataFrame(
        {"symbol": ["CHD8", "OLDX"], "score": [0.99, 0.2]},
        index=pd.Index(["ENSG00000100888", "ENSG00000000001"], name="ensembl_gene_id"),
    )
    graph = protein.iloc[:1].assign(score=0.5)
    hgnc = pd.DataFrame(
        {
            "hgnc_id": ["HGNC:20153", "HGNC:1"],
            "symbol": ["CHD8", "GENEX"],
            "name": ["chromodomain helicase DNA binding protein 8", "gene X"],
            "locus_group": ["protein-coding gene", "non-coding RNA"],
            "location": ["14q11.2", "1p36"],
            "gene_group": ["SNF2 related family", None],
            "omim_id": ["610528", None],
            "other_symbols": ["", "OLDX"],
        },
        index=pd.Index(["ENSG00000100888", "ENSG00000000001"], name="ensembl_gene_id"),
    )
    uniprot = wa.read_uniprot(write(tmp_path, "up.tsv", UNIPROT_TSV))
    s24 = pd.Series({"ENSG00000100888": "1"})
    roles = pd.Series({"ENSG00000100888": "train_pos"})

    genes = wa.build_genes(
        protein, graph, roles, pool={"ENSG00000000001"}, s24=s24, hgnc=hgnc, uniprot=uniprot
    )
    assert genes.columns.tolist() == wa.COLUMNS
    chd8, genex = genes.to_dict("records")
    assert (chd8["sym"], chd8["pr"], chd8["gr"], chd8["type"]) == ("CHD8", 1, 1, "pc")
    assert chd8["acc"] == "Q9HCK8" and chd8["s24"] == "1"
    assert (genex["sym"], genex["alt"], genex["pool"], genex["role"]) == ("GENEX", "OLDX", 1, "")
    assert pd.isna(genex["gr"])

    nameless = protein.rename(index={"ENSG00000000001": "ENSG00000000002"}).assign(symbol=None)
    no_symbol = wa.build_genes(nameless, graph.iloc[:0], roles, pool=set(), s24=s24)
    assert no_symbol["sym"].tolist() == ["ENSG00000100888", "ENSG00000000002"]

    annotations = wa.build_annotations(genes, hgnc, uniprot)
    assert annotations["ENSG00000000001"] == {"protein": "Protein X", "keywords": ["Membrane"]}
    annotations.pop("ENSG00000000001")

    out = tmp_path / "app"
    (out / "annotations").mkdir(parents=True)
    (out / "annotations" / "stale.json").write_text("{}")
    wa.write_app_data(genes, annotations, {"n_genes": 2}, out)
    table = json.loads((out / "genes.json").read_text())
    assert table["columns"] == wa.COLUMNS and len(table["rows"]) == 2
    assert table["rows"][1][wa.COLUMNS.index("gr")] is None
    assert sorted(p.name for p in (out / "annotations").iterdir()) == ["88.json"]
    shard = json.loads((out / "annotations" / "88.json").read_text())
    assert shard["ENSG00000100888"]["omim"] == "610528"
