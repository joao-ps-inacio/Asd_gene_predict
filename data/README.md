# Dados

Os dados **não vão para o Git** (ver `.gitignore`). Esta pasta só guarda a estrutura.

| Pasta | Conteúdo |
|---|---|
| `raw/` | Ficheiros originais, nunca editados: export do SFARI, lista de Krishnan et al. (2016), FASTA do Ensembl, links PPI do STRING |
| `external/` | Recursos de terceiros para validação (VariCarta, MSSNG, ficheiros `.gmt`) |
| `interim/` | Resultados intermédios: sequências limpas |
| `processed/` | Tabelas em Parquet: `gene_map.parquet` (IDs), embeddings e labels |

## Fontes

Todas as fontes estão declaradas em [`configs/sources.yaml`](../configs/sources.yaml) (URL, versão, etapa).

```bash
asd fetch                    # todas as fontes automáticas; verifica as manuais
asd fetch --stage gene_map   # só HGNC, MANE e STRING aliases
asd fetch hgnc --force       # voltar a descarregar uma fonte
```

- O primeiro download grava o SHA-256 em `configs/sources.lock.yaml`, **que vai para o Git**. Os downloads seguintes são verificados contra esse ficheiro; se a fonte mudar, o `asd fetch` falha em vez de usar dados diferentes em silêncio. Para aceitar a mudança: `asd fetch --update-lock`.
- **SFARI** e **Krishnan** vêm do repo original da tese, fixados num commit, para que os labels sejam exatamente os da tese.

Ligações:

- **SFARI Gene**: https://gene.sfari.org/database/human-gene/ (a tese usou a release de 16/01/2024)
- **Krishnan et al. 2016** (negativos): https://www.nature.com/articles/nn.4353
- **Ensembl FTP** (cDNA e proteína): https://www.ensembl.org/info/data/ftp/index.html
- **STRING** (PPI humano): https://string-db.org/cgi/download
- **HGNC** (símbolos e IDs): https://www.genenames.org/download/
- **MANE** (transcrito canónico): https://www.ncbi.nlm.nih.gov/refseq/MANE/
