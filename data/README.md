# Dados

Os dados **não vão para o Git** (ver `.gitignore`). Esta pasta só guarda a estrutura.

| Pasta | Conteúdo |
|---|---|
| `raw/` | Ficheiros originais, nunca editados: export do SFARI, lista de Krishnan et al. (2016), FASTA do Ensembl, links PPI do STRING |
| `external/` | Recursos de terceiros para validação (VariCarta, MSSNG, ficheiros `.gmt`) |
| `interim/` | Resultados intermédios: sequências limpas, tabela de conversão de IDs |
| `processed/` | Tabelas finais prontas para treino: embeddings + label, em Parquet |

## Fontes

- **SFARI Gene**: https://gene.sfari.org/database/human-gene/ (a tese usou a release de 16/01/2024)
- **Krishnan et al. 2016** (negativos): https://www.nature.com/articles/nn.4353
- **Ensembl FTP** (cDNA e proteína): https://www.ensembl.org/info/data/ftp/index.html
- **STRING** (PPI humano): https://string-db.org/cgi/download
