# Data sources

| Application | Source | Dataset contribution |
|---|---|---|
| Genetic circuits | [Nielsen et al., 2016](https://doi.org/10.1126/science.aac7341); [Cello-UCF](https://github.com/CIDARLAB/Cello-UCF/tree/222827d30a28730404f0601d64f63436354fee9d) | Truth-table contexts, source circuit identifiers, part sequences, device templates and gate characterization |
| Bacterial RBS | [EMOPEC](https://doi.org/10.1038/nmeth.3727); [original implementation](https://github.com/micked/EMOPEC/tree/f4c3665dfea6bb8655dac180b7913d280d4721a9); [U00096.3](https://www.ncbi.nlm.nih.gov/nuccore/U00096.3) | SD activity lookup and spacing correction; fixed native gene contexts |
| Human 5′UTR | [Sample et al., 2019](https://doi.org/10.1038/s41587-019-0164-5); [GSE114002](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114002); [model repository](https://github.com/pjsample/human_5utr_modeling/tree/d53df410c7fb3fcd4bc4541bd7e8c6dc52b66fbe) | Reporter context, historical UTR anchors, original tables and model checkpoints |
| Metabolic pathways | [iJO1366](https://doi.org/10.1038/msb.2011.65); [Rhea](https://www.rhea-db.org/); [UniProt](https://www.uniprot.org/) | Host reconstruction and curated reaction/enzyme annotations |

The UTR source records are GSM3130435 (random unmodified library), GSM3130443
(designed library) and GSM4084997 (variable-length library). The two calibration
files preserve the original data hashes, mapping coefficients and membership
selection rules used by the evaluator.

The pathway catalogue gives primary-paper, reaction and enzyme links for every
option. The five task-level biochemical sources are recorded in their task
provenance and in the original case manifest. These annotations preserve enzyme
identity and cofactor assumptions at the reaction level.

[external-resources.json](external-resources.json) contains exact download URLs,
commits and hashes. [sources.json](sources.json) connects these resources to case
identifiers. Upstream raw data, software and model assets are obtained directly
from their original providers. BioSyn's assembled task records retain the source
identifiers and attribution.

For four EMOPEC files, the experimental copy omits the leading UTF-8 encoding
comment. The download recipe checks the upstream bytes, removes that single
comment and checks the reconstructed bytes against the experimental hash. The
functions and lookup values are identical.

The Cello catalogue retains component characterization and device records and
omits the upstream `motif_library` collection of precomputed circuits. This
filter reproduces the original agent-visible catalogue exactly.
