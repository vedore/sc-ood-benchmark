# sc-ood-benchmark

# Do frozen embeddings from single-cell foundation models improve cross-domain cell-type annotation relative to conventional PCA and scVI embeddings, under donor/population and laboratory/study shifts?

A benchmark for evaluating the out-of-distribution generalization of single-cell foundation models across populations, donors, laboratories, and studies.


## Core hypothesis:

- Frozen foundation-model embeddings improve macro-F1, especially for rare or subtle cell types, when query cells come from unseen donors or laboratories.
- The advantage should grow as shift severity grows.
- Is any improvement due to better biological transfer, or merely to better mixing / removal of technical variation?

## Domain Shift

| Shift                    | Reference vs query difference                                          | What it tests            |
|--------------------------|------------------------------------------------------------------------|--------------------------|
| Population / donor shift | Different donors; ideally varying ancestry, age, sex, disease state    | Biological heterogeneity |
| Laboratory / study shift | Different labs, protocols, chemistry, sequencing centre, dataset/study | Technical generalization |
| Combined shift           | Different donors and laboratories                                      | Realistic worst case     |


## Representations

| Family              | Representation                      |
|---------------------|-------------------------------------|
| Conventional        | Log-normalized HVGs → PCA           |
| Conventional neural | scVI latent representation          |
| Foundation model    | Frozen model embedding from model A |
| Foundation model    | Frozen model embedding from model B |
| Optional            | Frozen model embedding from model C |

## Relevant candidates worth reviewing are:

- scGPT — published in Nature Methods, 2024, DOI 10.1038/s41592-024-02201-0
- Geneformer — Nature, 2023, DOI 10.1038/s41586-023-06139-9
- scFoundation — Nature Methods, 2024, DOI 10.1038/s41592-024-02305-7
- potentially UCE / Universal Cell Embeddings, depending on checkpoint access and reproducibility

## “Frozen” must mean frozen

Be precise in the methods:

- Foundation-model weights remain fixed.
- No fine-tuning on reference or query.
- A lightweight supervised annotation head is fitted only on labelled reference cells.
- Query labels remain inaccessible until evaluation.

Use two annotation heads:

1. Logistic regression / linear probe  
   Tests whether cell identity is linearly separable in each representation.

2. kNN classifier  
   Tests whether local neighbourhoods preserve cell-type structure.

If foundation embeddings only win with a huge MLP classifier, then the conclusion is not “the frozen representation transferred better”; it is “we trained a stronger downstream model.” Different claim, different thesis.

Standardize each embedding dimension-wise using statistics learned from the reference only before fitting the classifier.

## scVI needs two clearly separated regimes

scVI is an excellent baseline — its documentation confirms it produces a latent representation and supports query transfer — but it can be made accidentally unfair very easily.

Run it in two labelled regimes and do not mix their conclusions:

A. Inductive / strict transfer — primary comparison
- Fit PCA/HVG selection/scVI using reference data only.
- Embed the unseen query without using its labels.
- Train classifier on reference.
- Predict query.

This is closest to “can a frozen foundation representation generalize better than representations learned from available reference data?”

B. Transductive unsupervised adaptation — secondary comparison
- Allow each conventional method to see unlabelled query expression during representation learning/integration.
- No query labels are allowed.
- Report this separately.

This asks a different but useful question: “does frozen foundation-model inference compete with methods allowed to adapt to the target distribution without labels?”

## Split design: the heart of the thesis

Build a split manifest first: one row per cell, with at least:

text
cell_id
dataset_or_study
laboratory_or_site
protocol_or_chemistry
donor_id
tissue
disease_state
cell_type_original
cell_type_harmonized


Then define these experiments:

Experiment 1 — Within-study donor generalization
- Train on several donors in one study/lab.
- Test on held-out donors from the same study/lab.
- Repeated donor-level splits.

Measures population shift with relatively controlled technical variation.

Experiment 2 — Cross-study / cross-lab generalization
- Train on one or more studies/labs.
- Test on an entirely held-out study/lab.
- Ensure no donor overlap.

Measures laboratory/study transfer.

Experiment 3 — Combined external generalization
- Train on a reference collection.
- Test on a completely independent dataset: new lab, new donors, preferably a different protocol.
- This should be your headline result.

Experiment 4 — Shift severity curve
Create reference sets with progressively less diversity:

- one donor → held-out donors
- few donors → held-out donors
- many donors → held-out donors

That gives a proper curve: does pretrained knowledge help most when labelled reference data are scarce? This is where foundation models may have their best case.

8. Labels are more dangerous than the models

Label harmonization is not cleanup work; it is a central methodological component.

Use:

- original labels preserved in cell_type_original;
- a documented mapping to cell_type_harmonized;
- at least two annotation levels where possible:
  - coarse: broad immune populations;
  - fine: CD4 naïve, CD4 memory, CD8, Treg, classical/non-classical monocytes, etc.

Rules:

- Evaluate only labels that exist in both reference and query for the closed-set experiment.
- Report the excluded labels.
- Add an unseen-cell-type experiment separately, rather than silently forcing a novel type into the closest known class.

If a reference has no dendritic cells and the query does, a normal classifier will confidently call them something stupid. That is not necessarily a representation failure; it is a closed-set assumption failure.

9. Metrics that will survive supervisor interrogation

Make macro-F1 the primary metric. It gives rare types a real voice rather than letting abundant T cells run the election.

Report:

- macro-F1 — primary;
- balanced accuracy;
- per-cell-type F1;
- confusion matrices;
- top-2 / top-3 accuracy for fine labels;
- calibration: expected calibration error or Brier score, if your classifier gives probabilities;
- rejection / unknown detection metrics for the open-set experiment;
- inference runtime, GPU/CPU memory, and embedding dimensionality.

For statistical comparison:

- Aggregate metrics per held-out donor or held-out dataset, not per cell.
- Use paired confidence intervals or a paired non-parametric test across split replicates.
- Bootstrap donors/datasets, not only cells.

A million correlated cells are not one million independent experiments. Biology will slap nôs if nôs pretend otherwise.

10. Start this week — real sequence

Days 1–2: protocol, before code
Write a one-page preregistration-style protocol:

- primary question and hypothesis;
- selected models and exact checkpoints;
- datasets;
- exclusion criteria;
- label mapping policy;
- splits;
- primary metric;
- secondary analyses;
- compute budget.

This prevents benchmark shopping after seeing results.

Days 2–4: one dataset, one split, three baselines
Use a small, curated PBMC reference/query pair and implement:

text
raw counts
  ├── HVG → log normalization → PCA
  ├── scVI latent
  └── frozen foundation-model embedding
             ↓
    same logistic-regression classifier
             ↓
    held-out donor / held-out-study macro-F1


Do this with only:

- PCA;
- scVI;
- one foundation model;
- one linear classifier;
- one donor-held-out split.

Until that runs end-to-end, do not download six more atlases.

Week 2: make it scientifically valid
- Add repeated group-aware splits.
- Add kNN.
- Add the laboratory/study-held-out split.
- Save every split as versioned CSV/Parquet, including random seed and metadata.
- Generate a results table automatically.

Weeks 3–4: robustness and interpretation
- Add the second foundation model.
- Add label-resolution analysis.
- Add class-frequency analysis.
- Check whether errors are biologically plausible neighbours or absurd ones.
- Run a small ablation for number of labelled donors/cells.

11. Suggested thesis structure

1. Introduction  
   Cell-type annotation, batch/domain shifts, emergence of single-cell foundation models.

2. Related work  
   PCA/HVG, scVI, supervised transfer annotation, foundation models, benchmark limitations.

3. Methods  
   Datasets, metadata curation, label ontology, embedding extraction, strict split protocol, classifiers, metrics and statistics.

4. Results  
   Population-shift results, lab-shift results, combined shift, data-scarcity curves, rare cell types, calibration and compute.

5. Discussion  
   Where frozen representations help, where they do not, fairness caveats, domain-shift confounding, reproducibility and practical cost.

6. Conclusion  
   Answer the actual question with conditions, not a fake universal winner.

12. Expected outcomes — do not marry a hypothesis

A defensible conclusion may be one of these:

- Frozen foundation embeddings consistently win under lab shifts and low-label regimes.
- They improve fine-grained or rare-cell annotation, but not broad cell classes.
- scVI matches or beats them when allowed target-aware unsupervised adaptation.
- PCA is surprisingly competitive for broad labels, at a fraction of cost.
- Results depend strongly on model gene coverage, pretraining corpus overlap, and label granularity.

Any of those is a good thesis result if nôs design prevents leakage and reports it honestly.

The actual first deliverable

Before code, produce these three small artifacts:

1. study_protocol.md — the locked experimental decisions;
2. data_catalog.csv — candidate datasets and their donor/lab/protocol/label metadata;
3. label_mapping.csv — original labels → harmonized coarse/fine labels, with exclusions explained.

Then nôs build the benchmark harness around those. The representation extractors should be plugins; everything else — splits, annotation heads, metrics, plots — must be shared. That architecture keeps the science comparable and stops model-specific spaghetti from colonizing the lab. 🫡

## Independent datasets

Developing: https://cellxgene.cziscience.com/collections/b0cf0afa-ec40-4d65-b570-ed4ceacc6813

Final dataset: https://cellxgene.cziscience.com/collections/dde06e0f-ab3b-46be-96a2-a8082383c4a1


### Rubberish

  Cell 0
  ├── adata.X[0]        gene-expression measurements
  ├── adata.raw.X[0]    raw gene counts
  └── manifest.iloc[0]  metadata and labels


If a classifier learns cell types from known donors or laboratories, which representation works best when it sees cells from a new donor or laboratory?

## Dataset metadata map

Create aggregate hierarchy tables and plots from the generated manifest:

```bash
python3 notebooks/02_dataset_map.py
```

This requires `pandas` and writes a visual `dataset_map.html` plus supporting
CSV tables to `reports/data_map/`. It reads metadata only, not the `.h5ad`
expression matrix.
