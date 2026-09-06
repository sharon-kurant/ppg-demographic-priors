# Aggregate benchmark results

These CSV files contain the non-participant aggregate values reported in the
conference paper. They are provided so that the principal table and figure can
be rebuilt with `python scripts/build_paper_outputs.py`.

No row-level predictions, subject or measurement identifiers, demographics,
waveforms, embeddings, or fitted estimators are included. The aggregate files
are evidence for the reported run; use `scripts/reproduce_benchmark.sh` to run
the benchmark again from the original data and pinned upstream checkpoints.
