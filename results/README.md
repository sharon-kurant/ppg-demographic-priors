# Aggregate benchmark results

These CSV files contain the non-participant aggregate values reported in the
conference paper. `benchmark_metrics.csv` contains all baselines plus the
frozen and fine-tuned foundation-model conditions. `seed_metrics.csv` retains
the three fine-tuning seeds separately. The paired-contrast tables quantify
the effect of adding demographics in the frozen and fine-tuned conditions.

The released Pulse-PPG fine-tuning values use fixed training-pool embedding
standardization and zero initialization of the scalar output layer. The
comparison and audit files document this isolated replacement. The two
fine-tuning acceptance receipts verify the original full matrix and the 240
replacement Pulse-PPG runs; `pulseppg_amendment_audit.json` verifies that only
the intended aggregate rows changed.

No row-level predictions, subject or measurement identifiers, demographics,
waveforms, embeddings, or fitted estimators are included. The aggregate files
are evidence for the reported run. Use `scripts/reproduce_benchmark.sh` for the
frozen/Ridge stage and `scripts/reproduce_finetuning.sh` for the end-to-end
fine-tuning stage, starting from the original data and pinned checkpoints.
