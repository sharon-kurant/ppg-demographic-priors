from __future__ import annotations

from pathlib import Path

import pandas as pd

from ppg_bp_incremental.evaluation.paper_outputs import (
    DATASET_ORDER,
    MODEL_ORDER,
    TARGET_ORDER,
    save_paper_outputs,
)


def test_paper_outputs_are_built_from_complete_aggregate_tables(tmp_path: Path):
    metrics = []
    contrasts = []
    constants = []
    complementarity = []
    spread = []
    for dataset in DATASET_ORDER:
        demographic_fields = "age/sex" if dataset == "PulseDB-MIMIC" else "age/sex/BMI"
        for target in TARGET_ORDER:
            metrics.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "model": "demographics",
                    "condition": "demographics",
                    "mae": 10.0,
                    "demographic_fields": demographic_fields,
                    "pretraining_overlap": "not_applicable",
                }
            )
            constants.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "constant_method": "outer_fit_participant_weighted_median",
                    "constant_mae": 11.0,
                }
            )
            spread.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "n_subjects": 10,
                    "truth_sd_mmhg": 12.0,
                    "prediction_sd_mmhg": 6.0,
                    "prediction_to_truth_sd_ratio": 0.5,
                    "demographic_fields": demographic_fields,
                }
            )
            for index, model in enumerate(MODEL_ORDER):
                metrics.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "model": model,
                        "condition": "frozen",
                        "mae": 12.0,
                        "demographic_fields": "none",
                        "pretraining_overlap": (
                            "source_overlap"
                            if dataset.startswith("PulseDB") and model == "papagei_p"
                            else "none"
                        ),
                    }
                )
                contrasts.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "model": model,
                        "contrast": "add_demographics_to_waveform",
                        "delta_mae_candidate_minus_reference": -0.2 if index < 4 else 0.1,
                        "ci_low": -0.4,
                        "ci_high": -0.01 if index < 2 else 0.3,
                    }
                )
                complementarity.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "model": model,
                        "waveform_lift_beyond_demographics_mmhg": 0.2 if index < 3 else -0.1,
                    }
                )

    paths = {}
    for name, frame in {
        "metrics": pd.DataFrame(metrics),
        "contrasts": pd.DataFrame(contrasts),
        "constants": pd.DataFrame(constants),
        "complementarity": pd.DataFrame(complementarity),
        "spread": pd.DataFrame(spread),
    }.items():
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path

    outputs = save_paper_outputs(
        metrics_csv=paths["metrics"],
        paired_contrasts_csv=paths["contrasts"],
        constants_csv=paths["constants"],
        complementarity_csv=paths["complementarity"],
        prediction_spread_csv=paths["spread"],
        output_root=tmp_path / "paper",
    )

    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
    table = pd.read_csv(outputs["table_csv"])
    assert len(table) == 8
    assert set(table["demographics_improves_waveform_count"]) == {4}
    assert set(table["waveform_improves_demographics_count"]) == {3}
