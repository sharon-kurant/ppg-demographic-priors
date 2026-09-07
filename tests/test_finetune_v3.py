from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from ppg_bp_incremental.models.bp_head_v3 import (  # noqa: E402
    FoundationBPRegressorV3,
    TargetZScoreDecoder,
)
from ppg_bp_incremental.training.finetune_v3 import (  # noqa: E402
    DemographicTransformV3,
    FineTuneConfig,
    PreparedWaveforms,
    TargetZScoreTransform,
    _cohort_mae_mmhg,
    _participant_balanced_weight_vector,
    run_finetuning_v3,
    selection_artifact_path,
)


class _TensorModel(torch.nn.Module):
    def __init__(self, length: int = 8):
        super().__init__()
        self.backbone = torch.nn.Linear(length, 512)

    def forward(self, waveform):
        return self.backbone(waveform[:, 0])


class _PapageiSModel(torch.nn.Module):
    def __init__(self, length: int = 8):
        super().__init__()
        self.backbone = torch.nn.Linear(length, 512)
        self.dense = torch.nn.Linear(512, 512)
        self.expert_layers_1 = torch.nn.ModuleList([torch.nn.Linear(512, 1)])
        self.expert_layers_2 = torch.nn.ModuleList([torch.nn.Linear(512, 1)])
        self.gating_network_1 = torch.nn.Linear(512, 1)
        self.gating_network_2 = torch.nn.Linear(512, 1)

    def forward(self, waveform):
        pooled = self.backbone(waveform[:, 0])
        projected = self.dense(pooled)
        return (
            projected,
            self.expert_layers_1[0](pooled),
            self.expert_layers_2[0](pooled),
            pooled,
        )


class _Encoder:
    def __init__(self, model_key: str, length: int = 8):
        self.model = (
            _PapageiSModel(length) if model_key == "papagei_s" else _TensorModel(length)
        )
        self._length = length

    def fingerprint(self):
        return SimpleNamespace(embedding_dimension=512)


def test_explicit_target_zscore_decoder_round_trip():
    decoder = TargetZScoreDecoder(120.0, 15.0, "outer_training_selection")
    target = torch.tensor([[90.0], [120.0], [150.0]])
    assert torch.allclose(decoder(decoder.encode(target)), target)
    metadata = decoder.metadata()
    assert metadata["fit_pool_role"] == "outer_training_selection"
    assert metadata["estimator_output_scale"] == "outer_training_pool_zscore"
    assert metadata["final_output_scale"] == "raw_mmhg"


def test_papagei_s_native_outputs_preserved_and_auxiliary_heads_frozen():
    wrapper = FoundationBPRegressorV3(
        _Encoder("papagei_s"),
        "papagei_s",
        target_mean_mmhg=120.0,
        target_standard_deviation_mmhg=10.0,
        target_fit_pool_role="outer_training_selection",
    )
    wrapper.set_encoder_trainable(True)
    output = wrapper(torch.randn(3, 1, 8))
    assert tuple(output.native.components) == (
        "downstream_dense_embedding",
        "ipa",
        "sqi",
        "pooled_embedding",
    )
    assert output.native.embedding.shape == (3, 512)
    assert output.prediction_zscore.shape == (3, 1)
    assert output.prediction_mmhg.shape == (3, 1)
    parameters = dict(wrapper.encoder_model.named_parameters())
    assert parameters["backbone.weight"].requires_grad
    assert parameters["dense.weight"].requires_grad
    assert not parameters["expert_layers_1.0.weight"].requires_grad
    assert not parameters["gating_network_2.weight"].requires_grad


def test_zero_initialized_bp_head_decodes_to_training_target_mean():
    wrapper = FoundationBPRegressorV3(
        _Encoder("anyppg"),
        "anyppg",
        target_mean_mmhg=123.0,
        target_standard_deviation_mmhg=17.0,
        target_fit_pool_role="full_development_refit",
        zero_initialize_output_layer=True,
    )
    output = wrapper(torch.randn(4, 1, 8))
    assert torch.equal(output.prediction_zscore, torch.zeros(4, 1))
    assert torch.equal(output.prediction_mmhg, torch.full((4, 1), 123.0))


def test_demographics_use_train_only_imputation_and_unknown_sex():
    train = pd.DataFrame(
        {
            "subject_id": ["a", "b", "c"],
            "age": [40.0, np.nan, 60.0],
            "sex": ["female", "male", np.nan],
            "bmi": [20.0, 30.0, np.nan],
        }
    )
    transform = DemographicTransformV3.fit(
        train,
        ("age", "sex", "bmi"),
        missing_indicator_fields=("age", "bmi"),
    )
    design = transform.transform(train)
    assert design.shape == (3, 7)
    assert np.isfinite(design).all()
    unknown_column = transform.feature_names.index("sex_unknown")
    assert design[2, unknown_column] == 1
    # Match the benchmark's deterministic lower weighted-median convention.
    assert transform.numeric_medians == {"age": 40.0, "bmi": 20.0}


def test_end_to_end_refit_updates_encoder_and_saves_both_output_scales(tmp_path):
    rng = np.random.default_rng(7)
    subjects = [f"s{index}" for index in range(9)]
    waveforms = rng.normal(size=(9, 8)).astype(np.float32)
    sbp = 115.0 + 8.0 * waveforms[:, 0] + np.arange(9, dtype=float)
    frame = pd.DataFrame(
        {
            "dataset": "Toy",
            "source": "synthetic-test",
            "subject_id": subjects,
            "measurement_id": subjects,
            "segment_id": [f"r{index}" for index in range(9)],
            "sbp": sbp,
            "dbp": sbp - 40.0,
            "age": np.linspace(30, 70, 9),
            "sex": ["female", "male", "female"] * 3,
            "bmi": np.linspace(20, 28, 9),
        }
    )
    split = pd.DataFrame(
        {
            "subject_id": subjects,
            "fold": 0,
            "role": ["train"] * 5 + ["validation"] * 2 + ["test"] * 2,
            "dataset_fingerprint": "test",
        }
    )
    prepared = PreparedWaveforms(
        frame=frame,
        waveforms=waveforms,
        preprocessing_diagnostics={"test": True},
    )

    factory_calls: list[tuple[str, str]] = []

    def factory(model_key: str, device: str):
        factory_calls.append((model_key, device))
        del device
        torch.manual_seed(19)
        return _Encoder(model_key)

    prediction_path, metadata_path = run_finetuning_v3(
        frame,
        split,
        model_key="anyppg",
        condition="finetuned_demographics",
        target="sbp",
        fold=0,
        seed=17,
        output_root=tmp_path,
        device="cpu",
        config=FineTuneConfig(
            batch_size=4,
            max_epochs=1,
            patience=1,
            learning_rates=(1e-3,),
            mixed_precision=False,
            standardize_embedding=True,
            freeze_batchnorm_running_stats=True,
            zero_initialize_output_layer=True,
        ),
        prepared_waveforms=prepared,
        encoder_factory=factory,
    )
    predictions = pd.read_csv(prediction_path)
    metadata = json.loads(metadata_path.read_text())
    assert len(predictions) == 2
    assert np.isfinite(predictions[["y_pred_zscore", "y_pred_mmhg"]]).all().all()
    assert set(predictions["estimator_output_scale"]) == {
        "full_development_pool_zscore"
    }
    assert set(predictions["decoder"]) == {
        "affine_full_development_pool_inverse_zscore"
    }
    assert metadata["selected_epoch"] == 1
    assert metadata["encoder_weight_change_audit"][
        "changed_sampled_parameter_tensors"
    ] > 0
    stabilization = metadata["stabilization"]
    assert stabilization["standardize_embedding"] is True
    assert stabilization["freeze_batchnorm_running_stats"] is True
    assert stabilization["zero_initialize_output_layer"] is True
    assert stabilization["selection_epoch_zero_embedding_standardization"][
        "fitting_scope"
    ] == "selection_outer_train_epoch_zero"
    assert stabilization["selection_epoch_zero_embedding_standardization"][
        "fitted_row_count"
    ] == 5
    assert stabilization["development_epoch_zero_embedding_standardization"][
        "fitting_scope"
    ] == "refit_full_development_epoch_zero"
    assert stabilization["development_epoch_zero_embedding_standardization"][
        "fitted_row_count"
    ] == 7
    assert stabilization["development_batchnorm_running_stats_audit"][
        "running_buffers_unchanged"
    ] is True
    assert len(factory_calls) == 2  # one selection fit plus one fresh refit
    required_prediction_columns = {
        "height_cm",
        "weight_kg",
        "preprocessing_policy",
        "padding_policy",
        "padding_required",
        "source_fidelity",
        "contract_version",
        "demographic_fields",
        "estimator_output",
        "ridge_input_dimension",
        "embedding_standardization",
        "batchnorm_running_stats",
        "output_layer_initialization",
    }
    assert required_prediction_columns.issubset(predictions.columns)

    reusable = selection_artifact_path(
        tmp_path,
        cohort="Toy",
        model_key="anyppg",
        condition="finetuned_demographics",
        target="sbp",
        fold=0,
    )
    assert reusable.exists()
    factory_calls.clear()
    _, reused_metadata_path = run_finetuning_v3(
        frame,
        split,
        model_key="anyppg",
        condition="finetuned_demographics",
        target="sbp",
        fold=0,
        seed=23,
        output_root=tmp_path,
        device="cpu",
        config=FineTuneConfig(
            batch_size=4,
            max_epochs=1,
            patience=1,
            learning_rates=(1e-3,),
            mixed_precision=False,
            standardize_embedding=True,
            freeze_batchnorm_running_stats=True,
            zero_initialize_output_layer=True,
        ),
        prepared_waveforms=prepared,
        encoder_factory=factory,
        reuse_selection=reusable,
    )
    reused_metadata = json.loads(reused_metadata_path.read_text())
    assert len(factory_calls) == 1  # refit only: no candidate selection training
    assert reused_metadata["hyperparameter_selection_reused"] is True
    assert reused_metadata["hyperparameter_selection_seed"] == 17


def test_time_limited_config_rejects_more_than_ten_epochs():
    with pytest.raises(ValueError, match="capped at 10"):
        FineTuneConfig(max_epochs=11).checked()


def test_target_transform_is_explicit_and_reversible():
    frame = pd.DataFrame(
        {"subject_id": ["a", "b", "c"], "sbp": [100.0, 120.0, 140.0]}
    )
    transform = TargetZScoreTransform.fit(frame, "sbp")
    values = frame["sbp"].to_numpy()
    assert np.allclose(transform.decode(transform.encode(values)), values)
    metadata = transform.metadata("outer_training_selection")
    assert metadata["fit_pool_role"] == "outer_training_selection"
    assert metadata["estimator_output_scale"] == "outer_training_pool_zscore"
    assert metadata["decoder"] == "affine_outer_training_pool_inverse_zscore"


def test_ppgbp_mae_averages_repeated_recordings_before_error():
    rows = pd.DataFrame(
        {
            "dataset": ["PPG-BP"] * 3,
            "subject_id": ["a", "a", "b"],
            "measurement_id": ["a", "a", "b"],
            "sbp": [100.0, 100.0, 140.0],
        }
    )
    # Subject a's two predictions cancel only when averaged before calculating
    # its absolute error. Participant errors are therefore 0 and 40 mmHg.
    score = _cohort_mae_mmhg(rows, "sbp", [80.0, 120.0, 100.0])
    assert score == pytest.approx(20.0)
    assert score != pytest.approx(np.mean([20.0, 20.0, 40.0]))


def test_pulsedb_mae_averages_segments_per_measurement_then_macro_subjects():
    rows = pd.DataFrame(
        {
            "dataset": ["PulseDB-Vital"] * 5,
            "subject_id": ["a", "a", "a", "a", "b"],
            "measurement_id": ["a1", "a1", "a2", "a2", "b1"],
            "dbp": [80.0, 80.0, 80.0, 80.0, 80.0],
        }
    )
    # Both measurements for a have zero error after segment averaging. Subject
    # b has 50-mmHg error, so participant-macro MAE is (0 + 50) / 2 = 25.
    score = _cohort_mae_mmhg(rows, "dbp", [60.0, 100.0, 70.0, 90.0, 130.0])
    assert score == pytest.approx(25.0)
    assert score != pytest.approx(np.mean([20.0, 20.0, 10.0, 10.0, 50.0]))


def test_target_and_demographic_statistics_are_participant_balanced():
    rows = pd.DataFrame(
        {
            "dataset": ["BUT PPG"] * 11,
            "subject_id": ["many"] * 10 + ["one"],
            "measurement_id": ["many-m"] * 10 + ["one-m"],
            "segment_id": [f"r{index}" for index in range(11)],
            "sbp": [100.0] * 10 + [200.0],
            "age": [100.0] * 10 + [200.0],
            "sex": ["female"] * 10 + ["male"],
            "bmi": [20.0] * 10 + [40.0],
        }
    )
    weights = _participant_balanced_weight_vector(rows, np.arange(len(rows)))
    # Despite 10:1 row imbalance, each subject has half the objective weight.
    assert weights[:10].sum() == pytest.approx(weights[10:].sum())
    target = TargetZScoreTransform.fit(rows, "sbp", weights)
    demographics = DemographicTransformV3.fit(
        rows, ("age", "sex", "bmi"), sample_weight=weights
    )
    assert target.mean_mmhg == pytest.approx(150.0)
    assert target.standard_deviation_mmhg == pytest.approx(50.0)
    assert demographics.numeric_means["age"] == pytest.approx(150.0)
    assert demographics.numeric_means["bmi"] == pytest.approx(30.0)
