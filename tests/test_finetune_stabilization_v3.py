from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from ppg_bp_incremental.models.stabilization_v3 import (  # noqa: E402
    BatchNormRunningStatsSnapshot,
    EmbeddingStandardizationStatistics,
    FixedEmbeddingStandardizer,
    freeze_batchnorm_running_stats,
    zero_initialize_scalar_output_layer,
)


def test_epoch_zero_embedding_statistics_are_weighted_fixed_and_floored():
    embeddings = np.asarray(
        [
            [1.0, 10.0, 7.0],
            [3.0, 20.0, 7.0],
            [9.0, 40.0, 7.0],
        ],
        dtype=np.float32,
    )
    rows = pd.DataFrame(
        {
            "subject_id": ["a", "b", "c"],
            "segment_id": ["r0", "r1", "r2"],
        }
    )
    weights = np.asarray([1.0, 2.0, 1.0])
    statistics = EmbeddingStandardizationStatistics.fit(
        embeddings,
        rows,
        sample_weight=weights,
        standard_deviation_floor=1e-3,
        fitting_scope="selection_outer_train_epoch_zero",
    )
    assert statistics.means == pytest.approx((4.0, 22.5, 7.0))
    assert statistics.floored_feature_count == 1
    assert statistics.standard_deviations[2] == pytest.approx(1e-3)
    assert statistics.fitted_row_count == 3
    assert statistics.fitted_subject_count == 3
    assert len(statistics.statistics_sha256) == 64
    assert len(statistics.fitted_row_identity_sha256) == 64

    standardizer = FixedEmbeddingStandardizer(3)
    unchanged = torch.from_numpy(embeddings)
    assert standardizer(unchanged) is unchanged
    standardizer.configure(statistics)
    transformed = standardizer(torch.from_numpy(embeddings)).detach().numpy()
    normalized_weights = weights / weights.sum()
    assert np.average(transformed[:, :2], axis=0, weights=normalized_weights) == pytest.approx(
        np.zeros(2), abs=1e-6
    )
    assert np.sqrt(
        np.average(transformed[:, :2] ** 2, axis=0, weights=normalized_weights)
    ) == pytest.approx(np.ones(2), abs=1e-6)
    assert transformed[:, 2] == pytest.approx(np.zeros(3), abs=1e-6)

    reloaded = FixedEmbeddingStandardizer(3)
    reloaded.load_state_dict(standardizer.state_dict())
    assert reloaded.enabled is True
    assert torch.equal(
        reloaded(torch.from_numpy(embeddings)),
        standardizer(torch.from_numpy(embeddings)),
    )


def test_freeze_batchnorm_stats_keeps_dropout_active_and_affine_trainable():
    class Network(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(4, 4)
            self.batch_norm = torch.nn.BatchNorm1d(4)
            self.dropout = torch.nn.Dropout(0.2)
            self.output = torch.nn.Linear(4, 1)

        def forward(self, values):
            return self.output(self.dropout(self.batch_norm(self.projection(values))))

    model = Network()
    model.train()
    snapshot = BatchNormRunningStatsSnapshot.capture(model)
    summary = freeze_batchnorm_running_stats(model)
    assert summary["module_count"] == 1
    assert summary["affine_parameter_count"] == 8
    assert summary["trainable_affine_parameter_count"] == 8
    assert not model.batch_norm.training
    assert model.dropout.training
    assert model.batch_norm.weight.requires_grad
    assert model.projection.weight.requires_grad

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    model(torch.randn(8, 4)).square().mean().backward()
    assert model.batch_norm.weight.grad is not None
    assert model.projection.weight.grad is not None
    optimizer.step()
    audit = snapshot.compare(model)
    assert audit["running_buffers_unchanged"] is True
    assert audit["maximum_absolute_change"] == 0.0


def test_embedding_standardization_rejects_nontraining_row_misalignment():
    with pytest.raises(ValueError, match="misaligned"):
        EmbeddingStandardizationStatistics.fit(
            np.zeros((2, 4), dtype=np.float32),
            pd.DataFrame({"subject_id": ["a"], "segment_id": ["r0"]}),
            fitting_scope="selection_outer_train_epoch_zero",
        )


def test_zero_output_initialization_starts_at_zero_target_zscore():
    output = torch.nn.Linear(8, 1)
    zero_initialize_scalar_output_layer(output)
    prediction = output(torch.randn(5, 8))
    assert torch.equal(prediction, torch.zeros_like(prediction))
    assert output.weight.requires_grad
    assert output.bias is not None and output.bias.requires_grad
