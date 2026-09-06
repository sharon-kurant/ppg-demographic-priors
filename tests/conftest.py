from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for subject in range(60):
        age = 20 + subject % 45
        sex = "F" if subject % 2 == 0 else "M"
        bmi = 19 + subject % 12
        subject_effect = rng.normal(0, 2)
        for segment in range(3):
            sbp = 90 + 0.65 * age + 0.6 * bmi + subject_effect + rng.normal(0, 1)
            dbp = 55 + 0.28 * age + 0.35 * bmi + subject_effect + rng.normal(0, 1)
            rows.append(
                {
                    "subject_id": f"s{subject:03d}",
                    "segment_id": f"s{subject:03d}_{segment}",
                    "age": age,
                    "sex": sex,
                    "bmi": bmi,
                    "sbp": sbp,
                    "dbp": dbp,
                    "source": "synthetic",
                }
            )
    return pd.DataFrame(rows)
