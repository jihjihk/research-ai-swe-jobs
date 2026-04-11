from pathlib import Path

import pandas as pd
import pytest

from tests.helpers.imports import load_module


stage11 = load_module("stage11_llm_integrate", "preprocessing/scripts/stage11_llm_integrate.py")


@pytest.mark.unit
def test_run_stage11_copies_stage10_integrated_output(tmp_path):
    input_path = tmp_path / "stage10_llm_integrated.parquet"
    output_path = tmp_path / "stage11_llm_integrated.parquet"
    rows = pd.DataFrame(
        [
            {
                "job_id": "job-1",
                "description_core_llm": "Build APIs.",
                "seniority_final": "entry",
                "seniority_final_source": "llm",
            },
            {
                "job_id": "job-2",
                "description_core_llm": None,
                "seniority_final": "unknown",
                "seniority_final_source": "unknown",
            },
        ]
    )
    rows.to_parquet(input_path, index=False)

    stage11.run_stage11(input_path=input_path, output_path=output_path)

    pd.testing.assert_frame_equal(pd.read_parquet(output_path), rows)
