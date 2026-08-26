"""Self-calibrating digital twins."""

from __future__ import annotations

import numpy as np
import pytest

import zimablue as zb


def _recording(
    *,
    slope: float = 2.0,
    offset: float = 1.0,
    times: np.ndarray | None = None,
    heading: np.ndarray | None = None,
) -> zb.Recording:
    time = np.asarray(times if times is not None else np.linspace(0.0, 4.0, 21))
    return zb.Recording(
        manifest={"format": "zbr", "schema_version": 1, "ground_truth": True},
        frames={
            "time": time,
            "x": slope * time + offset,
            "y": np.zeros_like(time),
            "heading": np.zeros_like(time) if heading is None else np.asarray(heading),
        },
    )


def test_trajectory_loss_aligns_clocks_and_wraps_heading():
    reference_time = np.linspace(0.0, 4.0, 21)
    candidate_time = np.linspace(0.0, 4.0, 9)
    reference = _recording(
        times=reference_time,
        heading=np.full_like(reference_time, np.pi - 0.01),
    )
    candidate = _recording(
        times=candidate_time,
        heading=np.full_like(candidate_time, -np.pi + 0.01),
    )

    loss = zb.trajectory_loss(reference, candidate)
    assert loss == pytest.approx(0.02**2 / 3.0)


def test_calibrator_recovers_a_known_twin():
    reference = _recording(slope=2.0, offset=1.0)

    def simulate(values):
        return _recording(slope=values["slope"], offset=values["offset"])

    calibrator = zb.TwinCalibrator(
        reference,
        simulate,
        [
            zb.Parameter("slope", 0.5, 3.5, initial=1.0),
            zb.Parameter("offset", -1.0, 3.0, initial=0.0),
        ],
    )
    result = calibrator.fit(seed=12, population=16, generations=45)

    assert result.parameters["slope"] == pytest.approx(2.0, abs=2e-3)
    assert result.parameters["offset"] == pytest.approx(1.0, abs=5e-3)
    assert result.loss < 1e-5
    assert result.evaluations == 16 * 46
    assert [step.loss for step in result.history] == sorted(
        (step.loss for step in result.history), reverse=True
    )


def test_calibration_is_reproducible_and_respects_bounds():
    reference = _recording()

    def simulate(values):
        return _recording(slope=values["slope"])

    calibrator = zb.TwinCalibrator(reference, simulate, [zb.Parameter("slope", 0.0, 4.0)])
    first = calibrator.fit(seed=4, population=8, generations=5)
    second = calibrator.fit(seed=4, population=8, generations=5)

    assert first.parameters == second.parameters
    assert first.loss == second.loss
    assert 0.0 <= first.parameters["slope"] <= 4.0


def test_result_metadata_survives_a_recording_round_trip(tmp_path):
    reference = _recording()
    calibrator = zb.TwinCalibrator(
        reference,
        lambda values: _recording(slope=values["slope"]),
        [zb.Parameter("slope", 1.0, 3.0, initial=2.0)],
    )
    result = calibrator.fit(generations=0, population=4)
    annotated = result.annotate()
    path = annotated.save(tmp_path / "fitted.zbr")
    loaded = zb.Recording.load(path)

    assert loaded.manifest["calibration"]["parameters"] == result.parameters
    assert loaded.manifest["calibration"]["evaluations"] == 4


@pytest.mark.parametrize(
    "parameter",
    [
        lambda: zb.Parameter("", 0.0, 1.0),
        lambda: zb.Parameter("gain", 1.0, 1.0),
        lambda: zb.Parameter("gain", 0.0, 1.0, initial=2.0),
    ],
)
def test_bad_parameter_specs_are_rejected(parameter):
    with pytest.raises(ValueError):
        parameter()
