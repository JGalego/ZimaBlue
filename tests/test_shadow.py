"""The live, read-only shadow twin."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import zimablue as zb


class CaptureController:
    name = "capture"

    def __init__(self, command=None):
        self.command = command or zb.DriveCommand(left=0.18, right=0.22)
        self.readings = {}

    def reset(self, robot):
        self.readings = {}

    def step(self, control_input):
        self.readings = dict(control_input.readings)
        return self.command


def test_a_matching_shadow_has_zero_residuals():
    controller = CaptureController()
    source = zb.Simulation(
        pool="rectangular",
        robot="tracked",
        dirt="clean",
        controller=controller,
        seed=9,
        record=False,
    )
    shadow = zb.ShadowTwin(pool="rectangular", robot="tracked", seed=9)

    for _ in range(20):
        source.step()
        health = shadow.observe(controller.command, controller.readings)

    assert health.healthy
    assert health.residuals
    assert all(stats.maximum == pytest.approx(0.0) for stats in health.residuals.values())
    assert source.state.x == pytest.approx(shadow.simulation.state.x)
    assert source.state.heading == pytest.approx(shadow.simulation.state.heading)
    shadow.close()
    source.backend.close()


def test_shadow_flags_a_persistent_encoder_disagreement():
    controller = CaptureController()
    source = zb.Simulation(
        pool="rectangular",
        robot="tracked",
        dirt="clean",
        controller=controller,
        seed=3,
        record=False,
    )
    shadow = zb.ShadowTwin(
        pool="rectangular",
        robot="tracked",
        seed=3,
        thresholds={"encoder.left": 0.05, "encoder.right": 0.05},
        minimum_samples=5,
    )

    for _ in range(12):
        source.step()
        readings = dict(controller.readings)
        encoder = readings["encoder"]
        readings["encoder"] = replace(encoder, values=encoder.values + np.array([0.2, 0.2]))
        health = shadow.observe(controller.command, readings)

    assert not health.healthy
    assert health.score > 1.0
    assert set(health.anomalies) == {"encoder.left", "encoder.right"}
    assert "anomalies" in health.summary()
    shadow.close()
    source.backend.close()


def test_shadow_follows_measured_loop_timing():
    shadow = zb.ShadowTwin(timestep=0.02)
    health = shadow.observe(zb.DriveCommand.stop(), {}, dt=0.025)
    assert health.time == pytest.approx(0.025)

    with pytest.raises(ValueError, match="finite and positive"):
        shadow.observe(zb.DriveCommand.stop(), {}, dt=0.0)
    shadow.close()


def test_shadow_thresholds_are_physical_positive_magnitudes():
    with pytest.raises(ValueError, match="finite and positive"):
        zb.ShadowTwin(thresholds={"encoder.left": 0.0})
