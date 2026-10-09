# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdSkel

from soma.io import load_usd_animation, load_usd_animation_sequence


def _write_animation(path, *, playback_range=True):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.1)
    stage.SetTimeCodesPerSecond(48)
    stage.SetFramesPerSecond(24)
    if playback_range:
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(2)
    animation = UsdSkel.Animation.Define(stage, "/Motion")
    # Distinct joint paths may share a leaf name; no SOMA naming or rig is needed.
    animation.CreateJointsAttr().Set(["Left/joint", "Right/joint"])
    rotation = Gf.Quatf(float(np.sqrt(0.5)), 0, 0, float(np.sqrt(0.5)))
    animation.CreateRotationsAttr().Set([rotation, Gf.Quatf(1)])
    animation.CreateScalesAttr().Set([Gf.Vec3h(2, 3, 4), Gf.Vec3h(1)])
    translations = animation.CreateTranslationsAttr()
    translations.Set([Gf.Vec3f(0, 10, 0), Gf.Vec3f(0)])
    translations.Set([Gf.Vec3f(0, 10, 0), Gf.Vec3f(0)], 0)
    translations.Set([Gf.Vec3f(20, 10, 0), Gf.Vec3f(0)], 2)
    stage.GetRootLayer().Save()


def test_sequence_preserves_general_joint_paths_axes_units_and_scale(tmp_path):
    path = tmp_path / "motion.usda"
    _write_animation(path)
    data = load_usd_animation_sequence(path)

    assert data.joint_paths == ["Left/joint", "Right/joint"]
    assert data.up_axis == "Z"
    assert data.meters_per_unit == 0.1
    assert data.time_codes_per_second == 48
    np.testing.assert_array_equal(data.time_codes, [0, 1, 2])
    assert data.local_transforms.shape == (3, 2, 4, 4)
    np.testing.assert_allclose(
        data.local_transforms[:, 0, :3, 3], [[0, 10, 0], [10, 10, 0], [20, 10, 0]]
    )
    np.testing.assert_allclose(
        data.local_transforms[0, 0, :3, :3], [[0, -3, 0], [2, 0, 0], [0, 0, 4]], atol=1e-6
    )
    np.testing.assert_array_equal(data.scales[:, 0], [[2, 3, 4]] * 3)


def test_sequence_supports_explicit_subframes_and_held_values(tmp_path):
    path = tmp_path / "motion.usda"
    _write_animation(path)
    data = load_usd_animation_sequence(path, time_codes=[-1, 0.25, 1.75, 3])
    np.testing.assert_array_equal(data.time_codes, [-1, 0.25, 1.75, 3])
    np.testing.assert_allclose(data.local_transforms[:, 0, 0, 3], [0, 2.5, 17.5, 20])


def test_sequence_infers_sample_range_without_stage_playback_range(tmp_path):
    path = tmp_path / "motion.usda"
    _write_animation(path, playback_range=False)
    data = load_usd_animation_sequence(path)
    np.testing.assert_array_equal(data.time_codes, [0, 1, 2])


def test_sequence_does_not_overshoot_fractional_end_time(tmp_path):
    path = tmp_path / "motion.usda"
    _write_animation(path)
    stage = Usd.Stage.Open(str(path))
    stage.SetEndTimeCode(1.5)
    stage.GetRootLayer().Save()
    np.testing.assert_array_equal(load_usd_animation_sequence(path).time_codes, [0, 1])


def test_sequence_requires_selection_when_multiple_animations_exist(tmp_path):
    path = tmp_path / "motion.usda"
    _write_animation(path)
    stage = Usd.Stage.Open(str(path))
    UsdSkel.Animation.Define(stage, "/Other")
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="found 2"):
        load_usd_animation_sequence(path)
    data = load_usd_animation_sequence(path, animation_prim_path="/Motion")
    assert len(data.joint_paths) == 2
    with pytest.raises(ValueError, match="No SkelAnimation"):
        load_usd_animation_sequence(path, animation_prim_path="/Missing")


@pytest.mark.parametrize("times", [[], [float("nan")], [[0, 1]]])
def test_sequence_rejects_invalid_sample_times(tmp_path, times):
    path = tmp_path / "motion.usda"
    _write_animation(path)
    with pytest.raises(ValueError, match="time_codes"):
        load_usd_animation_sequence(path, time_codes=times)


def test_existing_default_pose_reader_keeps_its_return_contract(tmp_path):
    path = tmp_path / "motion.usda"
    _write_animation(path)
    rotations, translations = load_usd_animation(path)
    assert rotations.shape == (2, 3, 3)
    np.testing.assert_allclose(rotations[0], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-6)
    np.testing.assert_array_equal(translations, [[0, 10, 0], [0, 0, 0]])
