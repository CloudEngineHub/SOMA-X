# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from pxr import Gf, Usd, UsdGeom, UsdSkel

from soma.body import SOMALayer
from soma.geometry.rig_utils import (
    apply_joint_orient_local,
    joint_local_to_world,
    precompute_joint_orient,
)
from soma.hand import SOMAHandLayer
from soma.io import load_soma_animation
from soma.procedural_transforms import load_soma_procedural_transform_definition
from soma.units import Unit


def _write_animation(path, *, scale=1.0):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    stage.SetStartTimeCode(10)
    stage.SetEndTimeCode(12)
    stage.SetTimeCodesPerSecond(24)
    animation = UsdSkel.Animation.Define(stage, "/Animation")
    # Deliberately author a different order than the public skeleton.
    animation.CreateJointsAttr().Set(["Root/Hips", "Root"])
    rotation = Gf.Quatf(float(np.sqrt(0.5)), 0, 0, float(np.sqrt(0.5)))
    animation.CreateRotationsAttr().Set([rotation, Gf.Quatf(1)])
    animation.CreateScalesAttr().Set([Gf.Vec3h(scale), Gf.Vec3h(1)])
    translations = animation.CreateTranslationsAttr()
    translations.Set([Gf.Vec3f(0, 100, 0), Gf.Vec3f(0)], 10)
    translations.Set([Gf.Vec3f(200, 100, 0), Gf.Vec3f(0)], 12)
    stage.GetRootLayer().Save()


def test_usd_motion_honors_units_joint_names_and_unkeyed_playback_frames(tmp_path):
    path = tmp_path / "motion.usda"
    _write_animation(path)
    transforms, fps = load_soma_animation(path, ["Root", "Hips"])

    assert transforms.shape == (3, 2, 4, 4)
    assert fps == 24
    np.testing.assert_allclose(transforms[:, 0], np.broadcast_to(np.eye(4), (3, 4, 4)))
    np.testing.assert_allclose(transforms[:, 1, :3, 3], [[0, 1, 0], [1, 1, 0], [2, 1, 0]])
    np.testing.assert_allclose(
        transforms[0, 1, :3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-6
    )


def test_usd_motion_rejects_missing_joints_and_scaled_poses(tmp_path):
    path = tmp_path / "motion.usda"
    _write_animation(path, scale=2.0)
    with pytest.raises(ValueError, match="missing SOMA joints: LeftHand"):
        load_soma_animation(path, ["Root", "LeftHand"])
    with pytest.raises(ValueError, match="does not support joint scaling"):
        load_soma_animation(path, ["Root", "Hips"])


@pytest.mark.parametrize("suffix", [".npy", ".NPY"])
def test_legacy_npy_motion_is_rejected(tmp_path, suffix):
    path = tmp_path / f"custom{suffix}"
    transforms = np.broadcast_to(np.eye(4, dtype=np.float32), (2, 78, 4, 4)).copy()
    with path.open("wb") as handle:
        np.save(handle, transforms)
    with pytest.raises(ValueError, match=r"Legacy \.npy animations are no longer supported"):
        load_soma_animation(path, ["Root", "Hips"])


@pytest.mark.parametrize("representation", ["absolute", "template", "custom", "stored", "lookup"])
def test_soma_motion_uses_selected_reference_without_prepared_identity(tmp_path, representation):
    path = tmp_path / "motion.usda"
    _write_animation(path)
    t_pose_world = torch.eye(4).repeat(3, 1, 1)
    t_pose_world[2, :3, :3] = torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    # Include a non-public transform to check indexing on procedural rigs.
    layer = SOMALayer.__new__(SOMALayer)
    torch.nn.Module.__init__(layer)
    layer._public_joint_names = ["Root", "Hips"]
    layer.device = torch.device("cpu")
    layer.t_pose_world = t_pose_world
    layer.public_transform_joint_indices = torch.tensor([0, 2])
    layer.public_joint_parent_ids = torch.tensor([0, 0])
    layer.root_joint_idx = 1
    layer.output_unit = Unit.CENTIMETERS
    layer._default_reference_pose = None
    reference = t_pose_world[[0, 2]].clone()
    options = {}
    if representation == "absolute":
        options["absolute_pose"] = True
    elif representation in ("custom", "stored", "lookup"):
        reference[1, :3, :3] = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        if representation == "stored":
            layer._default_reference_pose = reference
        elif representation == "lookup":
            layer.get_reference_pose = lambda *, version: reference if version == "0.2.0" else None
            options["reference_pose"] = {"version": "0.2.0"}
        else:
            options["reference_pose"] = reference
    poses, translation, fps = layer.load_motion(path, **options)
    orient, parent_t = precompute_joint_orient(reference, layer.public_joint_parent_ids)
    root = torch.eye(3).view(1, 1, 3, 3).expand(3, -1, -1, -1)
    absolute = torch.cat([root, poses], dim=1)
    if representation != "absolute":
        absolute = apply_joint_orient_local(absolute, orient, parent_t)
    source, _ = load_soma_animation(path, layer.public_joint_names)
    torch.testing.assert_close(absolute, torch.from_numpy(source[..., :3, :3]))
    torch.testing.assert_close(
        translation, torch.tensor([[0.0, 100, 0], [100, 100, 0], [200, 100, 0]])
    )
    assert fps == 24


def test_checked_in_example_animation_uses_authored_playback_range():
    path = Path(__file__).resolve().parents[1] / "assets" / "example_animation.usd"
    motion, fps = load_soma_animation(path, ["Root", "Hips", "LeftHand", "RightHand"])
    assert motion.shape == (931, 4, 4, 4)
    assert fps == 24
    np.testing.assert_allclose(motion[0, 1, :3, 3], [0, 0.998041, -0.04945001], atol=1e-6)


def test_checked_in_example_animation_is_self_contained_animation_only():
    path = Path(__file__).resolve().parents[1] / "assets" / "example_animation.usd"
    stage = Usd.Stage.Open(str(path))
    prims = list(stage.TraverseAll())
    assert len(prims) == 1
    animation = prims[0]
    assert animation.IsA(UsdSkel.Animation)
    assert str(animation.GetPath()) == "/Animation"
    assert stage.GetDefaultPrim() == animation
    assert {attr.GetName() for attr in animation.GetAuthoredAttributes()} == {
        "joints",
        "rotations",
        "scales",
        "translations",
    }
    assert not animation.GetAuthoredRelationships()
    assert not stage.GetRootLayer().subLayerPaths
    assert not stage.GetRootLayer().GetExternalReferences()


def test_checked_in_example_animation_contains_only_public_joint_channels():
    assets = Path(__file__).resolve().parents[1] / "assets"
    definition = load_soma_procedural_transform_definition(
        assets / "SOMA_procedural_transforms.json"
    )
    stage = Usd.Stage.Open(str(assets / "example_animation.usd"))
    animation = UsdSkel.Animation.Get(stage, "/Animation")
    paths = list(animation.GetJointsAttr().Get())
    names = [path.rsplit("/", 1)[-1] for path in paths]
    assert len(names) == len(set(names)) == len(definition.public_joint_names)
    assert set(names) == set(definition.public_joint_names)
    for path in paths:
        parent = path.rpartition("/")[0]
        assert not parent or parent in paths
    for attr in (
        animation.GetRotationsAttr(),
        animation.GetTranslationsAttr(),
        animation.GetScalesAttr(),
    ):
        for time in [Usd.TimeCode.Default(), *attr.GetTimeSamples()]:
            values = attr.Get(time)
            if values is not None:
                assert len(values) == len(paths)


def _write_hand_motion(path, layout):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    stage.SetTimeCodesPerSecond(48)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(1)
    animation = UsdSkel.Animation.Define(stage, "/Animation")
    quarter = float(np.sqrt(0.5))
    z_rotation = Gf.Quatf(quarter, 0, 0, quarter)
    x_rotation = Gf.Quatf(quarter, quarter, 0, 0)
    if layout == "body":
        # Child before ancestors checks source joint ordering independence.
        paths = ["Root/Arm/LeftHand/Finger", "Root/Arm/LeftHand", "Root/Arm", "Root"]
        rotations = [x_rotation, Gf.Quatf(1), z_rotation, Gf.Quatf(1)]
        translations = [[10, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 0]]
    else:
        prefix = "Root/Arm/" if layout == "prefixed_hand" else ""
        paths = [prefix + "LeftHand/Finger", prefix + "LeftHand"]
        rotations = [x_rotation, z_rotation]
        translations = [[10, 0, 0], [0, 200, 0]]
    animation.CreateJointsAttr().Set(paths)
    animation.CreateRotationsAttr().Set(rotations)
    animation.CreateTranslationsAttr().Set([Gf.Vec3f(*v) for v in translations])
    animation.CreateScalesAttr().Set([Gf.Vec3h(1)] * len(paths))
    stage.GetRootLayer().Save()


def _minimal_hand_layer():
    layer = SOMAHandLayer.__new__(SOMAHandLayer)
    torch.nn.Module.__init__(layer)
    layer.rig_data = {"joint_names": ["LeftHand", "Finger"]}
    layer.t_pose_world = torch.eye(4).repeat(2, 1, 1).double()
    # A non-identity wrist reference catches incorrect relative root handling.
    layer.t_pose_world[0, :3, :3] = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    layer.joint_parent_ids = torch.tensor([0, 0])
    layer._default_reference_pose = None
    layer.output_unit = Unit.CENTIMETERS
    return layer


@pytest.mark.parametrize("layout", ["body", "hand", "prefixed_hand"])
@pytest.mark.parametrize("absolute_pose", [False, True])
def test_hand_motion_includes_wrist_transform_once(tmp_path, layout, absolute_pose):
    path = tmp_path / "hand.usda"
    _write_hand_motion(path, layout)
    layer = _minimal_hand_layer()
    poses, translation, fps = layer.load_motion(path, absolute_pose=absolute_pose)
    assert poses.dtype == layer.t_pose_world.dtype
    assert translation.dtype == layer.t_pose_world.dtype
    assert fps == 48
    absolute = poses
    if not absolute_pose:
        orient, parent_t = precompute_joint_orient(layer.t_pose_world, layer.joint_parent_ids)
        absolute = apply_joint_orient_local(poses, orient, parent_t)
    local = torch.eye(4, dtype=poses.dtype).repeat(2, 2, 1, 1)
    local[..., :3, :3] = absolute
    local[:, 0, :3, 3] = translation
    local[:, 1, :3, 3] = torch.tensor([10, 0, 0])
    world = joint_local_to_world(local, [0, 0])
    expected_rotations = torch.tensor(
        [[[0, -1, 0], [1, 0, 0], [0, 0, 1]], [[0, 0, 1], [1, 0, 0], [0, 1, 0]]],
        dtype=poses.dtype,
    )
    torch.testing.assert_close(
        world[..., :3, :3], expected_rotations.expand(2, -1, -1, -1), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        world[..., :3, 3],
        torch.tensor([[[0, 200, 0], [0, 210, 0]]], dtype=poses.dtype).expand(2, -1, -1),
        atol=1e-4,
        rtol=1e-6,
    )


@pytest.mark.parametrize("layer_class", [SOMALayer, SOMAHandLayer])
def test_motion_rejects_reference_with_absolute_pose(tmp_path, layer_class):
    layer = layer_class.__new__(layer_class)
    torch.nn.Module.__init__(layer)
    with pytest.raises(ValueError, match="cannot be combined"):
        layer.load_motion(tmp_path / "unused.usda", absolute_pose=True, reference_pose=torch.eye(3))


def test_body_and_hand_motion_have_same_signature():
    assert inspect.signature(SOMALayer.load_motion) == inspect.signature(SOMAHandLayer.load_motion)


@pytest.mark.parametrize(
    "problem", ["missing_finger", "scaled_ancestor", "ancestor_gap", "ambiguous_name", "wrong_axis"]
)
def test_hand_motion_rejects_invalid_source(tmp_path, problem):
    path = tmp_path / "hand.usda"
    _write_hand_motion(path, "body")
    stage = Usd.Stage.Open(str(path))
    animation = UsdSkel.Animation.Get(stage, "/Animation")
    names = list(animation.GetJointsAttr().Get())
    if problem == "missing_finger":
        names[0] = "Root/Arm/LeftHand/Other"
        match = "missing SOMA joints: Finger"
    elif problem == "scaled_ancestor":
        animation.GetScalesAttr().Set([Gf.Vec3h(1), Gf.Vec3h(1), Gf.Vec3h(2), Gf.Vec3h(1)])
        match = "joint scaling"
    elif problem == "ancestor_gap":
        names[2] = "Elsewhere"
        match = "missing ancestor"
    elif problem == "ambiguous_name":
        names[2] = "Elsewhere/Finger"
        match = "ambiguous"
    else:
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        match = "Y-up"
    animation.GetJointsAttr().Set(names)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match=match):
        _minimal_hand_layer().load_motion(path)
