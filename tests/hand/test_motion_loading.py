# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand motion loading preserves world placement and corrective articulation."""

from pathlib import Path

import pytest
import torch
from pxr import Gf, Usd, UsdGeom, UsdSkel
from scipy.spatial.transform import Rotation

from soma import SOMAHandLayer
from soma.io import load_usd_animation_sequence
from soma.units import Unit

ASSETS = Path(__file__).resolve().parents[2] / "assets"
pytestmark = [pytest.mark.asset_heavy, pytest.mark.cpu]


def _write_motion(path, paths, transforms, meters_per_unit, fps):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(transforms) - 1)
    stage.SetTimeCodesPerSecond(fps)
    animation = UsdSkel.Animation.Define(stage, "/Animation")
    animation.CreateJointsAttr().Set(paths)
    animation.CreateScalesAttr().Set([Gf.Vec3h(1)] * len(paths))
    rotations = animation.CreateRotationsAttr()
    translations = animation.CreateTranslationsAttr()
    for frame, local in enumerate(transforms):
        quaternions = Rotation.from_matrix(local[:, :3, :3].copy()).as_quat()
        rotations.Set([Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])) for q in quaternions], frame)
        translations.Set([Gf.Vec3f(*xyz) for xyz in local[:, :3, 3].tolist()], frame)
    stage.GetRootLayer().Save()


@pytest.fixture(scope="module")
def body_motion(tmp_path_factory):
    source = load_usd_animation_sequence(ASSETS / "example_animation.usd")
    local = source.local_transforms[[15, 150, 500]].copy()
    paths = source.joint_paths
    path_to_index = {path: index for index, path in enumerate(paths)}
    world = local.copy()
    # Independent FK oracle for the checked-in, parent-first source animation.
    for index, path in enumerate(paths):
        parent_path = path.rpartition("/")[0]
        if parent_path:
            parent = path_to_index[parent_path]
            assert parent < index
            world[:, index] = world[:, parent] @ local[:, index]
    directory = tmp_path_factory.mktemp("hand_motion")
    path = directory / "body.usda"
    _write_motion(path, paths, local, source.meters_per_unit, source.time_codes_per_second)
    return source, local, world, path, directory


@pytest.fixture(scope="module", params=["left", "right"])
def hand(request):
    return SOMAHandLayer(
        data_root=ASSETS,
        hand_type=request.param,
        device="cpu",
        mode="dense",
        output_unit=Unit.CENTIMETERS,
        reference_pose={"version": "0.1.0"},
    )


@pytest.fixture(scope="module")
def hand_motion(hand, body_motion):
    source, local, world, body_path, directory = body_motion
    names = [path.rsplit("/", 1)[-1] for path in source.joint_paths]
    indices = [names.index(name) for name in hand.rig_data["joint_names"]]
    paths = [source.joint_paths[index] for index in indices]
    wrist = torch.from_numpy(world[:, indices[0]].copy())
    wrist[:, :3, 3] *= source.meters_per_unit / hand.output_unit.meters_per_unit

    hand_local = local[:, indices].copy()
    hand_local[:, 0] = world[:, indices[0]]
    files = {"body": body_path}
    prefix = paths[0].rpartition("/")[0] + "/"
    for kind, joint_paths in (
        ("hand", [path.removeprefix(prefix) for path in paths]),
        ("prefixed_hand", paths),
    ):
        path = directory / f"{hand.hand_type}_{kind}.usda"
        _write_motion(
            path, joint_paths, hand_local, source.meters_per_unit, source.time_codes_per_second
        )
        files[kind] = path

    # The old playback convention posed an identity wrist, then applied its
    # world transform externally. Keep that independent oracle for geometry.
    rotations = torch.from_numpy(local[:, indices, :3, :3].copy())
    rotations[:, 0] = torch.eye(3)
    hand.prepare_identity(torch.zeros(1, hand.num_shape_components))
    with torch.no_grad():
        local_output = hand.pose(
            rotations, pose2rot=False, absolute_pose=True, apply_correctives=True
        )
    expected = {
        "transforms": wrist[:, None] @ local_output["transforms"],
    }
    for name in ("vertices", "joints"):
        expected[name] = (
            torch.einsum("bij,bvj->bvi", wrist[:, :3, :3], local_output[name])
            + wrist[:, None, :3, 3]
        )
    return files, wrist, expected, source.time_codes_per_second


@pytest.mark.parametrize("source_kind", ["body", "hand", "prefixed_hand"])
@pytest.mark.parametrize("representation", ["absolute", "template", "historical", "default"])
def test_hand_load_motion_matches_external_wrist_playback(
    hand, hand_motion, source_kind, representation
):
    files, wrist, expected, source_fps = hand_motion
    options = {}
    if representation == "absolute":
        options["absolute_pose"] = True
    elif representation == "template":
        options["reference_pose"] = hand.t_pose_world
    elif representation == "historical":
        options["reference_pose"] = {"version": "0.1.0"}

    poses, translation, fps = hand.load_motion(files[source_kind], **options)
    assert poses.shape == (3, 25, 3, 3)
    assert poses.dtype == hand.bind_shape.dtype
    assert poses.device == hand.bind_shape.device
    assert fps == source_fps
    torch.testing.assert_close(translation, wrist[:, :3, 3], atol=2e-5, rtol=2e-6)
    if representation == "absolute":
        torch.testing.assert_close(poses[:, 0], wrist[:, :3, :3], atol=2e-6, rtol=2e-6)

    with torch.no_grad():
        actual = hand.pose(
            poses,
            pose2rot=False,
            global_translation=translation,
            apply_correctives=True,
            **options,
        )
        fk = hand.pose(
            poses, pose2rot=False, global_translation=translation, fk_only=True, **options
        )
    for name in ("vertices", "joints", "transforms"):
        torch.testing.assert_close(actual[name], expected[name], atol=5e-5, rtol=2e-6)
    assert "vertices" not in fk
    torch.testing.assert_close(fk["transforms"], actual["transforms"], atol=5e-5, rtol=2e-6)
