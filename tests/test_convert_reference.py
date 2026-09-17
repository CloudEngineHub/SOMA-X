# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference conversion preserves articulation for body and both hands."""

import pytest
import torch

from soma.geometry.lbs import batch_rodrigues
from tests.hand.test_reference_poses import _absolute_oracle
from tests.test_constructor_reference_pose import _assert_outputs, _make_layer
from tests.test_reference_poses import _assert_output_close as _assert_body_outputs
from tests.test_reference_poses import _world_fk_oracle

pytestmark = pytest.mark.asset_heavy
HISTORICAL = {"alias": "pre-v0.1.0"}
CURRENT = {"version": "v0.3.0"}


@pytest.fixture(
    scope="module",
    params=[pytest.param("cpu", marks=pytest.mark.cpu)]
    + ([pytest.param("cuda", marks=pytest.mark.gpu)] if torch.cuda.is_available() else []),
)
def device(request):
    return request.param


@pytest.fixture(scope="module", params=["body", "left", "right"])
def layer_kind(request):
    return request.param


@pytest.fixture(scope="module")
def layer(layer_kind, device):
    return _make_layer(layer_kind, device).double()


def _rotations(angles):
    return batch_rodrigues(angles.reshape(-1, 3), dtype=angles.dtype).reshape(
        *angles.shape[:-1], 3, 3
    )


def _angles(layer, layer_kind, batch=2):
    count = 77 if layer_kind == "body" else 25
    return torch.linspace(
        -0.3,
        0.5,
        batch * count * 3,
        dtype=layer.bind_shape.dtype,
        device=layer.bind_shape.device,
    ).reshape(batch, count, 3)


def _reference(angles, layer_kind):
    rotations = _rotations(angles)
    if layer_kind == "body":
        rotations = torch.cat(
            [torch.eye(3, device=angles.device, dtype=angles.dtype)[None], rotations]
        )
    return rotations


def _absolute(rotations, reference, layer, layer_kind):
    if layer_kind == "body":
        return _world_fk_oracle(rotations, reference, layer.output_joint_parent_ids.tolist())[0]
    return _absolute_oracle(rotations, reference, layer.joint_parent_ids.tolist())[0]


@pytest.mark.parametrize("transforms", [False, True])
def test_conversion_preserves_independent_absolute_rotations(layer, layer_kind, transforms):
    angles = _angles(layer, layer_kind)
    rotations = _rotations(angles)
    source = _reference(angles[0] * 0.7, layer_kind)
    target = _reference(angles[0].flip(0) * -0.5, layer_kind)
    expected = _absolute(rotations, source, layer, layer_kind)
    if transforms:

        def as_transform(reference):
            result = torch.eye(4, device=reference.device, dtype=reference.dtype).repeat(
                len(reference), 1, 1
            )
            result[:, :3, :3] = reference
            result[:, :3, 3] = 123.0
            return result

        source_input, target_input = as_transform(source), as_transform(target)
    else:
        source_input, target_input = source, target
    actual = layer.convert_reference(rotations, source_input, target_input)
    torch.testing.assert_close(
        _absolute(actual, target, layer, layer_kind), expected, atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(
        layer.convert_reference(actual, target_input, source_input),
        rotations,
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(layer.convert_reference(rotations, source, source), rotations)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_conversion_preserves_shape_device_dtype_and_input(layer, layer_kind, dtype):
    rotations = _rotations(_angles(layer, layer_kind)).to(dtype=dtype)
    saved = rotations.clone()
    actual = layer.convert_reference(rotations, HISTORICAL, CURRENT)
    assert actual.shape == rotations.shape
    assert actual.device == rotations.device
    assert actual.dtype == rotations.dtype
    torch.testing.assert_close(rotations, saved, atol=0, rtol=0)


def test_alias_hash_and_tensor_selectors_agree(layer, layer_kind):
    rotations = _rotations(_angles(layer, layer_kind))
    record = next(item for item in layer.list_reference_poses() if "asset_revision" in item)
    source = layer.get_reference_pose(asset_revision=record["asset_revision"])
    target = layer.get_reference_pose(**CURRENT)
    torch.testing.assert_close(layer.get_reference_pose(**HISTORICAL), source, atol=0, rtol=0)
    expected = layer.convert_reference(rotations, source, target)
    for selector in (HISTORICAL, {"asset_revision": record["asset_revision"]}):
        torch.testing.assert_close(layer.convert_reference(rotations, selector, CURRENT), expected)


def test_converted_pose_matches_original_reference(layer, layer_kind):
    layer.prepare_identity(
        layer.bind_shape.new_zeros(1, layer.num_shape_components),
        repose_to_bind_pose=layer_kind == "body",
    )
    rotations = _rotations(_angles(layer, layer_kind))
    converted = layer.convert_reference(rotations, HISTORICAL, CURRENT)
    expected = layer.pose(
        rotations, pose2rot=False, reference_pose=HISTORICAL, apply_correctives=False
    )
    # Stored float32 orientations accumulate rounding through body FK.
    assert_outputs = _assert_body_outputs if layer_kind == "body" else _assert_outputs
    assert_outputs(
        layer.pose(converted, pose2rot=False, reference_pose=CURRENT, apply_correctives=False),
        expected,
    )
    assert_outputs(layer.pose(converted, pose2rot=False, apply_correctives=False), expected)


def test_alias_can_be_reused_as_constructor_default(layer_kind, device):
    configured = _make_layer(layer_kind, device, reference_pose=HISTORICAL)
    torch.testing.assert_close(
        configured._default_reference_pose, configured.get_reference_pose(**HISTORICAL)
    )


def test_conversion_gradients_for_rotations_and_both_references(layer, layer_kind):
    angles = _angles(layer, layer_kind, batch=1)
    motion = angles.clone().requires_grad_()
    source = (angles[0] * 0.7).requires_grad_()
    target = (angles[0].flip(0) * -0.5).requires_grad_()

    def convert(motion_angles, source_angles, target_angles):
        return layer.convert_reference(
            _rotations(motion_angles),
            _reference(source_angles, layer_kind),
            _reference(target_angles, layer_kind),
        )

    assert torch.autograd.gradcheck(convert, (motion, source, target), fast_mode=True)
    converted = convert(motion, source, target)
    weights = torch.linspace(-0.3, 0.9, converted.numel(), device=converted.device).reshape_as(
        converted
    )
    gradients = torch.autograd.grad((converted * weights).sum(), (motion, source, target))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.abs().max() > 1e-5


def test_conversion_does_not_require_or_mutate_identity_cache(layer, layer_kind, monkeypatch):
    rotations = _rotations(_angles(layer, layer_kind))
    saved = {}
    for name, value in vars(layer).items():
        if name.startswith("_cached_"):
            saved[name] = value
            monkeypatch.setattr(layer, name, None)
    monkeypatch.setattr(layer, "_default_reference_pose", layer.get_reference_pose(**CURRENT))

    def unexpected_identity(*args, **kwargs):
        raise AssertionError("Reference conversion must not prepare identity")

    monkeypatch.setattr(layer, "prepare_identity", unexpected_identity)
    layer.convert_reference(rotations, HISTORICAL, CURRENT)
    with pytest.raises(KeyError):
        layer.convert_reference(rotations, HISTORICAL, {"alias": "missing"})
    for name in saved:
        assert getattr(layer, name) is None


@pytest.mark.parametrize(
    "problem", ["type", "integer", "unbatched", "count", "axis_angle", "nan", "inf"]
)
def test_invalid_motion_input_fails(layer, layer_kind, problem):
    rotations = _rotations(_angles(layer, layer_kind))
    if problem == "type":
        rotations = rotations.tolist()
    elif problem == "integer":
        rotations = rotations.to(torch.int64)
    elif problem == "unbatched":
        rotations = rotations[0]
    elif problem == "count":
        rotations = rotations[:, :-1]
    elif problem == "axis_angle":
        rotations = rotations[..., 0]
    else:
        rotations[0, 0, 0, 0] = float(problem)
    with pytest.raises((TypeError, ValueError)):
        layer.convert_reference(rotations, HISTORICAL, CURRENT)


@pytest.mark.parametrize("which", ["source", "target"])
@pytest.mark.parametrize(
    "problem", ["none", "empty", "unknown_key", "unknown_alias", "shape", "nan", "nonrotation"]
)
def test_invalid_reference_fails(layer, layer_kind, which, problem):
    rotations = _rotations(_angles(layer, layer_kind))
    invalid = layer.get_reference_pose(**CURRENT)
    if problem == "none":
        invalid = None
    elif problem == "empty":
        invalid = {}
    elif problem == "unknown_key":
        invalid = {"misspelled": "0.1.0"}
    elif problem == "unknown_alias":
        invalid = {"alias": "missing"}
    elif problem == "shape":
        invalid = invalid[:-1]
    elif problem == "nan":
        invalid[1, 0, 0] = float("nan")
    else:
        invalid[1] = 0
    source, target = (invalid, CURRENT) if which == "source" else (HISTORICAL, invalid)
    with pytest.raises((TypeError, ValueError, KeyError)):
        layer.convert_reference(rotations, source, target)
