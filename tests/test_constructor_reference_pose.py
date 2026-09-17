# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constructor references are reusable defaults for body and hand articulation."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from soma import SOMAHandLayer, SOMALayer
from soma.geometry.lbs import batch_rodrigues

ASSETS = Path(__file__).resolve().parents[1] / "assets"
HISTORICAL = {"version": "0.1.0", "data_key": "t_pose_world"}
CURRENT = {"version": "v0.3.0", "data_key": "t_pose_world"}
pytestmark = pytest.mark.asset_heavy


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


def _make_layer(kind, device, **kwargs):
    options = dict(
        data_root=ASSETS,
        device=device,
        lod="xlo",
        mode="dense",
        identity_model_type="soma",
        correctives_model_path=None,
    )
    options.update(kwargs)
    if kind == "body":
        return SOMALayer(**options)
    return SOMAHandLayer(hand_type=kind, **options)


@pytest.fixture(scope="module")
def layers(layer_kind, device):
    plain = _make_layer(layer_kind, device).double()
    calls = []
    cls = type(plain)
    original = cls.get_reference_pose

    def tracked_getter(self, *args, **kwargs):
        calls.append(kwargs.copy())
        return original(self, *args, **kwargs)

    with patch.object(cls, "get_reference_pose", tracked_getter):
        configured = _make_layer(layer_kind, device, reference_pose=HISTORICAL.copy()).double()
        assert calls == [HISTORICAL]
    return plain, configured


@pytest.fixture
def prepared(layers, layer_kind):
    plain, configured = layers
    identity = plain.bind_shape.new_zeros(1, plain.num_shape_components)
    for layer in layers:
        layer.prepare_identity(identity, repose_to_bind_pose=layer_kind == "body")
    count = 77 if layer_kind == "body" else 25
    angles = torch.linspace(
        -0.2, 0.3, count * 3, device=plain.bind_shape.device, dtype=plain.bind_shape.dtype
    ).reshape(1, count, 3)
    return plain, configured, identity, angles


def _assert_outputs(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=2e-6)


def test_constructor_default_is_resolved_once_and_reused(prepared, monkeypatch):
    plain, configured, identity, angles = prepared
    expected = plain.pose(angles, reference_pose=HISTORICAL, apply_correctives=False)

    def unexpected_lookup(*args, **kwargs):
        raise AssertionError("Constructor dictionary must resolve only once")

    monkeypatch.setattr(configured, "get_reference_pose", unexpected_lookup)
    _assert_outputs(configured.pose(angles, apply_correctives=False), expected)
    _assert_outputs(configured.pose(angles, reference_pose=None, apply_correctives=False), expected)
    _assert_outputs(configured(angles, identity, apply_correctives=False), expected)
    _assert_outputs(
        configured(angles, identity, reference_pose=None, apply_correctives=False), expected
    )
    assert HISTORICAL == {"version": "0.1.0", "data_key": "t_pose_world"}


@pytest.mark.parametrize("as_dict", [False, True])
def test_call_override_does_not_replace_constructor_default(prepared, as_dict):
    plain, configured, identity, angles = prepared
    expected_default = configured.pose(angles, apply_correctives=False)
    reference = CURRENT.copy() if as_dict else plain.get_reference_pose(**CURRENT)
    expected_override = plain.pose(angles, reference_pose=reference, apply_correctives=False)
    assert not torch.allclose(expected_default["transforms"], expected_override["transforms"])
    _assert_outputs(
        configured.pose(angles, reference_pose=reference, apply_correctives=False),
        expected_override,
    )
    _assert_outputs(
        configured(angles, identity, reference_pose=reference, apply_correctives=False),
        expected_override,
    )
    _assert_outputs(configured.pose(angles, apply_correctives=False), expected_default)


def test_absolute_pose_bypasses_default_but_rejects_explicit_reference(prepared):
    plain, configured, identity, angles = prepared
    expected = plain.pose(angles, absolute_pose=True, apply_correctives=False)
    _assert_outputs(configured.pose(angles, absolute_pose=True, apply_correctives=False), expected)
    _assert_outputs(
        configured(angles, identity, absolute_pose=True, apply_correctives=False), expected
    )
    for reference in (HISTORICAL, plain.get_reference_pose(**HISTORICAL)):
        with pytest.raises(ValueError, match="absolute_pose"):
            configured.pose(angles, reference_pose=reference, absolute_pose=True)
        with pytest.raises(ValueError, match="absolute_pose"):
            configured(angles, identity, reference_pose=reference, absolute_pose=True)


def test_constructor_reference_moves_with_layer(layers, device):
    _, configured = layers
    reference = configured._default_reference_pose
    assert "_default_reference_pose" in dict(configured.named_buffers())
    assert reference.dtype == torch.float64
    assert reference.device.type == device
    expected = reference.clone()
    configured.float()
    assert configured._default_reference_pose.dtype == torch.float32
    configured.double()
    torch.testing.assert_close(configured._default_reference_pose, expected, atol=1e-7, rtol=0)
    if device == "cuda":
        configured.cpu()
        assert configured._default_reference_pose.device.type == "cpu"
        configured.to(device)
        assert configured._default_reference_pose.device.type == "cuda"


@pytest.mark.parametrize(
    "reference,error",
    [
        ({}, ValueError),
        ({"version": "0.0.0"}, KeyError),
        ({"soma_version": "0.1.0"}, TypeError),
        (torch.eye(3), ValueError),
    ],
)
def test_invalid_constructor_reference_fails(layer_kind, device, reference, error):
    with pytest.raises(error):
        _make_layer(layer_kind, device, reference_pose=reference)


def test_constructor_tensor_preserves_reference_gradients(layer_kind, device):
    count = 78 if layer_kind == "body" else 25
    reference_angles = torch.linspace(-0.1, 0.2, count * 3, device=device).reshape(count, 3)
    reference_angles[0] = 0
    reference_angles.requires_grad_()
    reference = batch_rodrigues(reference_angles)
    layer = _make_layer(layer_kind, device, reference_pose=reference)
    identity = layer.bind_shape.new_zeros(1, layer.num_shape_components)
    pose_count = count - 1 if layer_kind == "body" else count
    angles = layer.bind_shape.new_zeros(1, pose_count, 3)
    actual = layer(angles, identity, apply_correctives=False)
    weights = torch.linspace(-0.5, 0.7, actual["transforms"].numel(), device=device).reshape_as(
        actual["transforms"]
    )
    gradient = torch.autograd.grad((actual["transforms"] * weights).sum(), reference_angles)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max() > 1e-4
