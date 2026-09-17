# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Historical aliases resolve exactly and retain the canonical asset provenance."""

import json
from pathlib import Path

import numpy as np
import pytest

from soma.reference_poses import HISTORY_METADATA_KEY, ReferencePoseHistory


def _history(records):
    metadata = {
        "schema_version": 2,
        "default_reference_id": records[0]["id"],
        "conventions": {"orientation_space": "world"},
        "references": records,
    }
    arrays = {HISTORY_METADATA_KEY: np.array(json.dumps(metadata))}
    for record in records:
        key = record["id"]
        arrays[key] = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3)).copy()
        arrays[key + "__joint_names"] = np.array(["Root", "Hips"])
        arrays[key + "__parent_ids"] = np.zeros(2, dtype=np.int64)
    return arrays


def _record(version="0.1.0", data_key="t_pose_world", aliases=None):
    return {
        "id": f"{data_key}__v{version}",
        "soma_version": version,
        "data_key": data_key,
        "aliases": ["named-reference"] if aliases is None else aliases,
    }


def test_alias_lookup_is_exact_scoped_and_independent():
    first = _record()
    second = _record(data_key="other_pose_world")
    history = ReferencePoseHistory(_history([first, second]))
    assert history.resolve_reference_id(alias="named-reference") == first["id"]
    assert (
        history.resolve_reference_id(alias="named-reference", data_key="other_pose_world")
        == second["id"]
    )
    records = history.list_reference_poses()
    records[0]["aliases"].clear()
    assert history.resolve_reference_id(alias="named-reference") == first["id"]
    for alias in ("unknown", "Named-reference", "named"):
        with pytest.raises(KeyError, match="alias"):
            history.resolve_reference_id(alias=alias)
    with pytest.raises(KeyError, match="alias"):
        history.resolve_reference_id(alias="named-reference", data_key="unknown")


@pytest.mark.parametrize("alias", ["", "space name", " name", "name/part", 7, []])
def test_malformed_alias_rejected_at_lookup_and_load(alias):
    history = ReferencePoseHistory(_history([_record()]))
    with pytest.raises(ValueError, match="alias"):
        history.resolve_reference_id(alias=alias)
    with pytest.raises(ValueError, match="alias"):
        ReferencePoseHistory(_history([_record(aliases=[alias])]))


@pytest.mark.parametrize("aliases", ["name", {"name": True}, None])
def test_alias_metadata_requires_list(aliases):
    record = _record()
    record["aliases"] = aliases
    with pytest.raises(ValueError, match="aliases"):
        ReferencePoseHistory(_history([record]))


@pytest.mark.parametrize("same_record", [False, True])
def test_duplicate_alias_rejected(same_record):
    records = (
        [_record(aliases=["duplicate", "duplicate"])]
        if same_record
        else [_record(), _record("0.2.0")]
    )
    with pytest.raises(ValueError, match="unique"):
        ReferencePoseHistory(_history(records))


@pytest.mark.parametrize(
    "selector",
    [
        {"reference_id": "t_pose_world__v0.1.0"},
        {"soma_version": "0.1.0"},
        {"asset_revision": "sha256:" + "a" * 64},
    ],
)
def test_alias_cannot_mix_with_other_selectors(selector):
    history = ReferencePoseHistory(_history([_record()]))
    with pytest.raises(ValueError, match="exactly one"):
        history.resolve_reference_id(alias="named-reference", **selector)


def test_alias_rejects_invalid_data_key():
    history = ReferencePoseHistory(_history([_record()]))
    with pytest.raises(ValueError, match="data_key"):
        history.resolve_reference_id(alias="named-reference", data_key=[])


def test_current_asset_has_kimodo_alias_without_duplicate_arrays():
    asset = Path(__file__).resolve().parents[1] / "assets" / "SOMA_neutral.npz"
    revision = "sha256:f95daf63a6638b0a890ddb5d646d96da17497f1d8bf3d651bd55c0513671419e"
    with np.load(asset, allow_pickle=False) as arrays:
        history = ReferencePoseHistory(arrays)
        resolved = history.resolve_reference_id(alias="pre-v0.1.0")
        assert resolved == history.resolve_reference_id(asset_revision=revision)
        record = next(
            record for record in history.list_reference_poses() if record["id"] == resolved
        )
        assert record["asset_revision"] == revision
        assert record["aliases"] == ["pre-v0.1.0"]
        assert not any("pre-v0.1.0" in key for key in arrays.files)
    with pytest.raises(ValueError, match="semantic version"):
        history.resolve_reference_id(soma_version="pre-v0.1.0")
