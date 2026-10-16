import json
from pathlib import Path
import subprocess
import sys
import hashlib

import pytest


DINO_IDENTITY = {
    "type": "act_dinov3",
    "vision_backbone": "dinov3_vits16",
    "dino_model_id": "timm/vit_small_patch16_dinov3.lvd1689m",
    "dino_revision": "3bf4720a82ec2066db88137180ff1f83a675cef0",
    "dino_weight_sha256": "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040",
}


def _checkpoint(root, step, *, model_type="act_dinov3", backbone="dinov3_vits16",
                digest=DINO_IDENTITY["dino_weight_sha256"], dino=True):
    root = Path(root)
    root.mkdir(parents=True)
    identity = {
        **DINO_IDENTITY,
        "type": model_type,
        "vision_backbone": backbone,
        "dino_weight_sha256": digest,
    }
    if not dino:
        identity = {"type": "act", "vision_backbone": "resnet18"}
    (root / "config.json").write_text(json.dumps(identity))
    pre_state = "policy_preprocessor_step_3_normalizer_processor.safetensors"
    post_state = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    (root / "policy_preprocessor.json").write_text(json.dumps({"steps": [
        {"registry_name": "normalizer_processor", "state_file": pre_state},
    ]}))
    (root / "policy_postprocessor.json").write_text(json.dumps({"steps": [
        {"registry_name": "unnormalizer_processor", "state_file": post_state},
    ]}))
    contents = {
        "model.safetensors": b"synthetic DINO ACT weights",
        pre_state: b"synthetic state normalizer",
        post_state: b"synthetic action statistics",
    }
    if dino:
        contents["training_state.pt"] = b"synthetic optimizer and scheduler state"
        contents["auxiliary/scheduler.json"] = b'{"step":200000}'
    for name, value in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    return root


def _exclusions():
    return {"complete_known_inputs": True, "seeds": [], "fingerprints": ["known-layout"]}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_exclusion_import_reads_only_manifests_not_images(tmp_path):
    from pac_act.exclusions import extract
    source=tmp_path/'manifest.jsonl'
    source.write_text(json.dumps(dict(seed=4200000,layout_fingerprint='abc',images={'cam':['/does/not/exist.png']}))+'\n')
    result=extract([source])
    assert result['seeds']==[4200000] and result['fingerprints']==['abc']
    assert result['evidence'][0]['bytes']>0


def test_exclusions_include_nested_prior_and_read_complete_active_prefix(tmp_path):
    from pac_act.exclusions import extract
    a=tmp_path/'schedule.json'; a.write_text(json.dumps({'prior_consumed_layouts':[{'seed':3150000,'layout_fingerprint':'x'}]}))
    b=tmp_path/'active.jsonl'; b.write_text('{"seed":5000000,"layout_fingerprint":"y"}\n{"seed":')
    result=extract([a,b])
    assert result['seeds']==[3150000,5000000]
    assert result['fingerprints']==['x','y']
    assert result['evidence'][1]['ignored_incomplete_tail'] is True


def test_copy_of_bundle_source_contains_every_identity_file(tmp_path):
    from pac_act.identity import copy_source_tree,source_files,PROJECT
    destination=tmp_path/'source'
    copy_source_tree(PROJECT,destination)
    assert source_files(destination)==source_files(PROJECT)
    assert (destination/'sim/model/assets').is_dir()
    # Control loader must run with the copied code root, not this checkout.
    p=subprocess.run([sys.executable,'-B','-c','from pac_act.hybrid import load_trial4; f,m=load_trial4(); print(m.PROJECT)'],
        cwd=destination,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert str(destination) in p.stdout


def test_dinov3_200k_bundle_copies_and_hashes_every_checkpoint_file(tmp_path):
    from pac_act.identity import build_bundle

    checkpoint = _checkpoint(tmp_path / "step_200000", 200000)
    bundle = tmp_path / "dinov3-bundle"
    result = build_bundle(bundle, checkpoint, _exclusions())
    manifest = result["manifest"]

    assert manifest["base_step"] == 200000
    assert manifest["model_type"] == "act_dinov3"
    assert manifest["vision_backbone"] == "dinov3_vits16"
    assert manifest["dino_model_id"] == DINO_IDENTITY["dino_model_id"]
    assert manifest["dino_revision"] == DINO_IDENTITY["dino_revision"]
    assert manifest["dino_weight_sha256"] == DINO_IDENTITY["dino_weight_sha256"]
    assert manifest["control_profile_id"] == "dinov3_sync5"
    assert manifest["control_profile_files"] == {
        name: manifest["files"][name]
        for name in ("control_profile.json", "evaluation_config.yaml")
    }
    assert manifest["checkpoint_files"] == {
        str(path.relative_to(checkpoint)): _sha256(path)
        for path in sorted(checkpoint.rglob("*")) if path.is_file()
    }
    assert "training_state.pt" in manifest["checkpoint_files"]
    assert "auxiliary/scheduler.json" in manifest["checkpoint_files"]
    for name, expected in manifest["checkpoint_files"].items():
        copied = bundle / "base_model" / name
        assert copied.is_file()
        assert _sha256(copied) == expected
        assert manifest["files"][f"base_model/{name}"] == expected


@pytest.mark.parametrize("case", [
    "wrong_step", "wrong_type", "wrong_backbone", "wrong_digest", "tampered_file",
])
def test_dinov3_bundle_rejects_wrong_step_type_or_tampered_file(tmp_path, case):
    from pac_act.identity import build_bundle, verify_bundle

    if case == "wrong_step":
        checkpoint = _checkpoint(tmp_path / "step_100000", 100000)
        with pytest.raises(ValueError, match="DINO|200000"):
            build_bundle(tmp_path / "bad-bundle", checkpoint, _exclusions())
        return
    elif case == "wrong_type":
        checkpoint = _checkpoint(tmp_path / "step_200000", 200000, model_type="act")
    elif case == "wrong_backbone":
        checkpoint = _checkpoint(tmp_path / "step_200000", 200000, backbone="resnet18")
    elif case == "wrong_digest":
        checkpoint = _checkpoint(tmp_path / "step_200000", 200000, digest="0" * 64)
    elif case == "tampered_file":
        checkpoint = _checkpoint(tmp_path / "step_200000", 200000)
        result = build_bundle(tmp_path / "dino-bundle", checkpoint, _exclusions())
        (result["model"] / "training_state.pt").write_bytes(b"tampered")
        with pytest.raises(ValueError, match="DINO|200000|pinned|identity"):
            verify_bundle(tmp_path / "dino-bundle")
        return
    with pytest.raises(ValueError, match="DINO|200000|pinned|identity"):
        build_bundle(tmp_path / "bad-bundle", checkpoint, _exclusions())


def test_existing_resnet_100k_bundle_contract_is_unchanged(tmp_path):
    from pac_act.identity import build_bundle

    checkpoint = _checkpoint(tmp_path / "step_100000", 100000, dino=False)
    bundle = tmp_path / "resnet-bundle"
    result = build_bundle(bundle, checkpoint, _exclusions())
    manifest = result["manifest"]

    assert manifest["base_step"] == 100000
    assert set(manifest) == {
        "schema", "base_step", "model", "files", "sources",
        "reference_commit", "schedule",
    }
    assert "model_type" not in manifest and "checkpoint_files" not in manifest
    assert {path.name for path in (bundle / "base_model").iterdir()} == {
        "config.json", "model.safetensors", "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    }
    assert result["identity"]
