from pathlib import Path

import pytest


def test_inference_lock_discovery_ignores_nonproject_and_inaccessible_siblings(tmp_path, monkeypatch):
    import pac_act.identity as identity
    from pac_act.safety import inference_lock_paths

    project=tmp_path/'current'; (project/'pac_act').mkdir(parents=True)
    (project/'.git').mkdir()
    (project/'runs').mkdir()
    sibling=tmp_path/'sibling'; (sibling/'pac_act').mkdir(parents=True)
    (sibling/'.git').mkdir()
    lock=sibling/'runs/.act_inference.lock'; lock.parent.mkdir(); lock.write_text('')
    unrelated=tmp_path/'systemd-private'
    (unrelated/'runs').mkdir(parents=True)
    (unrelated/'runs/.act_inference.lock').write_text('')
    inaccessible=tmp_path/'inaccessible'; (inaccessible/'pac_act').mkdir(parents=True)
    (inaccessible/'.git').mkdir()
    (inaccessible/'runs').mkdir()
    blocked_candidate=inaccessible/'runs/.act_inference.lock'

    monkeypatch.setattr(identity,'PROJECT',project)
    original_is_file=Path.is_file
    def is_file(path):
        if path==blocked_candidate:
            raise PermissionError('simulated private directory')
        return original_is_file(path)
    monkeypatch.setattr(Path,'is_file',is_file)

    paths=inference_lock_paths()
    assert project/'runs/.act_inference.lock' in paths
    assert lock in paths
    assert all(unrelated not in path.parents for path in paths)
    assert blocked_candidate not in paths
