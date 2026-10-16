import importlib.metadata
import pytest


def test_execution_refuses_wrong_torch_version_but_inspection_reports_it(monkeypatch):
    from pac_act.lifecycle import inspect, require_runtime
    actual = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, 'version',
                        lambda name:'2.10.0' if name=='torch' else actual(name))
    report = inspect()
    assert report['runtime_compatible'] is False
    assert not report['training_started']
    with pytest.raises(RuntimeError, match='torch'):
        require_runtime()
