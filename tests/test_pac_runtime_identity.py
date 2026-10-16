import fcntl
import pytest


def test_local_lease_refuses_real_native_lock_without_waiting_or_stopping_owner(tmp_path):
    from pac_act.safety import local_inference_lease
    lock = tmp_path/'native.lock'
    lock.touch()
    with lock.open('r') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='inference'):
            with local_inference_lease([lock]):
                raise AssertionError('must not enter')
    with local_inference_lease([lock]):
        with lock.open('r') as competitor:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competitor, fcntl.LOCK_EX|fcntl.LOCK_NB)


def test_executable_or_model_drift_is_detected_while_running(tmp_path):
    from pac_act.identity import FileIdentityWatch
    from pac_act.storage import sha256
    path = tmp_path/'weights'
    path.write_bytes(b'base')
    watch = FileIdentityWatch({path: sha256(path)})
    watch()
    path.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='identity drift'):
        watch()


def test_guard_polls_identity_hook_but_does_not_serialize_it(tmp_path):
    from pac_act.safety import Guard
    guard = Guard(tmp_path, started_at=0, deadline=300, collect_deadline=250,
                  free_gib=8, output_gib=5, clock=lambda:100.)
    def fail():
        raise RuntimeError('identity drift')
    guard.identity_check = fail
    with pytest.raises(RuntimeError, match='identity drift'):
        guard.poll(free_bytes=9*2**30, rss_bytes=0, output_bytes=0, pressure=False)
    assert 'identity_check' not in guard.identity()
