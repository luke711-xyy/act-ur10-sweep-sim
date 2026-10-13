"""Real rollout state-machine tests with controlled physics at its boundary."""
from types import SimpleNamespace
import copy
import numpy as np
import pytest

from sim.config import Config, load_config
from sim.act.rollout import run_action_path
from sim.act.expert import expert_generation_protocol
from sim.act.collect_dataset import _validate_existing_generation_protocol


class ScriptedEnv:
    table_top_z = 0.
    control_dt = .1

    def __init__(self,target=6,height_before_goal=False,overforce=False,
                 stable_loss=False,reopen_count=False,low_force=False):
        self.layout = [object() for _ in range(6)]
        self.ee = SimpleNamespace(tcp_yaw=lambda:0.)
        self.time = 0.
        self._tcp = np.array([.3,0.,0.])
        self.target = target
        self.height_before_goal = height_before_goal
        self.overforce = overforce
        self.stable_loss = stable_loss
        self.reopen_count = reopen_count
        self.low_force = low_force
        self.delivery_seen = False
        self.delivery_time = None

    def tcp(self): return self._tcp.copy()
    def component_positions(self): return np.zeros((6,3))
    def wrench(self): return np.zeros(6)
    def normal_force(self):
        if self._tcp[0] <= .185:
            return 30. if self.overforce else (0. if self.low_force else .1)
        return .1
    def step_control(self,command):
        self._tcp = command.as_array()[:3].copy()
        if self._tcp[0] <= (.25 if self.height_before_goal else .185):
            self._tcp[2] = .005
        self.time += self.control_dt
        if self._tcp[0] <= .105:
            self.delivery_seen = True
            if self.delivery_time is None:
                self.delivery_time = self.time
    def collected_mask(self):
        m = np.zeros(6,dtype=bool)
        if self._tcp[0] <= .185:
            m[:self.target] = True
        if self.reopen_count and .13 < self._tcp[0] < .17:
            m[self.target-1] = False
        if self.stable_loss and self.delivery_time is not None and self.time > self.delivery_time+.01:
            m[self.target-1] = False
        return m
    def partial_collection_overlap_mask(self): return np.zeros(6,dtype=bool)
    def brush_fully_inside_target(self): return self._tcp[0] <= .105
    def brush_table_contact(self): return self._tcp[2] < .001
    def unsafe_robot_collision(self): return False
    def lost_mask(self): return np.zeros(6,dtype=bool)
    def brush_component_contact_mask(self): return np.zeros(6,dtype=bool)


def execute(monkeypatch,env,enabled=True,mode='expert_admittance',failure_mode='',deep=True):
    import sim.act.rollout as rollout
    monkeypatch.setattr(rollout,'_tray_exit_x',lambda _cfg:.10)
    cfg = load_config(overrides=['sim.control_hz=10','act.action_hz=10',
        'episode.stable_time=1.0','episode.max_frames=100',
        f'task.target_count={env.target}',
        'controller.safe_force_dwell_steps=1'])
    cfg.set_path('episode.expert_height_guard_until_exact_count',enabled)
    xs = [.3,.25,.18,.16,.14,.12,.10] if deep else [.3,.25,.18,.16,.14,.12]
    path = np.array([[x,0.,0.,0.] for x in xs])
    return run_action_path(env,cfg,path,collect_observations=False,
        phases=['approach']+['sweep']*(len(path)-1),
        target_indices=np.arange(env.target),contact_z_mode=mode,failure_mode=failure_mode)


@pytest.mark.parametrize('target',range(1,7))
def test_exact_count_releases_height_guard_before_delivery_depth(monkeypatch,target):
    # Keeping the old height abort after exact collection fails this real loop.
    r = execute(monkeypatch,ScriptedEnv(target))
    assert r.success and r.collected == target
    assert any(row['phase']=='sweep' and row['tcp'][2]>.0015 for row in r.trace)
    assert r.trace[-1]['phase'] == 'stability'


def test_height_guard_release_is_latched_even_if_count_temporarily_drops(monkeypatch):
    r = execute(monkeypatch,ScriptedEnv(reopen_count=True))
    assert r.success
    assert any(row['collected']==5 and row['tcp'][2]>.0015 for row in r.trace)


def test_height_abort_before_exact_count_is_unchanged(monkeypatch):
    r = execute(monkeypatch,ScriptedEnv(height_before_goal=True))
    assert not r.success and r.failure_reason == 'contact lost for 100 ms'


def test_explicit_old_policy_still_aborts_after_count(monkeypatch):
    r = execute(monkeypatch,ScriptedEnv(),enabled=False)
    assert not r.success and r.failure_reason == 'contact lost for 100 ms'


def test_overforce_abort_is_not_waived_by_exact_count(monkeypatch):
    r = execute(monkeypatch,ScriptedEnv(overforce=True))
    assert not r.success and r.failure_reason == 'normal force exceeded safety threshold'


def test_delivery_depth_is_still_required(monkeypatch):
    r = execute(monkeypatch,ScriptedEnv(),deep=False)
    assert not r.success and r.failure_reason == 'exact count reached before configured tray depth'


def test_one_second_stability_is_still_required(monkeypatch):
    r = execute(monkeypatch,ScriptedEnv(stable_loss=True))
    assert not r.success and r.failure_reason == 'target count did not remain stable for 1 s'


def test_force_based_admittance_contact_loss_is_not_waived(monkeypatch):
    r = execute(monkeypatch,ScriptedEnv(low_force=True),mode='admittance')
    assert not r.success and r.failure_reason == 'contact lost for 100 ms'


def test_deliberate_failure_generation_keeps_height_guard(monkeypatch):
    r = execute(monkeypatch,ScriptedEnv(),failure_mode='misroute')
    assert not r.success and r.failure_reason == 'contact lost for 100 ms'


def protocols():
    strict = load_config()
    strict.set_path('task.expert_landing_clearance', .04)
    strict.set_path('task.expert_landing_selection', 'waypoint_minimum')
    strict.set_path('episode.expert_height_guard_until_exact_count',False)
    relaxed = strict.copy()
    relaxed.set_path('episode.expert_height_guard_until_exact_count',True)
    return expert_generation_protocol(strict),expert_generation_protocol(relaxed)


def test_new_height_policy_has_distinct_provenance():
    a,b = protocols()
    assert a['name'] == 'coverage_astar_layout_clearance_v2'
    assert b['name'] == 'coverage_astar_layout_clearance_v3'
    assert b['height_guard'] == 'until_exact_count'
    assert 'height_guard' not in a


def test_formal_resume_accepts_unchanged_stricter_v2_successes_only():
    a,b = protocols()
    record = {'episode_id':'saved','episode_kind':'expert','split':'train',
              'success':True,'expert_generation_protocol':a}
    before = copy.deepcopy(record)
    _validate_existing_generation_protocol([record],[b])
    assert record == before
    for field,value in [('success',False),('split','pilot'),('episode_kind','expert_preview')]:
        bad = {**record,field:value}
        with pytest.raises(ValueError,match='generation protocol'):
            _validate_existing_generation_protocol([bad],[b])


def test_reverse_upgrade_and_other_rule_changes_remain_rejected():
    a,b = protocols()
    rec = {'episode_id':'new','episode_kind':'expert','split':'train','success':True,
           'expert_generation_protocol':b}
    with pytest.raises(ValueError,match='generation protocol'):
        _validate_existing_generation_protocol([rec],[a])
    changed = copy.deepcopy(a)
    changed['components']['min_separation'] = .06
    rec['expert_generation_protocol'] = changed
    with pytest.raises(ValueError,match='generation protocol'):
        _validate_existing_generation_protocol([rec],[b])


def test_frozen_config_without_height_flag_keeps_original_protocol():
    cfg = load_config().to_dict()
    cfg['task']['expert_landing_clearance'] = .04
    cfg['task'].pop('expert_landing_selection',None)
    cfg['episode'].pop('expert_height_guard_until_exact_count',None)
    assert expert_generation_protocol(Config(cfg))['name'] == 'coverage_astar_layout_clearance_v2'


def test_parallel_resume_reservation_skips_saved_suffixes_after_a_hole(tmp_path):
    import json
    from sim.act.collect_dataset import EpisodeSpec,_reserve_episode_ids
    rows = [{'episode_id':f'preview_goal_6_{i:04d}'} for i in (1,2,3,4,6)]
    original = ''.join(json.dumps(r)+'\n' for r in rows)
    (tmp_path/'manifest.jsonl').write_text(original)
    pending = [EpisodeSpec('train',6,4201024,'missing_four','independent'),
               EpisodeSpec('train',6,4201536,'missing_six','independent')]
    chosen = _reserve_episode_ids(tmp_path,pending)
    assert chosen[('missing_four',6)] == 'preview_goal_6_0005'
    assert chosen[('missing_six',6)] == 'preview_goal_6_0007'
    assert (tmp_path/'manifest.jsonl').read_text() == original


def test_parallel_reservation_does_not_reuse_unregistered_episode_directories(tmp_path):
    from sim.act.collect_dataset import EpisodeSpec,_reserve_episode_ids
    (tmp_path/'preview_goal_6_0002').mkdir()
    pending = [EpisodeSpec('train',6,1,'first','independent'),
               EpisodeSpec('train',6,257,'second','independent')]
    chosen = _reserve_episode_ids(tmp_path,pending)
    assert list(chosen.values()) == ['preview_goal_6_0001','preview_goal_6_0003']
