"""Unchanged formal trial-4 configuration and fail-closed driver adapters."""
import ast
import inspect
import linecache

def candidate_config(frozen, target):
    if type(target) is not int or not 1 <= target <= 6: raise ValueError('target outside 1–6')
    cfg = frozen.copy()
    changes = {'task.target_count': target, 'controller.desired_force': 1.,
               'controller.admittance.m': 10., 'controller.admittance.b': 400.,
               'controller.admittance.k': 10., 'controller.force_filter_cutoff_hz': 5.,
               'controller.delta_z_limit': .002,
               'controller.delta_z_upward_rate_limit': .02,
               'controller.delta_z_downward_rate_limit': .1}
    for key, value in changes.items(): cfg.set_path(key, value)
    return cfg

def strict_exception_function(function):
    """Process-local copy: normal ACT code unchanged; all exceptions propagate.

    Keep the old handler statements unreachable after a bare raise so the
    reviewed adapter's structural checks still cover the exact original AST.
    No hold command or new prediction is attempted after an exception.
    """
    tree = ast.parse(inspect.getsource(function))
    handlers = [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]
    if not handlers: raise RuntimeError('native ACT exception structure changed')
    for handler in handlers: handler.body.insert(0, ast.Raise())
    source = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    filename = f'<formal_hybrid_fail_closed_{id(function)}>'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(function.__globals__)
    exec(compile(source, filename, 'exec'), namespace)
    return namespace[function.__name__]

def build_formal_act_driver(module, cfg, context):
    from sim.act import evaluate
    original = evaluate.run_act_episode
    try:
        evaluate.run_act_episode = strict_exception_function(original)
        return module.build_act_driver(cfg, context)
    finally:
        evaluate.run_act_episode = original

def validate_acquisition(context):
    if context['step_failures'] or context['invalid_samples']:
        raise RuntimeError('unexpected invalid prefix execution')
    if context['ready']: return
    terminal = context.get('safety_reason') or context.get('collection_reason')
    if terminal:
        known = {repr(RuntimeError(prefix + terminal)) for prefix in (
            'prefix terminated before ACT handoff: ', 'acquisition terminated: ')}
        if context.get('terminal_reason') == terminal and context.get('error') in known: return
        raise RuntimeError('unexpected exception on a terminal acquisition step: ' + str(context.get('error')))
    expected = {
        'stable wrist-force band not established within 30 s; ACT sweep not started',
        'shared frame budget exhausted before stable wrist contact',
        'expert descent ended without reaching 1 mm; no force search/retry permitted',
    }
    if context.get('error') not in {repr(RuntimeError(reason)) for reason in expected}:
        raise RuntimeError('unexpected acquisition exception: ' + str(context.get('error')))
