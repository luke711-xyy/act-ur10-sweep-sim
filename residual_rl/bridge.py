"""Isolated instrumentation helpers; the frozen sim source stays unchanged."""
import ast
from collections import deque
import inspect
import linecache
import textwrap
from types import MethodType

import numpy as np


def contact_mask(env):
    mask=np.zeros(6,dtype=bool)
    brush=set(env.ee.tip_geom_ids)
    for contact in env.data.contact[:env.data.ncon]:
        if float(contact.dist)>0.: continue
        g1,g2=int(contact.geom1),int(contact.geom2)
        part=env.geom_to_component.get(g2 if g1 in brush else g1 if g2 in brush else -1)
        if part is not None: mask[int(part)]=True
    return mask


class DiscardObservations(list):
    def append(self,value):
        pass


class RecentRows(list):
    """Native capture_state needs trace[-1]; full rows stream separately."""
    def append(self,value):
        super().append(value)
        if len(self)>2: del self[:-2]


def compile_copy(function,tree,extra):
    source=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    filename=f'<residual_isolated_{function.__name__}_{id(tree)}>'
    linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    namespace=dict(function.__globals__); namespace.update(extra)
    exec(compile(source,filename,'exec'),namespace)
    return namespace[function.__name__]


def instrument_physics(env,on_substep,after_control,before_control=None):
    original=env.step_control.__func__
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    count=0
    class Instrument(ast.NodeTransformer):
        def visit_Expr(self,node):
            nonlocal count
            if (isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                    and isinstance(node.value.func.value,ast.Name)
                    and node.value.func.value.id=='mujoco' and node.value.func.attr=='mj_step'):
                count+=1
                return [node,*ast.parse('_residual_substep(self)').body]
            return self.generic_visit(node)
    tree=Instrument().visit(tree)
    if count!=1: raise RuntimeError('physical step structure changed; no contact sampling shortcut')
    function=compile_copy(original,tree,{'_residual_substep':on_substep})
    def step(self,command):
        if before_control: before_control(self,command)
        function(self,command)
        after_control(self,command)
    env.step_control=MethodType(step,env)


def streaming_native(function):
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    count=0
    class Stream(ast.NodeTransformer):
        def visit_Assign(self,node):
            nonlocal count
            if (len(node.targets)==1 and isinstance(node.targets[0],ast.Tuple)
                and [getattr(x,'id',None) for x in node.targets[0].elts]==['observations','actions','trace']):
                node.value=ast.parse('(_residual_obs(), [], _residual_rows())',mode='eval').body
                count+=1
            return self.generic_visit(node)
    tree=Stream().visit(tree)
    if count!=1: raise RuntimeError('native trajectory retention structure changed')
    return compile_copy(function,tree,{'_residual_obs':DiscardObservations,'_residual_rows':RecentRows})


def closure_value(function,name):
    values=dict(zip(function.__code__.co_freevars,function.__closure__ or ()))
    if name not in values: raise RuntimeError('frozen controller hook changed: '+name)
    return values[name].cell_contents
