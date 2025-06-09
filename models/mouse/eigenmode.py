import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import brainpy as bp
import brainpy.math as bm

import jax
from jax import numpy as jnp
from brax.io import mjcf
from brax.io import html
from brax.generalized import pipeline

class Body(bp.dyn.NeuDyn):
    def __init__(self):
        super().__init__(size=12)
        try:
            self.sys = mjcf.load('mouse-free.xml')
        except:
            path = os.path.join(os.path.dirname(__file__), 'mouse-free.xml')
            self.sys = mjcf.load(path)
        self.state = bm.Variable(
                pipeline.init(self.sys,
                  jnp.pi/180*jnp.array(
                      [0., +100., 0.]*2 +
                      [0., -100., 0.]*2
                              ),
                    jnp.zeros_like(self.sys.init_q))
                )
    @property
    def angle(self):
        return self.state.value.q * 180 / jnp.pi # type: ignore

    @property
    def output(self):
        breakpoint()
        angle0 = jnp.array([-60., 110., 60.,]*2 + [60., -100., -30.]*2)
        x = angle0 - self.angle
        return 0.001 * jnp.concatenate([
            jax.nn.relu(+ x),
            jax.nn.relu(- x)
            ])

    def update(self):
        dt = bp.share['dt'] * 1e-3 # ms to s
        self.dt = bp.share['dt']
        sys = self.sys.replace(opt=self.sys.opt.replace(timestep=dt))
        angle0 = jnp.array([-60., 110., 60.,]*2 + [60., -100., -30.]*2)
        action = 1e-6 * -(self.angle-angle0)
        self.state.value = pipeline.step(sys, self.state.value, action) # type:ignore
        return self.state
    def render(self, mon, fn='index.html', height=840, js=''):
        states = [jax.tree_util.tree_map(lambda x: x[i], mon) for i in range(mon.q.shape[0])]
        states_nonnan = [x for x in states if not any(jnp.isnan(x.q))]
        if len(states) != len(states_nonnan):
            print('nanstates, only showing', len(states_nonnan), 'out of', len(states))
        states = states_nonnan
        with open(fn, 'w') as f:
            sys = self.sys.replace(opt=self.sys.opt.replace(timestep=self.dt))
            doc = html.render(sys, states, height=height)
            doc = doc.replace('var viewer = new Viewer(domElement, system);',
                              'var viewer = new Viewer(domElement, system); document.viewer=viewer; ' + js)
            print(doc, file=f)

class AngleProprioception(bp.dyn.NeuDyn):
    def __init__(self, pre, post, conn: bp.conn.IJConn, **kwargs):
        assert isinstance(pre, Body)
        super().__init__(pre=pre, post=post, conn=conn, name=kwargs.get("name"))
    def update(self):
        self.post.I_PC.value += total_increments

if __name__ == '__main__':
    duration = 2000
    dt = 1
    net = Body()
    monitors = {
        'body': net.state
            }
    runner = bp.DSRunner(net, monitors=monitors, dt=dt)
    runner.progress_bar = False
    runner._fun_predict = bm.jit(runner._fun_predict)
    runner.run(duration)
    mon = runner.mon['body']
    net.render(runner.mon['body'])
