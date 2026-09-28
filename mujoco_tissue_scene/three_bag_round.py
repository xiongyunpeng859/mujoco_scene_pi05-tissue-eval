"""Persistent three-object rounds, with bin clearing only between episodes."""
from __future__ import annotations
import numpy as np


class ThreeBagRound:
    def __init__(self, env, rng, region, yaw_jitter, start_state, appearance=None):
        self.env, self.rng, self.region = env, rng, region
        self.yaw_jitter, self.start_state = yaw_jitter, start_state
        self.appearance = appearance
        self.remaining = []
        self.round_index = -1
        self.completed = 0
        self.active = None
        self.failures = 0

    def begin_pick(self):
        env, rng = self.env, self.rng
        if not self.remaining:
            env.reset(options={'state': self.start_state, 'randomize_objects': False})
            self.round_index += 1
            self.completed = 0
            self.remaining = [b['name'] for b in env.config['boxes']]
            if len(self.remaining) != 3:
                raise ValueError('three-bag mode requires exactly three configured bags')
            points = []
            # Circumcircles plus clearance prevent overlap for all sampled yaws.
            radii = [np.linalg.norm(b['size'][:2])/2 for b in env.config['boxes']]
            for restart in range(100):
                points = []
                for i in range(3):
                    for _ in range(1000):
                        candidate = np.array([rng.uniform(*self.region['x']), rng.uniform(*self.region['y'])])/100
                        if all(np.linalg.norm(candidate-old) >= radii[i]+radii[j]+.015 for j,old in enumerate(points)):
                            points.append(candidate)
                            break
                    else:
                        break
                if len(points) == 3:
                    break
            if len(points) != 3:
                raise RuntimeError('cannot fit three separated bags in spawn region')
            self.spawn = []
            for box, xy in zip(env.config['boxes'], points):
                yaw = float(np.radians(rng.uniform(-self.yaw_jitter, self.yaw_jitter)))
                position = [xy[0]-env.config['table']['size'][0]/2,
                            xy[1]-env.config['table']['size'][1]/2,
                            env.config['table']['surface_z']+box['size'][2]/2+.002]
                env.move_object(box['name'], position, yaw)
                self.spawn.append({'name':box['name'], 'xy_cm':(xy*100).tolist(), 'yaw':yaw})
            self.look = self.appearance.apply(int(rng.integers(2**31))) if self.appearance else None
            for _ in range(round(.25/env.model.opt.timestep)):
                env.mujoco.mj_step(env.model, env.data)
            env._time_target = env.data.time
        # Fixed front-to-back order removes arbitrary multi-target action labels.
        if self.active is None:
            self.active = min(self.remaining, key=lambda n: (env.object_points(n).mean(0)[1], env.object_points(n).mean(0)[0]))
        env.set_target(self.active)
        center = env.object_points(self.active).mean(0)
        rotation = env.data.xmat[env.target_body].reshape(3, 3)
        yaw = float(np.arctan2(rotation[1,0], rotation[0,0]))
        box = next(b for b in env.config['boxes'] if b['name']==self.active)
        meta = {'round_index':self.round_index, 'pick_in_round':self.completed,
                'target':self.active, 'remaining_before':list(self.remaining),
                'spawn':self.spawn, 'appearance':self.look}
        return box, center[:2].copy(), yaw, meta

    def finish_pick(self, success):
        if not success:
            self.failures += 1
            if self.failures >= 3:
                raise RuntimeError('three consecutive failed attempts; round retained for diagnosis')
            return
        self.failures = 0
        name = self.active
        self.remaining.remove(name)
        self.completed += 1
        # This transition is outside both episodes; no teleport frame is labelled.
        self.env.move_object(name, [-2-self.completed, -2, .8])
        self.active = None
