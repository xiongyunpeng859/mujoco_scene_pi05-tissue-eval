"""Seeded appearance changes, held fixed for an entire three-bag round."""
from __future__ import annotations
import colorsys
import numpy as np


def table_texture(rng, kind, size=512):
    import cv2
    base = np.array(colorsys.hsv_to_rgb(rng.random(), rng.uniform(.05, .45), rng.uniform(.22, .8)))
    y, x = np.mgrid[:size, :size].astype(float) / size
    noise = rng.normal(0, 1, (size, size)).astype(np.float32)
    if kind == 'wood':
        base = np.array([.45, .29, .15]) * rng.uniform(.6, 1.5)
        grain = np.sin(x*220 + 4*np.sin(y*6) + cv2.GaussianBlur(noise, (0, 0), 4)*8)
        signal = .07*grain + .018*noise
    elif kind == 'fabric':
        signal = .04*np.sin(x*390) + .04*np.sin(y*390) + .025*noise
    elif kind == 'stone':
        signal = .12*cv2.GaussianBlur(noise, (0, 0), 4) + .04*noise
    elif kind == 'stripes':
        signal = .055*np.sin(x*rng.uniform(70, 180)) + .012*noise
    else:
        signal = .008*noise
    return np.clip((base[None, None, :] + signal[:, :, None])*255, 0, 255).astype(np.uint8)


class AppearanceRandomizer:
    def __init__(self, env):
        self.env = env
        m = env.model
        self.rgba = m.geom_rgba.copy()
        self.flex_rgba = m.flex_rgba.copy()
        self.diffuse = m.light_diffuse.copy()
        self.arm_geoms = [g for g in range(m.ngeom)
                          if m.body(int(m.geom_bodyid[g])).name in
                          ('base_link', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6')
                          and m.geom_contype[g] == 0]

    def apply(self, seed):
        env, rng = self.env, np.random.default_rng(seed)
        m = env.model
        m.geom_rgba[:] = self.rgba
        m.flex_rgba[:] = self.flex_rgba
        m.light_diffuse[:] = self.diffuse
        near_real = bool(rng.random() < .25)
        kind = 'plain' if near_real else str(rng.choice(['plain', 'wood', 'fabric', 'stone', 'stripes']))
        tex = table_texture(rng, kind)
        if near_real:
            tex = np.clip(rng.normal(7, 1, tex.shape), 0, 255).astype(np.uint8)
        tid = m.texture('dr_table_texture').id
        start = int(m.tex_adr[tid]); count = int(m.tex_width[tid]*m.tex_height[tid]*m.tex_nchannel[tid])
        m.tex_data[start:start+count] = tex.reshape(-1)
        table = m.geom('black_table').id
        m.geom_rgba[table] = [1, 1, 1, 1]
        arm_color = self.rgba[self.arm_geoms[0], :3].copy()
        if not near_real:
            arm_color = np.array(colorsys.hsv_to_rgb(rng.random(), rng.uniform(.1,.65), rng.uniform(.25,.8)))
            for gid in self.arm_geoms:
                m.geom_rgba[gid, :3] = np.clip(arm_color*rng.uniform(.9,1.1), 0, 1)
            for fid in range(m.nflex):
                if m.flex_matid[fid] < 0:
                    m.flex_rgba[fid, :3] = colorsys.hsv_to_rgb(rng.random(), rng.uniform(.18,.7), rng.uniform(.45,.9))
        m.light_diffuse[:] = np.clip(self.diffuse*rng.uniform(.85,1.15), 0, 1)
        arm_kind = 'plain' if near_real else str(rng.choice(['plain','fabric','stone','stripes']))
        arm_tex = table_texture(rng,arm_kind)
        if near_real:
            arm_tex = np.clip(rng.normal(55,2,arm_tex.shape),0,255).astype(np.uint8)
        arm_tid = m.texture('dr_arm_texture').id
        arm_mid = m.material('dr_arm_material').id
        arm_start = int(m.tex_adr[arm_tid])
        m.tex_data[arm_start:arm_start+arm_tex.size] = arm_tex.reshape(-1)
        m.mat_specular[arm_mid] = rng.uniform(.05,.45)
        m.mat_shininess[arm_mid] = rng.uniform(.05,.5)
        for gid in self.arm_geoms:
            m.geom_matid[gid] = arm_mid
            m.geom_rgba[gid,:3] = [1,1,1]
        if env._renderer is not None:
            env.mujoco.mjr_uploadTexture(m, env._renderer._mjr_context, tid)
            env.mujoco.mjr_uploadTexture(m, env._renderer._mjr_context, arm_tid)
        return {'seed': int(seed), 'near_real': near_real, 'table_texture': kind,
                'arm_rgb': arm_color.tolist(), 'arm_texture':arm_kind,
                'arm_texture_mean_rgb':(arm_tex.mean(axis=(0,1))/255).tolist(),
                'arm_specular':float(m.mat_specular[arm_mid]),'arm_shininess':float(m.mat_shininess[arm_mid]),
                'bag_rgba': m.flex_rgba.tolist(),
                'light_diffuse': m.light_diffuse.tolist()}
