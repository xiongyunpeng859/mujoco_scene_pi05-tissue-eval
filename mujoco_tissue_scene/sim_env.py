#!/usr/bin/env python3
"""pi0.5-compatible MuJoCo environment for the sim-to-real tissue scene.

    from sim_env import TissueSceneEnv
    env = TissueSceneEnv()
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action16)

Observations use the dataset's own key names, shapes and dtypes, so pi0.5 needs
no adapter and no extra action representation:

    observation.images.top    uint8 (480, 640, 3)   central UVC camera
    observation.images.left   uint8 (480, 640, 3)   left wrist RealSense
    observation.state         float32 (16,)         absolute joint positions

`step` takes the dataset's 16-D absolute joint-position action and advances
exactly one control period (configured physics rate, 30 Hz control), the same contract the
real robot runs under.  Camera images optionally receive the real lens distortion,
because the real videos contain it and MuJoCo renders an ideal pinhole.

`reset` samples the arm/hand start pose from the 214 real episode start states and
re-draws the three tissue bags inside the measured spawn region -- both without
rebuilding the model, so rollouts stay fast.

Run `python sim_env.py --demo --episode 0` for a self-test that replays a real
episode through `step` and reports the tracking error.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import action_layout as layout          # noqa: E402
import scene                            # noqa: E402

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = Path("/workspace/shared/new_program_qiuzhi/without_tactile/"
                       "pi05_normal_recovery_merged_214eps")


def distortion_map(intrinsics, resolution):
    """Pixel remap turning an ideal pinhole image into the real lens image."""
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    k1, k2, p1, p2, k3 = intrinsics["distortion_coefficients"][:5]
    width, height = resolution
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                 np.arange(height, dtype=np.float32))
    x = (grid_x - cx) / fx
    y = (grid_y - cy) / fy
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return ((xd * fx + cx).astype(np.float32), (yd * fy + cy).astype(np.float32))


class TissueSceneEnv:
    """A thin, honest wrapper: no observation or action reshaping of any kind."""

    metadata = {"render_modes": ["rgb_array"], "control_hz": 30.0}

    def __init__(self, config_path=ROOT / "configs/scene.yaml", render=True,
                 dataset=None, seed=0, apply_distortion=None, output_dir=None):
        import mujoco
        self.mujoco = mujoco
        self.config_path = Path(config_path)
        self.config = scene.apply_measured_layout(yaml.safe_load(self.config_path.read_text()))
        self.env_spec = self.config.get("env", {})
        control = self.config.get("control", {})
        self.control_hz = float(control.get("control_frequency_hz", 30.0))
        self.timestep = float(control.get("physics_timestep_s", 0.002))
        self.output_dir = Path(output_dir or ROOT / "outputs/env_preview")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.xml = scene.build(self.config_path, self.output_dir / "scene.xml", with_hand=True)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml))
        self.data = mujoco.MjData(self.model)
        self._cache_model_handles()

        reset_spec = self.config.get("reset", {})
        self.dataset = Path(dataset or reset_spec.get("dataset") or DEFAULT_DATASET)
        self.jitter = float(reset_spec.get("joint_jitter_rad", 0.0))
        self.settle_seconds = float(reset_spec.get("settle_seconds", 0.5))
        self._start_states = None
        self.rng = np.random.default_rng(seed)

        self.cameras = dict(self.env_spec.get("observation_keys", {}))
        if apply_distortion is None:
            apply_distortion = bool(self.env_spec.get("apply_lens_distortion", True))
        self.apply_distortion = apply_distortion
        self.resolution = tuple(self.env_spec.get("image_resolution", (640, 480)))
        self._maps = {}
        self._renderer = None
        if render:
            self._renderer = mujoco.Renderer(self.model, height=self.resolution[1],
                                             width=self.resolution[0])
        self._time_target = 0.0

    def _cache_model_handles(self):
        mujoco = self.mujoco
        self.ctrl_ids = np.array(layout.actuator_ids(self.model, mujoco))
        self.joint_adr = np.array(layout.joint_ids(self.model, mujoco))
        self.limits_lo = np.array([self.model.jnt_range[self.model.joint(name).id][0]
                                   for name in layout.SIM_JOINTS])
        self.limits_hi = np.array([self.model.jnt_range[self.model.joint(name).id][1]
                                   for name in layout.SIM_JOINTS])
        self.tray_body = self.model.body("tray").id
        self.target_name = "target"
        self.target_body = self.model.body(self.target_name).id
        self.flange_body = self.model.body("link6").id
        self.box_free = {}
        for box in self.config["boxes"]:
            joint = self.model.joint(box["name"] + "_free")
            self.box_free[box["name"]] = (joint.qposadr[0], joint.dofadr[0])

    def set_target(self, name):
        if name not in self.box_free:
            raise KeyError(name)
        self.target_name = name
        self.target_body = self.model.body(name).id

    def object_points(self, name):
        fid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_FLEX, name + "_soft")
        if fid < 0:
            from itertools import product
            box = next(b for b in self.config['boxes'] if b['name'] == name)
            body = self.model.body(name).id
            corners = np.array(list(product((-1, 1), repeat=3))) * np.asarray(box['size']) / 2
            return corners @ self.data.xmat[body].reshape(3, 3).T + self.data.xpos[body]
        start, num = int(self.model.flex_vertadr[fid]), int(self.model.flex_vertnum[fid])
        return self.data.flexvert_xpos[start:start+num]

    def move_object(self, name, position, yaw=0.):
        """Reposition a bag between episodes, including all its elastic nodes."""
        root = self.model.body(name).id
        for j in range(self.model.njnt):
            b = int(self.model.jnt_bodyid[j])
            while b and b != root:
                b = int(self.model.body_parentid[b])
            if b != root:
                continue
            qa, da = int(self.model.jnt_qposadr[j]), int(self.model.jnt_dofadr[j])
            nq, nv = (7, 6) if self.model.jnt_type[j] == self.mujoco.mjtJoint.mjJNT_FREE else (1, 1)
            self.data.qpos[qa:qa+nq] = self.model.qpos0[qa:qa+nq]
            self.data.qvel[da:da+nv] = 0
        qa, _ = self.box_free[name]
        self.data.qpos[qa:qa+3] = position
        self.data.qpos[qa+3:qa+7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        self.mujoco.mj_forward(self.model, self.data)

    # ---------------------------------------------------------------- reset ----
    def episode_start_states(self):
        """The real episode start states, used as the reset distribution."""
        if self._start_states is None:
            states = None
            if self.dataset.is_dir():
                import dataset_io
                episodes, order = dataset_io.load(self.dataset,
                                                  fields=("observation.state",))
                states = np.array([episodes[index]["observation.state"][0] for index in order])
            if states is None:
                states = np.array([list(self.config["arm"]["initial_qpos"]) + [0.0] * 10],
                                  dtype=np.float64)
            self._start_states = states
        return self._start_states

    def place_objects(self, randomize=True):
        """Write the tissue bags' free joints from a spawn draw (no model rebuild)."""
        config = scene.apply_measured_layout(yaml.safe_load(self.config_path.read_text()))
        spec = config.get("box_randomization", {})
        if not spec.get("enabled", False):
            return None
        rng = self.rng if randomize else np.random.default_rng(spec.get("seed", 0) or 0)
        draws, table_size_cm = scene.sample_box_positions(config, rng)
        surface = config["table"]["surface_z"]
        tray_left = scene.tray_left_edge_cm(config)
        half_tray = [config["tray"]["size"][0] * 50.0, config["tray"]["size"][1] * 50.0]
        centre = [(config["tray"]["center"][0] + config["table"]["size"][0] / 2.0) * 100.0,
                  (config["tray"]["center"][1] + config["table"]["size"][1] / 2.0) * 100.0]
        for box, (x_cm, y_cm, yaw) in zip(config["boxes"], draws):
            qpos_adr, dof_adr = self.box_free[box["name"]]
            inside = abs(x_cm - centre[0]) < half_tray[0] and abs(y_cm - centre[1]) < half_tray[1]
            height = box["size"][2]
            self.data.qpos[qpos_adr + 0] = x_cm / 100.0 - table_size_cm[0] / 200.0
            self.data.qpos[qpos_adr + 1] = y_cm / 100.0 - table_size_cm[1] / 200.0
            self.data.qpos[qpos_adr + 2] = surface + (0.01 if inside else 0.0) + height / 2.0 + 0.001
            self.data.qpos[qpos_adr + 3] = np.cos(yaw / 2.0)
            self.data.qpos[qpos_adr + 4] = 0.0
            self.data.qpos[qpos_adr + 5] = 0.0
            self.data.qpos[qpos_adr + 6] = np.sin(yaw / 2.0)
            self.data.qvel[dof_adr:dof_adr + 6] = 0.0
        return [{"name": box["name"], "centre_cm": [round(x, 2), round(y, 2)],
                 "yaw_deg": round(float(np.degrees(yaw)), 1)}
                for box, (x, y, yaw) in zip(config["boxes"], draws)]

    def reset(self, seed=None, options=None):
        mujoco = self.mujoco
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        states = self.episode_start_states()
        if "state" in options:
            start = np.asarray(options["state"], dtype=np.float64)
        else:
            start = states[int(self.rng.integers(len(states)))]
        if self.jitter:
            start = start + self.rng.normal(0.0, self.jitter, size=start.shape)
        start = np.clip(start, self.limits_lo, self.limits_hi)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.joint_adr] = start
        self.data.qvel[:] = 0.0
        placement = self.place_objects(options.get("randomize_objects", True))
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.ctrl_ids] = start
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(round(self.settle_seconds / self.timestep))):
            mujoco.mj_step(self.model, self.data)
        self._time_target = self.data.time
        info = {"reset_state": start.tolist(), "object_placement": placement,
                "episode_start_source": str(self.dataset)}
        return self.observation(), info

    # ------------------------------------------------------------- stepping ----
    def step(self, action):
        mujoco = self.mujoco
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != 16:
            raise ValueError("action must have 16 values, got %d" % action.size)
        if not np.isfinite(action).all():
            raise ValueError("action contains non-finite values")
        clipped = np.clip(action, self.limits_lo, self.limits_hi)
        self.data.ctrl[self.ctrl_ids] = clipped
        self._time_target += 1.0 / self.control_hz
        steps = 0
        while self.data.time < self._time_target - 1e-9:
            before_time = self.data.time
            mujoco.mj_step(self.model, self.data)
            if self.data.time <= before_time:
                raise RuntimeError("MuJoCo reset time after numerical instability; episode invalid")
            steps += 1
            if steps > 1000:
                raise RuntimeError("control step did not advance simulated time")
        achieved = self.data.qpos[self.joint_adr].copy()
        info = {"physics_steps": steps, "time": float(self.data.time),
                "tracking_error": (achieved - clipped).tolist(),
                "action_clipped": bool(np.any(np.abs(action - clipped) > 1e-9)),
                "success": self.success()}
        return self.observation(), 0.0, False, False, info

    def target_hand_contacts(self):
        """Count target contacts with hand collision geometry (including flex)."""
        fid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_FLEX,
                                   self.target_name + "_soft")
        count = 0
        for contact in self.data.contact:
            for side in (0, 1):
                gid = int(contact.geom[side])
                is_target = (fid >= 0 and int(contact.flex[side]) == fid)
                if gid >= 0:
                    is_target |= self.model.geom_bodyid[gid] == self.target_body
                other = int(contact.geom[1-side])
                if is_target and other >= 0:
                    name = self.model.body(int(self.model.geom_bodyid[other])).name or ""
                    count += int(name.startswith("hand_"))
        return count

    def success(self, margin=0.0):
        """Released target fully inside the tray; trajectory checks belong to the controller.

        Include the flex contact radius, inner walls, floor, and rim. A target
        suspended above the tray or penetrating its floor is not successful.
        The rim allows 0.5 mm numerical tolerance; floor contacts allow 2 mm
        solver penetration. Neither tolerance can admit a hovering bag.
        """
        fid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_FLEX,
                                   self.target_name + "_soft")
        if fid >= 0:
            start, num = int(self.model.flex_vertadr[fid]), int(self.model.flex_vertnum[fid])
            points = self.data.flexvert_xpos[start:start+num]
            radius = float(self.model.flex_radius[fid])
        else:
            box = next(b for b in self.config["boxes"] if b["name"] == self.target_name)
            from itertools import product
            corners = np.array(list(product((-1, 1), repeat=3))) * np.asarray(box["size"]) / 2
            points = corners @ self.data.xmat[self.target_body].reshape(3, 3).T + self.data.xpos[self.target_body]
            radius = 0.0
        local = (points - self.data.xpos[self.tray_body]) @ self.data.xmat[self.tray_body].reshape(3, 3)
        tray = self.config["tray"]
        half = np.asarray(tray["size"][:2]) / 2 - tray["wall_thickness"] - margin
        return bool(np.isfinite(local).all()
                    and (np.abs(local[:, :2]) + radius <= half).all()
                    and (local[:, 2] + radius <= tray["size"][2] + 0.0005).all()
                    and (local[:, 2] - radius >= 0.010 - 0.002).all()
                    and self.target_hand_contacts() == 0)

    # ---------------------------------------------------------- observation ----
    def observation(self):
        observation = {"observation.state":
                       self.data.qpos[self.joint_adr].astype(np.float32)}
        if self._renderer is not None:
            for dataset_key, camera in self.cameras.items():
                if dataset_key.startswith("observation.images."):
                    observation[dataset_key] = self.render(camera)
        return observation

    def render(self, camera, apply_distortion=None):
        if self._renderer is None:
            raise RuntimeError("environment was created with render=False")
        self._renderer.update_scene(self.data, camera=camera)
        image = self._renderer.render()
        use = self.apply_distortion if apply_distortion is None else apply_distortion
        if use:
            intrinsics = self._camera_intrinsics(camera)
            if intrinsics and "distortion_coefficients" in intrinsics:
                import cv2
                if camera not in self._maps:
                    self._maps[camera] = distortion_map(intrinsics, self.resolution)
                map_x, map_y = self._maps[camera]
                image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
        return np.ascontiguousarray(image)

    def _camera_intrinsics(self, camera):
        lookup = {"central": ("cameras", "central"), "left_wrist": ("wrist_camera",)}
        entry = self.config
        for key in lookup.get(camera, ()):
            entry = entry[key]
        return entry.get("intrinsics")

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def demo(args):
    import dataset_io
    env = TissueSceneEnv(dataset=args.dataset, seed=args.seed, render=not args.no_render,
                         output_dir=args.output_dir)
    obs, info = env.reset(seed=args.seed)
    print("observation contract:")
    for key, value in obs.items():
        print("  %-26s %-8s %s" % (key, value.dtype, value.shape))
    states = env.episode_start_states()
    print("episode start states available: %s" % (states.shape,))
    print("reset state is a real episode start (or jittered from one): %s"
          % bool(np.min(np.linalg.norm(states - np.array(info["reset_state"]), axis=1))
                 <= 6 * env.jitter + 1e-9))
    print("objects placed: %s" % info["object_placement"])

    if args.episode is not None:
        episodes, order = dataset_io.load(args.dataset)
        action = episodes[args.episode]["action"]
        state = episodes[args.episode]["observation.state"]
        obs, info = env.reset(seed=args.seed, options={"state": state[0]})
        reference = np.zeros((len(state), 3))
        for row in range(len(state)):
            env.data.qpos[env.joint_adr] = state[row]
            env.mujoco.mj_forward(env.model, env.data)
            reference[row] = env.data.xpos[env.flange_body]
        errors, flange = [], []
        for row, value in enumerate(action):
            obs, reward, terminated, truncated, info = env.step(value)
            errors.append(obs["observation.state"] - value)
            flange.append(np.linalg.norm(env.data.xpos[env.flange_body]
                                         - reference[min(row + 1, len(state) - 1)]))
        errors, flange = np.array(errors), np.array(flange)
        print("replayed real episode %d through env.step: %d control steps" %
              (args.episode, len(errors)))
        print("  joint tracking RMS vs commanded : %.5f rad  max %.5f"
              % (np.sqrt((errors ** 2).mean()), np.abs(errors).max()))
        print("  flange error vs recorded path    : RMS %.4f m  max %.4f m  final %.4f m"
              % (np.sqrt((flange ** 2).mean()), flange.max(), flange[-1]))
        print("  final success (bag inside tray)   : %s" % env.success())

    if not args.no_render:
        import cv2
        out = Path(args.output_dir or env.output_dir)
        for key, value in obs.items():
            if key.startswith("observation.images."):
                cv2.imwrite(str(out / (key.split(".")[-1] + ".png")),
                            cv2.cvtColor(value, cv2.COLOR_RGB2BGR))
        print("wrote previews to %s" % out)
    env.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/scene.yaml")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode", type=int, default=None,
                        help="replay this real episode through env.step")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if not args.demo:
        parser.print_help()
        return 0
    return demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
