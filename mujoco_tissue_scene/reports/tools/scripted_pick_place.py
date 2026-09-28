#!/usr/bin/env python3
"""A scripted pick-and-place controller -- no replayed trajectory.

Real demonstration joints seed IK; targets use collision-pad centres with the
passive finger couplings applied. The closed-hand pinch frame remains horizontal
throughout a Cartesian transfer. Object-centre feedback corrects XY during transfer; release holds the arm still.
Only continuously carried, released, settled-in-tray episodes may be collected.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import action_layout as layout
import dataset_io
import scene
import sim_env
sys.path.insert(0, str(ROOT / "reports/tools"))
from lerobot_writer import LerobotWriter

SUCCESS = Path("/workspace/shared/new_program_qiuzhi/without_tactile/success_episode/"
               "pick_up_the_tissue_pack_and_place_it_on_the_right_side_"
               "20260908_merged_all_108eps")
OUT = ROOT / "outputs/scripted_pick_place"
THUMB = "hand_L_thumb_tip"
FINGERS = ["hand_L_index_tip", "hand_L_middle_tip", "hand_L_ring_tip", "hand_L_pinky_tip"]
PITCH = [8, 10, 11, 13, 15]

# waypoint plan: (name, dz above the reference grasp height, hand state, steps to reach)
PLAN = [("approach", 0.09, "open", 30),
        ("descend", 0.0, "open", 22),
        ("close", 0.0, "closed", 10),
        ("lift", 0.14, "closed", 22),
        ("transfer", 0.14, "closed", 34),
        ("open", 0.14, "open", 10),
        ("retreat", 0.15, "open", 22)]


def phase_frames(steps, speed):
    """Shorten physical trajectories at fixed 30 Hz; never speed up the video clock."""
    return max(1, round(steps * 3 / speed))


def episode_frames(speed):
    return sum(phase_frames(p[3], speed) for p in PLAN) + phase_frames(30, speed) + 15


def rotation_distance(a, b):
    """Shortest SO(3) angle between palm rotations, independent of Euler wrapping."""
    return float(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))


class Arm:
    """FK/IK helper on a private model, so the env is never disturbed."""

    def __init__(self, config_path):
        import mujoco
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene.build(
            config_path, OUT / "ik.xml", with_hand=True)))
        self.data = mujoco.MjData(self.model)
        self.adr = np.array(layout.joint_ids(self.model, mujoco))
        self.thumb = self.model.body(THUMB).id
        self.base = self.model.body("base_link").id
        self.fingers = [self.model.body(n).id for n in FINGERS]
        # ACTUAL fingertip CONTACT geometry: the distal-phalanx collision geoms, not the
        # last-joint BODY origins (which sit 8-10 cm behind the skin).  This is the root
        # fix for the thumb ending up under the pack: we aim the real skin, not the joint.
        self.tip_geoms = []
        for _w in ("thumb_dip", "index_dip", "middle_dip", "ring_dip", "pinky_dip"):
            _g = -1
            for _i in range(self.model.ngeom):
                _b = self.model.body(int(self.model.geom_bodyid[_i])).name or ""
                if _b.startswith("hand_") and _w in _b and self.model.geom_contype[_i] != 0:
                    _g = _i
                    break
            self.tip_geoms.append(_g)
        if any(g < 0 for g in self.tip_geoms):
            raise ValueError("missing distal finger collision geometry")
        arm_joint_names = layout.SIM_ARM_JOINTS
        self.arm_qadr = np.array([self.model.joint(n).qposadr[0]
                                  for n in arm_joint_names])
        self.hand_qadr = np.array([self.model.joint(n).qposadr[0]
                                   for n in layout.SIM_HAND_JOINTS])
        self.lo = np.array([self.model.jnt_range[self.model.joint(n).id][0]
                            for n in arm_joint_names])
        self.hi = np.array([self.model.jnt_range[self.model.joint(n).id][1]
                            for n in arm_joint_names])

    def set_arm(self, q):
        self.data.qpos[self.arm_qadr] = q
        self.mujoco.mj_kinematics(self.model, self.data)

    def set_hand(self, h):
        reference_hand = getattr(self, 'contact_reference_hand', None)
        if reference_hand is not None:
            h = reference_hand
        self.data.qpos[self.hand_qadr] = np.asarray(h, dtype=float)
        # Kinematics does not enforce equality constraints. Populate passive mimic
        # joints explicitly, just as the dynamics solver does during execution.
        for k in range(self.model.neq):
            if self.model.eq_type[k] == self.mujoco.mjtEq.mjEQ_JOINT:
                j1, j2 = self.model.eq_obj1id[k], self.model.eq_obj2id[k]
                x = self.data.qpos[self.model.jnt_qposadr[j2]] if j2 >= 0 else 0.0
                self.data.qpos[self.model.jnt_qposadr[j1]] = np.polynomial.polynomial.polyval(
                    x, self.model.eq_data[k, :5])
        self.mujoco.mj_kinematics(self.model, self.data)

    def tip_mid(self):
        pts = self.data.geom_xpos[self.tip_geoms]
        if getattr(self, 'thumb_center_distance', None) is not None:
            direction = pts[1:].mean(axis=0)-pts[0]
            return pts[0]+direction/np.linalg.norm(direction)*self.thumb_center_distance
        return 0.5 * (pts[0] + pts[1:].mean(axis=0))

    def closing_dir(self):
        pts = self.data.geom_xpos[self.tip_geoms]
        return pts[1:].mean(axis=0) - pts[0]

    def solve(self, target_xy, target_z, want_yaw, seed, q_ref, hand=None):
        """Place the closed-hand pad frame, with fingers pointing downwards."""
        from scipy.optimize import least_squares
        tgt = np.array([*target_xy, target_z])
        want = np.array([np.cos(want_yaw), np.sin(want_yaw), 0.0])
        if hand is not None:
            self.set_hand(hand)

        def residual(q):
            self.set_arm(q)
            c = self.closing_dir()
            c /= np.linalg.norm(c)
            palm = self.data.xpos[self.model.body("hand_L_palm").id]
            reach = self.tip_mid() - palm
            reach -= c * (reach @ c)
            reach /= np.linalg.norm(reach)
            return np.r_[(self.tip_mid() - tgt) * 20, (c - want),
                         (reach - np.array([0, 0, -1.])), (q - q_ref) * 0.001,
                         (q - seed) * getattr(self, 'continuity_weight', 0.)]

        res = least_squares(residual, np.clip(seed, self.lo+1e-8, self.hi-1e-8),
                            bounds=(self.lo, self.hi), max_nfev=250,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        self.set_arm(res.x)
        err = float(np.linalg.norm(self.tip_mid() - tgt))
        return res.x, err, float(np.max(np.abs(res.x - seed)))


def smooth(a, b, n):
    t = np.linspace(0.0, 1.0, n + 2)[1:-1][:, None]
    # Sinusoidal acceleration ramps with a constant-velocity middle. Peak
    # velocity is 4/3 of the mean (cosine position easing needs pi/2), so the
    # 1.5x trajectory avoids excessive per-frame joint-command peaks.
    ramp = .25
    def start_curve(u):
        return (u - ramp / np.pi * np.sin(np.pi * u / ramp)) / (2 * (1-ramp))
    s = np.where(t < ramp, start_curve(t),
                 np.where(t > 1-ramp, 1-start_curve(1-t), (t-ramp/2)/(1-ramp)))
    return a[None, :] + (b - a)[None, :] * s


def main() -> int:
    global OUT
    import mujoco
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/scene.yaml")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--bags-per-round", type=int, choices=[1,3], default=1)
    parser.add_argument("--randomize-appearance", action="store_true")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--successes", type=int, default=None, help="stop after this many accepted episodes")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--speed", type=float, default=1.0, help="trajectory speed multiplier at fixed 30 Hz")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-every", type=int, default=12, help="preview sampling interval in control frames")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--descend-dz", type=float, default=0.0)
    parser.add_argument("--grasp-yaw-offset", type=float, default=0.0)
    parser.add_argument("--grasp-yaw-plan", type=float, nargs='+', default=None,
                        help="preview-only absolute world grasp yaws, one per pick")
    parser.add_argument("--object-yaw-deg", type=float, default=None, help="single-object diagnostic orientation")
    parser.add_argument("--grasp-policy", choices=["legacy", "least-motion"], default="legacy",
                        help="least-motion is kinematic screening only; validate dynamics before selecting data")
    parser.add_argument("--closure-scale", type=float, default=1.0, help="diagnostic interpolation from open to demonstrated closed hand")
    parser.add_argument("--thumb-closure-scale", type=float, default=1.0)
    parser.add_argument("--thumb-centered", action='store_true', help="experimental side-midpoint contact frame with width-matched reference aperture")
    parser.add_argument("--contact-squeeze", type=float, default=.15, help="closure fraction beyond width-matched aperture")
    parser.add_argument("--grasp-center-offset", type=float, nargs=2, default=[0.,0.], metavar=('DX','DY'), help="diagnostic grasp-frame offset in world metres; object placement unchanged")
    parser.add_argument("--release-dz", type=float, default=0.0)
    parser.add_argument("--collect", type=str, default=None)
    parser.add_argument("--empty-grasp-prefix", action="store_true",
                        help="Preview recovery from an intentionally offset empty grasp; export only recovery actions")
    parser.add_argument("--failure-mode", choices=["empty_grasp", "corner_grasp", "slip",
                        "knock_away", "taken_away", "strange_position", "closed_empty",
                        "target_moves", "execution_error"], default=None)
    parser.add_argument("--miss-offset", type=float, nargs=2, default=[-.16, 0.])
    parser.add_argument("--miss-drift", type=float, default=.04)
    parser.add_argument("--recovery-delay-frames", type=int, default=0,
                        help="delay recovery trigger into the failed prefix (0..18 frames)")
    parser.add_argument('--diversify-failure', action='store_true')
    parser.add_argument("--max-wrist-deg", type=float, default=None,
                        help="whole-episode joint4/5/6 excursion from actual episode start, degrees")
    parser.add_argument("--validated-grasp-plan", type=Path, default=None,
                        help="complete source/config-bound offline plan for a three-pick collection round")
    args = parser.parse_args()
    if args.failure_mode is not None:
        args.empty_grasp_prefix = True
    if not 0 <= args.recovery_delay_frames <= 18:
        parser.error('--recovery-delay-frames must be in [0,18]')
    validated_plan = None
    if args.validated_grasp_plan is not None:
        from grasp_plan_contract import validate_collection_plan, digest
        if args.grasp_yaw_plan is not None:
            parser.error("Do not combine raw yaw angles and a validated plan")
        validated_plan = validate_collection_plan(args.validated_grasp_plan, args.config,
            args.seed, args.speed, args.episodes, args.bags_per_round, args.randomize_appearance)
        args.grasp_yaw_plan = validated_plan['yaws']
        if args.max_wrist_deg is not None and args.max_wrist_deg != validated_plan['max_wrist_deg']:
            parser.error("Wrist limit must match validated plan")
        args.max_wrist_deg = validated_plan['max_wrist_deg']
    if args.max_wrist_deg is not None and (not np.isfinite(args.max_wrist_deg) or not 1 < args.max_wrist_deg <= 180):
        parser.error("--max-wrist-deg must be in (1,180]")
    if args.save_every < 1:
        parser.error("--save-every must be positive")
    if not 0 < args.closure_scale <= 1:
        parser.error("--closure-scale must be in (0,1]")
    if not 0 < args.thumb_closure_scale <= 1:
        parser.error("--thumb-closure-scale must be in (0,1]")
    if not np.isfinite(args.contact_squeeze) or not 0 <= args.contact_squeeze <= 1:
        parser.error("--contact-squeeze must be in [0,1]")
    if not np.isfinite(args.grasp_center_offset).all():
        parser.error("--grasp-center-offset must be finite")
    if args.object_yaw_deg is not None and args.bags_per_round != 1:
        parser.error("--object-yaw-deg is only for single-object diagnostics")
    if args.grasp_yaw_plan is not None and (len(args.grasp_yaw_plan) < args.episodes or not np.isfinite(args.grasp_yaw_plan).all()):
        parser.error("--grasp-yaw-plan needs one finite angle per episode")
    if args.collect and (args.grasp_policy != 'legacy' or args.object_yaw_deg is not None
                         or args.closure_scale != 1 or args.thumb_closure_scale != 1
                         or any(args.grasp_center_offset) or args.thumb_centered
                         or args.grasp_yaw_offset != 0 or args.descend_dz != 0 or args.release_dz != 0
                         or (args.max_wrist_deg is not None and validated_plan is None and not args.empty_grasp_prefix)
                         or (args.grasp_yaw_plan is not None and validated_plan is None)):
        parser.error("Experimental grasp options are preview-only until validated")
    if not np.isfinite(args.speed) or not 0 < args.speed <= 1.5:
        parser.error("--speed must be positive and at most 1.5")
    OUT = args.output_dir
    if args.no_render and (args.save or args.collect):
        parser.error("--no-render cannot be combined with --save or --collect")
    OUT.mkdir(parents=True, exist_ok=True)
    config = scene.apply_measured_layout(yaml.safe_load(args.config.read_text()))
    if args.randomize_appearance:
        config["domain_randomization"] = {"enabled": True}
    path = OUT / "scene.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    surface = config["table"]["surface_z"]
    ts = config["table"]["size"]
    tray = config["tray"]
    rng = np.random.default_rng(args.seed)

    # --- grasp posture and hand states, straight from real data
    data, order = dataset_io.load(SUCCESS, fields=("observation.state", "action"))
    st, ac = data[0]["observation.state"], data[0]["action"]
    closed_frames = np.where((np.abs(ac[:, PITCH]) > 0.35).all(axis=1))[0]
    q_ref = st[int(closed_frames[0])][:6]
    hand_open = ac[0][6:].copy()
    hand_closed = ac[int(closed_frames[len(closed_frames) // 2])][6:].copy()
    hand_closed = hand_open + args.closure_scale * (hand_closed-hand_open)
    hand_closed[2] = hand_open[2] + args.thumb_closure_scale*(hand_closed[2]-hand_open[2])
    print("q_ref (arm joints at a real closure frame): %s" % np.round(q_ref, 4))
    print("hand open/closed pitch sample: %s / %s"
          % (np.round(hand_open[PITCH[0] - 6:PITCH[0] - 5], 2),
             np.round(hand_closed[PITCH[0] - 6:PITCH[0] - 5], 2)))

    arm = Arm(path)
    original_arm_lo, original_arm_hi = arm.lo.copy(), arm.hi.copy()
    arm.set_hand(hand_closed)
    arm.set_arm(q_ref)
    ref = arm.tip_mid()
    print("reference fingertip midpoint: (%.3f, %.3f, %.3f); %.1f cm above the table"
          % (ref[0], ref[1], ref[2], (ref[2] - surface) * 100))
    boxes = config["boxes"]
    bx_world, by_world = tray["center"][0], tray["center"][1]
    writer = LerobotWriter(args.collect, task=("Previous grasp failed. Regrasp the tissue pack and place it inside the green box on the right."
                          if args.empty_grasp_prefix else "pick up the tissue pack and place it inside the green box on the right")) if args.collect else None
    preview_writer = LerobotWriter(OUT / 'full_preview_dataset') if args.empty_grasp_prefix and args.collect else None

    env = sim_env.TissueSceneEnv(config_path=path, dataset=SUCCESS, render=not args.no_render,
                                 output_dir=OUT)
    if not args.collect:
        env.cameras = {}  # Render diagnostics only at sampled frames.
    model, mdata = env.model, env.data
    home = np.clip(st[0], env.limits_lo, env.limits_hi).copy()
    import json
    _lib = json.loads((ROOT / "outputs/closure_library/library.json").read_text())
    release_lib = _lib["release"]
    library = _lib["grasp"]
    rel_x = np.array([p["x_cm"] for p in release_lib])
    rel_y = np.array([p["y_cm"] for p in release_lib])
    lib_x = np.array([p["x_cm"] for p in library])
    lib_y = np.array([p["y_cm"] for p in library])
    print("closure library: %d real grasp postures, x %.1f..%.1f, y %.1f..%.1f"
          % (len(library), lib_x.min(), lib_x.max(), lib_y.min(), lib_y.max()))
    region = config["box_randomization"]["region_xy_cm_from_left_bottom"]
    # Restrict the existing spawn region to the demonstrated XY span.
    region = {"x": [max(region["x"][0], float(lib_x.min())),
                    min(region["x"][1], float(lib_x.max()))],
              "y": [max(region["y"][0], float(lib_y.min())),
                    min(region["y"][1], 66.0)]}
    yaw_jit = float(config["box_randomization"].get("yaw_jitter_deg", 25.0))
    print("sampling packs from x %s cm, y %s cm, yaw jitter %.0f deg"
          % (region["x"], region["y"], yaw_jit))
    print()
    print("  %-4s %-18s %8s %8s %8s %9s %s" %
          ("ep", "pack cm, yaw", "ik err", "peak", "track", "success", "frames"))
    from domain_randomization import AppearanceRandomizer
    from three_bag_round import ThreeBagRound
    appearance = AppearanceRandomizer(env) if args.randomize_appearance else None
    round_manager = ThreeBagRound(env, rng, region, yaw_jit, st[0], appearance) if args.bags_per_round == 3 else None
    rows = []
    nominal_closed = hand_closed.copy()
    for e in range(args.episodes):
        hand_closed = nominal_closed.copy()
        arm.contact_reference_hand = None
        arm.thumb_center_distance = None
        if round_manager is not None:
            tgt, xy, pyaw, episode_meta = round_manager.begin_pick()
            px, py = xy
            px_cm, py_cm = (px+ts[0]/2)*100, (py+ts[1]/2)*100
        else:
            px_cm = float(rng.uniform(region["x"][0], region["x"][1]))
            py_cm = float(rng.uniform(region["y"][0], region["y"][1]))
            pyaw = float(np.radians(rng.uniform(-yaw_jit, yaw_jit)))
            if args.object_yaw_deg is not None:
                pyaw = float(np.radians(args.object_yaw_deg))
            px = px_cm / 100.0 - ts[0] / 2.0
            py = py_cm / 100.0 - ts[1] / 2.0

            env.reset(options={"state": st[0], "randomize_objects": False})
            # Restore elastic node positions/velocities before teleporting carriers.
            for j in range(model.njnt):
                if not model.joint(j).name:
                    mdata.qpos[model.jnt_qposadr[j]] = model.qpos0[model.jnt_qposadr[j]]
            mdata.qvel[:] = 0
            for box in boxes:                      # park everything, then place the target
                badr, bdof = env.box_free[box["name"]]
                mdata.qpos[badr + 0], mdata.qpos[badr + 1] = -2.0 - boxes.index(box), -2.0
                mdata.qpos[badr + 2] = surface + box["size"][2] / 2.0
                mdata.qpos[badr + 3] = 1.0
                mdata.qpos[badr + 4:badr + 7] = 0.0
                mdata.qvel[bdof:bdof + 6] = 0.0
            tgt = boxes[0]
            badr, bdof = env.box_free[tgt["name"]]
            mdata.qpos[badr + 0] = px
            mdata.qpos[badr + 1] = py
            mdata.qpos[badr + 2] = surface + tgt["size"][2] / 2.0 + 0.002
            mdata.qpos[badr + 3] = np.cos(pyaw / 2.0)
            mdata.qpos[badr + 4:badr + 6] = 0.0
            mdata.qpos[badr + 6] = np.sin(pyaw / 2.0)
            mdata.qvel[bdof:bdof + 6] = 0.0
            mujoco.mj_forward(model, mdata)

            env.set_target(tgt["name"])
            episode_meta = {"appearance": appearance.apply(int(rng.integers(2**31))) if appearance else None}
        episode_arm_start = mdata.qpos[env.joint_adr][:6].copy()
        arm.lo, arm.hi = original_arm_lo.copy(), original_arm_hi.copy()
        if args.max_wrist_deg is not None:
            # Reserve one degree for tracking transients, but acceptance checks
            # actual movement against the declared (not enlarged) limit.
            radius = np.radians(args.max_wrist_deg - 1.)
            arm.lo[3:6] = np.maximum(arm.lo[3:6], episode_arm_start[3:6]-radius)
            arm.hi[3:6] = np.minimum(arm.hi[3:6], episode_arm_start[3:6]+radius)
        arm.continuity_weight = .015 if args.max_wrist_deg is not None else 0.
        if "sampled_domain" in config:
            episode_meta['sampled_domain'] = config['sampled_domain']
        if validated_plan is not None:
            episode_meta['grasp_plan'] = {'path': str(args.validated_grasp_plan.resolve()),
                'sha256': digest(args.validated_grasp_plan),
                'max_pregrasp_palm_deg': validated_plan['max_pregrasp_palm_deg'],
                'selection': validated_plan['selection']}
        episode_meta['closure_scale'] = args.closure_scale
        episode_meta['max_wrist_deg'] = args.max_wrist_deg
        episode_meta['wrist_reference_rad'] = episode_arm_start[3:6].tolist()
        episode_meta['thumb_closure_scale'] = args.thumb_closure_scale
        episode_meta['grasp_yaw_offset_deg'] = args.grasp_yaw_offset

        # --- solve the waypoints
        px += args.grasp_center_offset[0]
        py += args.grasp_center_offset[1]
        episode_meta['grasp_center_offset_m'] = args.grasp_center_offset
        want_yaw = pyaw + np.pi / 2.0 + np.radians(args.grasp_yaw_offset)
        if args.grasp_yaw_plan is not None:
            want_yaw = float(np.radians(args.grasp_yaw_plan[e]))
            episode_meta['planned_grasp_yaw_deg'] = args.grasp_yaw_plan[e]
            if validated_plan is not None:
                from grasp_candidate_selection import grasp_family, relative_grasp_angle_deg
                episode_meta['grasp_family'] = validated_plan['families'][e]
                episode_meta['grasp_relative_to_long_axis_deg'] = relative_grasp_angle_deg(
                    args.grasp_yaw_plan[e], np.degrees(pyaw))
                if grasp_family(args.grasp_yaw_plan[e], np.degrees(pyaw)) != validated_plan['families'][e]:
                    raise RuntimeError('validated grasp family no longer matches object pose')
        near = int(np.argmin(np.hypot(lib_x - px_cm, lib_y - py_cm)))
        entry = library[near]
        bx_cm = (bx_world + ts[0] / 2.0) * 100.0
        by_cm = (by_world + ts[1] / 2.0) * 100.0
        near_r = int(np.argmin(np.hypot(rel_x - bx_cm, rel_y - by_cm)))
        rel = release_lib[near_r]
        box_yaw = float(tray["yaw"]) + np.pi / 2.0
        seed_pack = np.array(entry["q"], dtype=float)
        seed_box = np.array(rel["q"], dtype=float)
        ref_z_pack = surface + tgt["size"][2] / 2 + 0.005
        ref_z_box = surface + tray["size"][2] + tgt["size"][2] / 2 + 0.035
        if args.grasp_policy == 'least-motion':
            candidates = []
            offsets = set(np.radians(np.arange(-180,180,15)).tolist())
            # Include current pinch direction, not only object-relative axes.
            arm.set_hand(hand_closed)
            arm.set_arm(mdata.qpos[env.joint_adr][:6])
            current_direction = arm.closing_dir()
            offsets.add(float(np.arctan2(current_direction[1],current_direction[0])-pyaw))
            for offset in sorted(offsets):
                candidate_yaw = pyaw + offset
                qa, ea, _ = arm.solve((px,py), ref_z_pack+.09, candidate_yaw,
                                      seed_pack, q_ref, hand=hand_closed)
                qg, eg, _ = arm.solve((px,py), ref_z_pack+args.descend_dz, candidate_yaw,
                                      qa, q_ref, hand=hand_closed)
                direction=arm.closing_dir(); direction=direction/np.linalg.norm(direction)
                alignment=float(np.dot(direction,[np.cos(candidate_yaw),np.sin(candidate_yaw),0]))
                arm.set_hand(hand_open)
                points=arm.data.geom_xpos[arm.tip_geoms]
                span=float(abs((points[1:].mean(0)-points[0]) @ direction))
                width=float(abs(np.cos(offset))*tgt['size'][0]+abs(np.sin(offset))*tgt['size'][1])
                # A soft pack may compress slightly, but reject clearly oversized grasps.
                feasible=max(ea,eg)<.005 and alignment>.97 and width <= span+.015
                travel=float(np.linalg.norm(qa-mdata.qpos[env.joint_adr][:6])+.5*np.linalg.norm(qg-qa))
                candidates.append(dict(yaw_deg=float(np.degrees(candidate_yaw)),
                    offset_deg=float(np.degrees(offset)),width_m=width,open_span_m=span,
                    alignment=alignment,ik_error_m=max(ea,eg),score=travel,feasible=bool(feasible)))
            feasible=[c for c in candidates if c['feasible']]
            if not feasible:
                raise RuntimeError(f'No feasible grasp orientation: {candidates}')
            choice=min(feasible,key=lambda c:c['score'])
            want_yaw=float(np.radians(choice['yaw_deg']))
            episode_meta['grasp_selection']={'policy':'least-motion','selected':choice,'candidates':candidates}
            print('      grasp selection: '+json.dumps(episode_meta['grasp_selection']),flush=True)
        if args.thumb_centered:
            relative = want_yaw-pyaw
            width = abs(np.cos(relative))*tgt['size'][0]+abs(np.sin(relative))*tgt['size'][1]
            # Target the thumb collision-pad centre just outside the side midpoint.
            # Fit a reference aperture to the object's width, instead of solving
            # IK with fully closed fingers already intersecting the object.
            pad_clearance = .007
            aperture = width + 2*pad_clearance
            fits=[]
            for fraction in np.linspace(0.,1.,101):
                hand = hand_open+fraction*(hand_closed-hand_open)
                arm.set_hand(hand)
                span = float(np.linalg.norm(arm.closing_dir()))
                fits.append((abs(span-aperture),fraction,span,hand.copy()))
            _,fraction,span,reference = min(fits,key=lambda row:row[0])
            arm.contact_reference_hand=reference
            arm.thumb_center_distance=width/2+pad_clearance
            hand_closed=hand_open+min(1.,fraction+args.contact_squeeze)*(nominal_closed-hand_open)
            episode_meta['thumb_centering']=dict(width_m=float(width),reference_fraction=float(fraction),
                                                reference_span_m=span,thumb_to_center_m=float(arm.thumb_center_distance))
            print('      thumb centering: '+json.dumps(episode_meta['thumb_centering']),flush=True)
        solutions, worst, per_way = [], 0.0, []
        for name, dz, hand, steps in PLAN:
            over_pack = name in ("approach", "descend", "close", "lift")
            xy = (px, py) if over_pack else (bx_world, by_world)
            if name == "retreat":
                xy = (bx_world - 0.20, by_world + 0.10)
            s0 = seed_pack if over_pack or name == "transfer" else seed_box
            if args.max_wrist_deg is not None and solutions:
                s0 = solutions[-1][0].copy()
            # Preserve the pinch orientation during the entire carry.
            yaw_here = want_yaw
            z_here = (ref_z_pack if over_pack else ref_z_box) + dz
            if name in ("transfer", "open"):
                z_here = ref_z_pack + 0.14
            if name == "retreat":
                z_here = surface + 0.23
            if name in ("descend", "close", "lift"):
                z_here += args.descend_dz
            if name in ("transfer", "open", "retreat"):
                z_here += args.release_dz
            q, err, moved = arm.solve(
                xy, z_here, yaw_here, s0, q_ref,
                hand=hand_closed)
            if over_pack:
                seed_pack = q.copy()
            else:
                seed_box = q.copy()
            worst = max(worst, err)
            per_way.append((name, err, moved))
            solutions.append((q, hand, steps, name))
        print("      nearest real posture ep%d at (%.1f,%.1f) %.1f cm away; z %.3f" % (entry["episode"], entry["x_cm"], entry["y_cm"],
              float(np.hypot(entry["x_cm"] - px_cm, entry["y_cm"] - py_cm)),
              entry["z_m"]))
        print("      pack seed ep%d (%.1f,%.1f) %.1f cm away | box seed ep%d "
              "(%.1f,%.1f) %.1f cm from the box"
              % (entry["episode"], entry["x_cm"], entry["y_cm"],
                 float(np.hypot(entry["x_cm"] - px_cm, entry["y_cm"] - py_cm)),
                 rel["episode"], rel["x_cm"], rel["y_cm"],
                 float(np.hypot(rel["x_cm"] - bx_cm, rel["y_cm"] - by_cm))))
        print("      per-waypoint IK: " + "  ".join("%s %.3f" % (w[0], w[1])
                                                    for w in per_way))

        if args.empty_grasp_prefix:
            # Plan an empty closure beside the real object. No object/state edits
            # occur between this prefix and the subsequent recovery trajectory.
            miss_xy = np.array([px, py]) + np.asarray(args.miss_offset)
            failure_mode = args.failure_mode or 'empty_grasp'
            dr_rng = np.random.default_rng(args.seed*1009+e)
            failure_dr = dict(lift_height=.10, time_scale=1., force_start=0,
                force_frames=6, force_xy=[.3,-.4], error_joint=0, error_deg=4., corner_xy=[.045,.025])
            if args.diversify_failure:
                angle = float(dr_rng.uniform(-np.pi,np.pi))
                strength = float(dr_rng.uniform(.3,.8))
                failure_dr.update(lift_height=float(dr_rng.uniform(.06,.15)),
                    time_scale=float(dr_rng.uniform(.85,1.5)),force_start=int(dr_rng.integers(0,16)),
                    force_frames=int(dr_rng.integers(3,10)),
                    force_xy=[strength*np.cos(angle),strength*np.sin(angle)],
                    error_joint=int(dr_rng.integers(0,6)),error_deg=float(dr_rng.choice([-1,1])*dr_rng.uniform(2,8)),
                    corner_xy=[float(dr_rng.choice([-1,1])*dr_rng.uniform(.025,.055)),
                               float(dr_rng.choice([-1,1])*dr_rng.uniform(.015,.035))])
            episode_meta['failure_randomization'] = failure_dr
            if failure_mode in ('slip', 'taken_away'):
                miss_xy = np.array([px, py])
            elif failure_mode == 'corner_grasp':
                miss_xy = np.array([px, py]) + np.asarray(failure_dr['corner_xy'])
            elif failure_mode == 'knock_away':
                miss_xy = np.array([px, py]) + np.array([.045, 0.])
            episode_meta['failure_mode'] = failure_mode
            toward_bin = np.array([bx_world-px, by_world-py])
            toward_bin /= np.linalg.norm(toward_bin)
            prefix = []
            seed_q = mdata.qpos[env.joint_adr][:6].copy()
            for name, xy0, dz, hand, steps in [
                ('miss_approach', miss_xy, .09, 'open', 35),
                ('miss_descend', miss_xy, 0., 'open', 22),
                ('miss_close', miss_xy, 0., 'closed', 12),
                ('miss_lift', miss_xy, .10, 'closed', 22),
                ('miss_drift', miss_xy+toward_bin*args.miss_drift, .10, 'closed', 18),
                ('recovery_open', miss_xy+toward_bin*args.miss_drift, .10, 'open', 12),
            ]:
                if name in ('miss_lift','miss_drift','recovery_open'):
                    dz = failure_dr['lift_height']
                steps = max(1,round(steps*failure_dr['time_scale']))
                seed_q, error, _ = arm.solve(xy0, ref_z_pack+dz, want_yaw,
                                             seed_q, q_ref, hand=hand_closed)
                worst = max(worst, error)
                prefix.append((seed_q.copy(), hand, steps, name))
            # Seed recovery IK from the planned prefix endpoint rather than
            # the demonstration posture. Execution remains continuous; this
            # is not an observation-feedback recovery policy.
            seed_q = prefix[-1][0].copy()
            rec = []
            for name, xy0, dz, hand, steps in [
                ('regrasp_approach', (px, py), .09, 'open', 35),
                ('regrasp_descend', (px, py), 0., 'open', 22),
                ('regrasp_close', (px, py), 0., 'closed', 12),
            ]:
                seed_q, error, _ = arm.solve(xy0, ref_z_pack+dz, want_yaw,
                                             seed_q, q_ref, hand=hand_closed)
                worst = max(worst, error)
                rec.append((seed_q.copy(), hand, steps, name))
            solutions = prefix + rec + [s for s in solutions if s[3] not in
                                        ('approach', 'descend', 'close')]
            if failure_mode in ('slip', 'taken_away', 'corner_grasp'):
                # Opening after lift releases the bag through contact dynamics;
                # never reposition the object to manufacture a drop.
                drift = next(i for i,s in enumerate(solutions) if s[3]=='miss_drift')
                q0, _, steps0, name0 = solutions[drift]
                solutions[drift] = (q0, 'open', steps0+30, name0)
            if args.recovery_delay_frames:
                drift = next(i for i,s in enumerate(solutions) if s[3]=='miss_drift')
                q0, h0, steps0, name0 = solutions[drift]
                solutions[drift] = (q0, h0, steps0+args.recovery_delay_frames*args.speed/3, name0)

        # --- interpolate into a 30 Hz action stream and execute
        hand_vec = {"open": hand_open, "closed": hand_closed}
        # ease from where the arm ACTUALLY is after reset, not from a fixed posture
        actions, prev, seg_bounds = [], mdata.qpos[env.joint_adr].copy(), []
        for q, hand, steps, wname in solutions:
            if wname in ("open", "retreat"):
                # Cartesian transfer may finish on another wrist IK branch than
                # the coarse waypoint solve. Continue from its actual last command.
                arm.set_hand(hand_closed)
                arm.set_arm(q)
                point = arm.tip_mid().copy()
                q, path_error, _ = arm.solve(point[:2], point[2], want_yaw,
                                               prev[:6], q_ref, hand=hand_closed)
                worst = max(worst, path_error)
            target16 = np.concatenate([q, hand_vec[hand]])
            seg_start = len(actions)
            if wname == "transfer":
                # A long joint-space interpolation rolls the hand between valid
                # endpoints. Follow a Cartesian path to preserve the pinch frame.
                arm.set_hand(hand_closed)
                arm.set_arm(prev[:6])
                start_point = arm.tip_mid().copy()
                arm.set_arm(q)
                end_point = arm.tip_mid().copy()
                seed_q = prev[:6].copy()
                for point in smooth(start_point, end_point, phase_frames(steps, args.speed)):
                    seed_q, path_error, _ = arm.solve(point[:2], point[2], want_yaw,
                                                       seed_q, q_ref, hand=hand_closed)
                    worst = max(worst, path_error)
                    actions.append(np.r_[seed_q, hand_closed])
            else:
                for row in smooth(prev, target16, phase_frames(steps, args.speed)):
                    actions.append(row)
            seg_bounds.append((wname, seg_start, len(actions)))
            prev = np.asarray(actions[-1]).copy()
        # Return physically to the same real-demonstration start pose every episode.
        # The final half-second hold is kept at real time for settling/verification.
        seg_start = len(actions)
        actions.extend(smooth(prev, home, phase_frames(30, args.speed)))
        seg_bounds.append(("return_home", seg_start, len(actions)))
        seg_start = len(actions)
        actions.extend(home.copy() for _ in range(15))
        seg_bounds.append(("home_settle", seg_start, len(actions)))
        start_z = float(mdata.qpos[env.box_free[tgt["name"]][0] + 2])
        recovery_start = next((a for name, a, b in seg_bounds if name == 'recovery_open'), 0)
        episode_meta['recovery_start_frame'] = recovery_start
        episode_meta['miss_offset_m'] = args.miss_offset if args.empty_grasp_prefix else None
        episode_meta['miss_drift_m'] = args.miss_drift if args.empty_grasp_prefix else None
        episode_meta['failure_delay_frames'] = args.recovery_delay_frames
        miss_contact_frames = 0
        failure_trace = []
        fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, tgt["name"] + "_soft")
        ep_frames, ep_states, ep_actions = [], [], []
        table_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "black_table")
        # only REAL collision geoms of the fingertips: the hand carries duplicate-named
        # visual geoms (contype 0) and three of them report a bogus 0.0000 distance to the
        # table at any arm height, which pinned every per-segment minimum to zero.
        hand_gids = [g for g in range(model.ngeom)
                     if (model.body(int(model.geom_bodyid[g])).name or "").startswith("hand_")
                     and model.geom_contype[g] != 0]
        skin = {}
        fsum = {}
        peak, track = 0.0, 0.0
        arm_track, arm_command_step = 0.0, 0.0
        command_travel = np.zeros(6)
        actual_travel = np.zeros(6)
        actual_joint_step = np.zeros(6)
        command_step_limit = np.radians(3.) if args.max_wrist_deg is not None else .14
        wrist_actual_excursion = np.zeros(3)
        wrist_command_excursion = np.zeros(3)
        previous_actual = mdata.qpos[env.joint_adr][:6].copy()
        palm_id = model.body("hand_L_palm").id
        initial_palm_rotation = mdata.xmat[palm_id].reshape(3, 3).copy()
        previous_palm_rotation = initial_palm_rotation.copy()
        pregrasp_palm_path = 0.0
        pregrasp_palm_max = 0.0
        rate_limited_frames = 0
        phase_command_steps = {}
        previous_command = mdata.qpos[env.joint_adr].copy()
        worst_vec = None
        worst_step, worst_pairs, worst_clip = -1, [], False
        worst_cmd = None
        first_div = None
        pos_rows = []
        hp = {}
        end_contacts = {}
        thumb_contact_trace = []
        segment_end = {}
        carry_contacts = []
        final_success = []
        tipz = []
        surf_low = []
        frames = []
        center_correction = np.zeros(2)
        transfer_start = next(a for name, a, b in seg_bounds if name == "transfer")
        transfer_end = next(b for name, a, b in seg_bounds if name == "transfer")
        release_arm_commands = []
        home_errors = []
        return_contacts = 0
        obstacle_contact_frames = 0
        robot_gids = {g for g in range(model.ngeom)
                      if (model.body(int(model.geom_bodyid[g])).name or "").startswith("hand_")
                      or model.body(int(model.geom_bodyid[g])).name in ("base_link", "link1", "link2", "link3", "link4", "link5", "link6")}
        previous_obs = env.observation() if writer is not None else None
        for step_i, value in enumerate(actions):
            if args.empty_grasp_prefix and step_i == recovery_start:
                # Privileged simulation teacher: observe the physical target
                # and actual joints after failure, without resetting either.
                va, vn = int(model.flex_vertadr[fid]), int(model.flex_vertnum[fid])
                recovery_center = mdata.flexvert_xpos[va:va+vn].mean(0).copy()
                if args.failure_mode not in (None, 'empty_grasp', 'closed_empty', 'execution_error'):
                    points_xy = mdata.flexvert_xpos[va:va+vn,:2] - recovery_center[:2]
                    _, axes = np.linalg.eigh(points_xy.T @ points_xy)
                    observed_yaw = float(np.arctan2(axes[1,-1], axes[0,-1]))
                    observed_yaw += np.pi * round((pyaw-observed_yaw)/np.pi)
                    want_yaw = observed_yaw + np.pi/2
                    episode_meta['recovery_observed_yaw_rad'] = observed_yaw
                measured_state = mdata.qpos[env.joint_adr].copy()
                episode_meta['recovery_observed_center_m'] = recovery_center.tolist()
                episode_meta['recovery_observed_state'] = measured_state.tolist()
                replanned = []
                prev_recovery = measured_state.copy()
                for phase_name, begin, end in seg_bounds:
                    if begin < recovery_start:
                        continue
                    nframes = end-begin
                    if phase_name in ('return_home', 'home_settle'):
                        target = home.copy()
                    elif phase_name == 'recovery_open':
                        target = np.r_[prev_recovery[:6], hand_open]
                    else:
                        base_phase = phase_name.removeprefix('regrasp_')
                        above_pack = base_phase in ('approach', 'descend', 'close', 'lift')
                        xy_target = recovery_center[:2] if above_pack else np.array([bx_world, by_world])
                        z_target = recovery_center[2] + .005
                        z_target += {'approach':.09, 'descend':0., 'close':0.,
                                     'lift':.14, 'transfer':.14, 'open':.14}.get(base_phase, 0.)
                        if base_phase == 'retreat':
                            xy_target = np.array([bx_world-.20, by_world+.10])
                            z_target = surface+.23
                        qnew, errnew, _ = arm.solve(xy_target, z_target, want_yaw,
                                                    prev_recovery[:6], q_ref, hand=hand_closed)
                        worst = max(worst, errnew)
                        target = np.r_[qnew, hand_closed if base_phase in ('close','lift','transfer') else hand_open]
                    if phase_name == 'transfer':
                        arm.set_hand(hand_closed)
                        arm.set_arm(prev_recovery[:6])
                        start_point = arm.tip_mid().copy()
                        arm.set_arm(target[:6])
                        end_point = arm.tip_mid().copy()
                        seed_new = prev_recovery[:6].copy()
                        segment = []
                        for point in smooth(start_point, end_point, nframes):
                            seed_new, errnew, _ = arm.solve(point[:2], point[2], want_yaw,
                                                           seed_new, q_ref, hand=hand_closed)
                            segment.append(np.r_[seed_new, hand_closed])
                    elif phase_name == 'home_settle':
                        segment = [home.copy() for _ in range(nframes)]
                    else:
                        segment = list(smooth(prev_recovery, target, nframes))
                    replanned.extend(segment)
                    prev_recovery = segment[-1].copy()
                assert len(replanned) == len(actions)-recovery_start
                actions[recovery_start:] = replanned
                value = actions[step_i]
                episode_meta['recovery_replanned_from_physics'] = True
            phase = next(name for name, a, b in seg_bounds if a <= step_i < b)
            # Exogenous pushes use forces, not qpos edits. They are restricted
            # to the diagnostic prefix and are zeroed before recovery begins.
            mdata.xfrc_applied[:] = 0
            if args.empty_grasp_prefix:
                failure_mode = args.failure_mode or 'empty_grasp'
                phase_start = next(a for name,a,b in seg_bounds if name == phase)
                force_phase = 'miss_close' if failure_mode=='target_moves' else 'miss_drift'
                if phase == force_phase and failure_dr['force_start'] <= step_i-phase_start < failure_dr['force_start']+failure_dr['force_frames']:
                    if failure_mode in ('taken_away','strange_position','target_moves'):
                        mdata.xfrc_applied[env.target_body,:3] = [*failure_dr['force_xy'],0.]
                if failure_mode == 'execution_error' and phase == 'miss_descend':
                    value = value.copy()
                    value[failure_dr['error_joint']] += np.radians(failure_dr['error_deg']) * np.sin(np.pi*(step_i-phase_start)/max(1,phase_frames(round(22*failure_dr['time_scale']),args.speed)))
            if step_i == transfer_start:
                va, vn = int(model.flex_vertadr[fid]), int(model.flex_vertnum[fid])
                arm.set_hand(hand_closed)
                arm.set_arm(previous_command[:6])
                center_correction = arm.tip_mid()[:2] - mdata.flexvert_xpos[va:va+vn].mean(0)[:2]
                if np.linalg.norm(center_correction) > 0.06:
                    center_correction[:] = 0  # A lost object is not a centering error.
            if step_i >= transfer_start and phase not in ("return_home", "home_settle"):
                fraction = min(1., (step_i-transfer_start+1) / (transfer_end-transfer_start))
                blend = 0.5 - 0.5*np.cos(np.pi*fraction)
                arm.set_hand(hand_closed)
                arm.set_arm(value[:6])
                point = arm.tip_mid().copy()
                q, correction_error, _ = arm.solve(point[:2] + center_correction*blend,
                                                   point[2], want_yaw, value[:6], q_ref,
                                                   hand=hand_closed)
                value = np.r_[q, value[6:]]
                worst = max(worst, correction_error)
            phase = next(name for name, a, b in seg_bounds if a <= step_i < b)
            if phase == "open":
                # Hold the last carry command exactly; only fingers move to release.
                value[:6] = previous_command[:6]
                release_arm_commands.append(value[:6].copy())
            if args.max_wrist_deg is not None:
                radius = np.radians(args.max_wrist_deg)
                value[3:6] = np.clip(value[3:6], episode_arm_start[3:6]-radius,
                                    episode_arm_start[3:6]+radius)
            # Bound the actual sent command too: feedback IK at phase boundaries
            # can add a jump even when the nominal interpolated path is smooth.
            delta = value[:6] - previous_command[:6]
            if np.max(np.abs(delta)) > command_step_limit:
                value[:6] = previous_command[:6] + delta * (command_step_limit / np.max(np.abs(delta)))
                rate_limited_frames += 1
            command_step = float(np.abs(value[:6] - previous_command[:6]).max())
            command_travel += np.abs(value[:6] - previous_command[:6])
            phase_command_steps[phase] = max(phase_command_steps.get(phase, 0.), command_step)
            arm_command_step = max(arm_command_step, command_step)
            previous_command = np.asarray(value).copy()
            wrist_command_excursion = np.maximum(wrist_command_excursion,
                                                  np.abs(value[3:6]-episode_arm_start[3:6]))
            obs, _, _, _, info = env.step(value)
            hit_obstacle = False
            for contact in mdata.contact:
                g1,g2=int(contact.geom1),int(contact.geom2)
                if (g1 in robot_gids) != (g2 in robot_gids):
                    if int(contact.flex[0]) != fid and int(contact.flex[1]) != fid:
                        hit_obstacle = True
            obstacle_contact_frames += int(hit_obstacle)
            current_actual = mdata.qpos[env.joint_adr][:6].copy()
            actual_joint_step = np.maximum(actual_joint_step, np.abs(current_actual-previous_actual))
            wrist_actual_excursion = np.maximum(wrist_actual_excursion,
                                                 np.abs(current_actual[3:6]-episode_arm_start[3:6]))
            current_palm_rotation = mdata.xmat[palm_id].reshape(3, 3).copy()
            if phase in ("approach", "descend", "close"):
                pregrasp_palm_path += rotation_distance(previous_palm_rotation, current_palm_rotation)
                pregrasp_palm_max = max(pregrasp_palm_max, rotation_distance(initial_palm_rotation, current_palm_rotation))
            previous_palm_rotation = current_palm_rotation
            actual_travel += np.abs(current_actual-previous_actual)
            previous_actual = current_actual
            if phase == "home_settle":
                home_errors.append(float(np.abs(mdata.qpos[env.joint_adr] - home).max()))
            if phase in ("return_home", "home_settle"):
                for contact in mdata.contact:
                    g1, g2 = int(contact.geom1), int(contact.geom2)
                    if (g1 in robot_gids) != (g2 in robot_gids):
                        return_contacts += 1
            arm_track = max(arm_track, float(np.abs(info["tracking_error"][:6]).max()))
            if args.save and (step_i % args.save_every == 0 or step_i == len(actions) - 1):
                frames.append((step_i,
                               env.render("central").copy(),
                               env.render("left_wrist").copy()))
            if writer is not None:
                ep_frames.append({_c: np.asarray(previous_obs[_c]).copy()
                                  for _c in ("observation.images.top",
                                             "observation.images.left") if _c in obs})
                ep_states.append(np.asarray(previous_obs["observation.state"]).copy())
                ep_actions.append(np.asarray(value).copy())
                previous_obs = obs
            if 30 <= step_i <= 110 and step_i % 6 == 0:
                _a = int(model.flex_vertadr[fid])
                _n = int(model.flex_vertnum[fid])
                _pc = mdata.flexvert_xpos[_a:_a + _n].mean(0)
                _tb = [model.body(x).id for x in
                       ("hand_L_thumb_tip", "hand_L_index_tip", "hand_L_middle_tip",
                        "hand_L_ring_tip", "hand_L_pinky_tip")]
                _tm = 0.5 * (mdata.xpos[_tb[0]]
                             + np.mean([mdata.xpos[i] for i in _tb[1:]], axis=0))
                _seg = next((w for w, a0, a1 in seg_bounds if a0 <= step_i < a1), "?")
                pos_rows.append((step_i, _seg,
                             (_pc[0] + ts[0] / 2) * 100.0, (_pc[1] + ts[1] / 2) * 100.0,
                             (_tm[0] + ts[0] / 2) * 100.0, (_tm[1] + ts[1] / 2) * 100.0,
                             float(np.hypot(_pc[0] - _tm[0], _pc[1] - _tm[1])) * 100.0))
            err = np.abs(np.asarray(info["tracking_error"]))
            _seg = next((w for w, a0, a1 in seg_bounds if a0 <= step_i < a1), "?")
            # mj_geomDistance returns a bogus 0 for several of this hand's geoms even
            # 0.3 m clear of the table, so use a bounding-sphere lower bound instead:
            # geom centre z minus its bound radius, relative to the table surface.
            _dmin = min(float(mdata.geom_xpos[_g][2] - model.geom_rbound[_g])
                        - config["table"]["surface_z"] for _g in hand_gids)
            skin[_seg] = min(skin.get(_seg, 9.9), float(_dmin))
            _n = 0
            thumb_points = []
            finger_points = []
            for _k in range(mdata.ncon):
                _ct = mdata.contact[_k]
                _f = -1
                if int(_ct.flex[0]) == fid:
                    _f = 0
                elif int(_ct.flex[1]) == fid:
                    _f = 1
                if _f < 0:
                    continue
                _o = int(_ct.geom1) if _f == 1 else int(_ct.geom2)
                if _o >= 0 and (model.body(int(model.geom_bodyid[_o])).name
                                or "").startswith("hand_"):
                    _n += 1
                    contact_name = model.body(int(model.geom_bodyid[_o])).name or ''
                    (thumb_points if 'thumb' in contact_name else finger_points).append(_ct.pos.copy())
            if _seg in ('close','lift'):
                va,vn=int(model.flex_vertadr[fid]),int(model.flex_vertnum[fid])
                center=mdata.flexvert_xpos[va:va+vn].mean(0)
                row=dict(frame=step_i,phase=_seg,center=center.tolist(),
                         thumb_count=len(thumb_points),finger_count=len(finger_points),
                         thumb_pad_offset=(mdata.geom_xpos[arm.tip_geoms[0]]-center).tolist())
                for label,points in [('thumb',thumb_points),('fingers',finger_points)]:
                    row[label+'_offset']=None if not points else (np.mean(points,axis=0)-center).tolist()
                thumb_contact_trace.append(row)
            if _seg.startswith('miss_'):
                miss_contact_frames += int(_n > 0)
                va,vn=int(model.flex_vertadr[fid]),int(model.flex_vertnum[fid])
                failure_trace.append(dict(frame=step_i,phase=_seg,contacts=_n,
                    center=mdata.flexvert_xpos[va:va+vn].mean(0).tolist(),
                    thumb_contacts=len(thumb_points),finger_contacts=len(finger_points),
                    contact_centroid=(np.mean(thumb_points+finger_points,axis=0).tolist()
                                      if thumb_points or finger_points else None)))
            _f = 0.0
            for _k in range(mdata.ncon):
                _ct = mdata.contact[_k]
                if int(_ct.flex[0]) == fid or int(_ct.flex[1]) == fid:
                    _cf = np.zeros(6)
                    mujoco.mj_contactForce(model, mdata, _k, _cf)
                    _f += float(np.linalg.norm(_cf[:3]))
            fsum[_seg] = max(fsum.get(_seg, 0.0), _f)
            if _seg == "transfer":
                carry_contacts.append(_n)
            if _seg in ("retreat", "return_home", "home_settle"):
                final_success.append(bool(info["success"]))
            end_contacts[_seg] = _n
            segment_end[_seg] = mdata.flexvert_xpos[int(model.flex_vertadr[fid]):int(model.flex_vertadr[fid])+int(model.flex_vertnum[fid])].mean(0).tolist()
            if _n:
                hp[_seg] = max(hp.get(_seg, 0), _n)
            _tb = [model.body(x).id for x in
                   ("hand_L_thumb_tip", "hand_L_index_tip", "hand_L_middle_tip",
                    "hand_L_ring_tip", "hand_L_pinky_tip")]
            _tm = 0.5 * (mdata.xpos[_tb[0]]
                         + np.mean([mdata.xpos[i] for i in _tb[1:]], axis=0))
            tipz.append(float(_tm[2]) - config["table"]["surface_z"])
            _a = int(model.flex_vertadr[fid])
            _nn = int(model.flex_vertnum[fid])
            surf_low.append(float(mdata.flexvert_xpos[_a:_a + _nn][:, 2].min())
                            - config["table"]["surface_z"])
            if first_div is None and float(err[:6].max()) > 0.5:
                ach = np.asarray(value) + np.asarray(info["tracking_error"])
                seg = next((w for w, a0, a1 in seg_bounds if a0 <= step_i < a1), "?")
                first_div = (step_i, seg, np.asarray(value)[:6].copy(), ach[:6].copy())
            if float(err.max()) > track:
                track = float(err.max())
                worst_vec = np.asarray(info["tracking_error"]).copy()
                worst_step = step_i
                worst_clip = bool(info.get("action_clipped", False))
                worst_cmd = np.asarray(value).copy()
                pairs = {}
                for k in range(mdata.ncon):
                    ct = mdata.contact[k]
                    if int(ct.flex[0]) >= 0 or int(ct.flex[1]) >= 0:
                        f0 = int(ct.flex[0]); f1 = int(ct.flex[1])
                        key = "FLEX %s <-> FLEX %s" % (
                            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, f0)
                            if f0 >= 0 else "-",
                            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, f1)
                            if f1 >= 0 else "-")
                    elif int(ct.geom1) < 0 or int(ct.geom2) < 0:
                        key = "UNKNOWN geom1=%d geom2=%d" % (int(ct.geom1), int(ct.geom2))
                    else:
                        key = "%s <-> %s" % (model.geom(int(ct.geom1)).name,
                                             model.geom(int(ct.geom2)).name)
                    pairs[key] = pairs.get(key, 0) + 1
                worst_pairs = sorted(pairs.items(), key=lambda kv: -kv[1])[:8]
            adr = int(model.flex_vertadr[fid])
            num = int(model.flex_vertnum[fid])
            peak = max(peak, float(mdata.flexvert_xpos[adr:adr + num].mean(0)[2])
                       - start_z)
        if args.save:
            import cv2
            for step_i, top, wrist in frames:
                cv2.imwrite(str(OUT / ("ep%02d_top_%03d.png" % (e, step_i))),
                            cv2.cvtColor(top, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(OUT / ("ep%02d_wrist_%03d.png" % (e, step_i))),
                            cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
        # ---- HONEST end state: is the pack RELEASED and inside, or still in the hand?
        _va = int(model.flex_vertadr[fid]); _vn = int(model.flex_vertnum[fid])
        _pts = mdata.flexvert_xpos[_va:_va + _vn]
        _c = _pts.mean(0)
        _tray = mdata.xpos[env.tray_body]
        _yaw = float(config["tray"].get("yaw", 0.0))
        _cy, _sy = np.cos(-_yaw), np.sin(-_yaw)
        _dx, _dy = _c[0] - _tray[0], _c[1] - _tray[1]
        _lx = _cy * _dx - _sy * _dy
        _ly = _sy * _dx + _cy * _dy
        _wall = float(config["tray"]["wall_thickness"])
        _hin = (config["tray"]["size"][0] / 2 - _wall,
                config["tray"]["size"][1] / 2 - _wall)
        _bottom = float(_pts[:, 2].min())
        print("      final bounds:", _pts.min(0), _pts.max(0))
        _floor = float(_tray[2]) + 0.010
        _rim = float(_tray[2]) + float(config["tray"]["size"][2])
        _held = 0
        for _k in range(mdata.ncon):
            _ct = mdata.contact[_k]
            if int(_ct.flex[0]) == fid or int(_ct.flex[1]) == fid:
                _f = 1 if int(_ct.flex[1]) == fid else 0
                _o = int(_ct.geom2) if _f == 0 else int(_ct.geom1)
                if _o >= 0 and (model.body(int(model.geom_bodyid[_o])).name
                                or "").startswith("hand_"):
                    _held += 1
        _lxi = _cy * (_pts[:, 0] - _tray[0]) - _sy * (_pts[:, 1] - _tray[1])
        _lyi = _sy * (_pts[:, 0] - _tray[0]) + _cy * (_pts[:, 1] - _tray[1])
        # "held to the box": the pack was still in the hand at the end of the `transfer`
        # segment, so a pack that slipped out early (while the hand was still closed)
        # is NOT counted -- the user explicitly rejected that.
        print("      end contacts:", end_contacts)
        print("      segment pack centers:", segment_end)
        _held_to_box = end_contacts.get("transfer", 0) > 0
        _really = bool(env.success() and _held_to_box and peak > 0.03
                       and carry_contacts and min(carry_contacts) > 0
                       and len(final_success) >= 15 and all(final_success[-15:]))
        motion_ok = arm_track <= 0.5 and arm_command_step <= 0.15 and obstacle_contact_frames == 0
        home_error = max(home_errors[-5:])
        home_ok = home_error <= 0.03 and return_contacts == 0
        ok = _really and motion_ok and home_ok
        wrist_actual_deg = np.degrees(wrist_actual_excursion).tolist()
        wrist_command_deg = np.degrees(wrist_command_excursion).tolist()
        wrist_ok = (args.max_wrist_deg is None or
                    max(wrist_actual_deg + wrist_command_deg) <= args.max_wrist_deg + 1e-7)
        if args.max_wrist_deg is not None:
            wrist_ok = bool(wrist_ok and np.max(actual_joint_step) <= np.radians(3.) + 1e-7)
        ok = bool(ok and wrist_ok)
        if args.empty_grasp_prefix:
            miss_centers = [np.asarray(v) for k, v in segment_end.items() if k.startswith('miss_')]
            miss_displacement = max(float(np.linalg.norm(v[:2]-np.array([px,py]))) for v in miss_centers)
            miss_ok = miss_contact_frames == 0 and miss_displacement < .015
            mode = args.failure_mode or 'empty_grasp'
            if mode not in ('empty_grasp','closed_empty','execution_error'):
                lift_rows = [r for r in failure_trace if r['phase']=='miss_lift']
                drift_rows = [r for r in failure_trace if r['phase']=='miss_drift']
                lifted = any(r['contacts']>0 and r['center'][2]>start_z+.04 for r in lift_rows)
                released = bool(drift_rows and drift_rows[-1]['contacts']==0)
                fallen = bool(drift_rows and max(r['center'][2] for r in lift_rows)-drift_rows[-1]['center'][2]>.035)
                if mode in ('slip','taken_away'):
                    miss_ok = lifted and released and fallen
                elif mode in ('knock_away','corner_grasp'):
                    corner_evidence = any(r['thumb_contacts']>0 and r['finger_contacts']>0
                        and abs(np.dot(np.asarray(r['contact_centroid'])[:2]-np.asarray(r['center'])[:2],
                                       [np.cos(pyaw),np.sin(pyaw)])) > tgt['size'][0]*.25
                        for r in failure_trace if r['phase'] in ('miss_close','miss_lift'))
                    episode_meta['corner_contact_evidence'] = corner_evidence
                    miss_ok = (corner_evidence and released if mode=='corner_grasp' else
                               miss_contact_frames>0 and miss_displacement>.025 and released and not lifted)
                else:
                    miss_ok = miss_displacement>.025 and released
            episode_meta['failure_trace'] = failure_trace
            episode_meta.update(miss_contact_frames=miss_contact_frames,
                                miss_object_displacement_m=miss_displacement, miss_ok=miss_ok)
            ok = bool(ok and miss_ok)
        episode_meta.update(wrist_actual_max_from_start_deg=wrist_actual_deg,
                            wrist_command_max_from_start_deg=wrist_command_deg, wrist_ok=bool(wrist_ok),
                            actual_joint_max_step_deg=np.degrees(actual_joint_step).tolist(),
                            command_joint_max_step_deg=float(np.degrees(arm_command_step)))
        if validated_plan is not None:
            ok = bool(ok and np.isfinite(pregrasp_palm_max)
                      and np.degrees(pregrasp_palm_max) <= validated_plan['max_pregrasp_palm_deg'])
            episode_meta['pregrasp_palm_max_from_start_deg'] = float(np.degrees(pregrasp_palm_max))
            episode_meta['pregrasp_palm_path_deg'] = float(np.degrees(pregrasp_palm_path))
        local_points=(_pts-mdata.xpos[env.tray_body]) @ mdata.xmat[env.tray_body].reshape(3,3)
        contact_radius=float(model.flex_radius[fid])
        half_inner=np.asarray(tray['size'][:2])/2-tray['wall_thickness']
        placement_margins=dict(
            side_clearance_m=float(np.min(half_inner-np.abs(local_points[:,:2])-contact_radius)),
            rim_clearance_m=float(tray['size'][2]+.0005-np.max(local_points[:,2]+contact_radius)),
            floor_clearance_m=float(np.min(local_points[:,2]-contact_radius)-(.010-.002)),
        )
        print('      placement margins: '+json.dumps(placement_margins),flush=True)
        print("      home error %.6f rad, return contacts %d, home ok %s" % (home_error, return_contacts, home_ok))
        print("      arm max tracking %.4f rad, max command step %.4f rad, motion ok %s"
              % (arm_track, arm_command_step, motion_ok))
        print("      END: pack local (%+.1f,%+.1f) cm  inner half (%.1f,%.1f)  "
              "bottom %.4f floor %.4f rim %.4f  hand contacts %d"
              % (_lx * 100, _ly * 100, _hin[0] * 100, _hin[1] * 100,
                 _bottom, _floor, _rim, _held))
        _held_names = []
        for _k in range(mdata.ncon):
            _ct = mdata.contact[_k]
            if int(_ct.flex[0]) == fid or int(_ct.flex[1]) == fid:
                _f = 1 if int(_ct.flex[1]) == fid else 0
                _o = int(_ct.geom2) if _f == 0 else int(_ct.geom1)
                if _o >= 0:
                    _bn = model.body(int(model.geom_bodyid[_o])).name or ""
                    if _bn.startswith("hand_"):
                        _held_names.append(_bn)
        from collections import Counter
        _hn = dict(Counter(_held_names))
        print("      hand parts still touching the pack: %s"
              % ("  ".join("%s=%d" % kv for kv in sorted(_hn.items())) if _hn else "NONE"))
        print("      REALLY IN BOX: %-5s   controller success(): %s"
              % (_really, ok))
        if writer is not None and ok:
            if preview_writer is not None:
                preview_writer.add_episode(ep_frames, ep_states, ep_actions, episode_meta)
            writer.add_episode(ep_frames[recovery_start:], ep_states[recovery_start:], ep_actions[recovery_start:],
                               {"pack_cm": [px_cm, py_cm], "speed_multiplier": args.speed,
                                "return_home": True, "home_state": home.tolist(), "home_error_rad": home_error, **episode_meta})
        print("      bounding-sphere lower bound to table (not skin clearance): "
              + "  ".join("%s %.4f" % kv for kv in skin.items()))
        print("      overall bounding-sphere lower bound: %.4f m (pack is 0.065 m tall)"
              % min(skin.values()))
        print("      peak contact force on the pack per segment (N): "
              + "  ".join("%s %.2f" % kv for kv in fsum.items()))
        print("      hand<->target contacts per segment: %s"
              % ("  ".join("%s %d" % kv for kv in hp.items()) if hp else "NONE EVER"))
        print("      fingertip-mid height above table: min %.3f max %.3f | pack lowest "
              "vertex %.3f m" % (min(tipz), max(tipz), min(surf_low)))
        if pos_rows:
            print("      step segment     pack(x,y) cm      tip-mid(x,y) cm    gap cm")
            for r in pos_rows:
                print("      %4d %-10s (%6.1f,%6.1f)   (%6.1f,%6.1f)   %6.1f"
                      % (r[0], r[1], r[2], r[3], r[4], r[5], r[6]))
        if first_div is not None:
            print("      FIRST divergence at step %d (segment %s)" % (first_div[0],
                                                                     first_div[1]))
            print("         cmd : %s" % np.round(first_div[2], 3))
            print("         ach : %s" % np.round(first_div[3], 3))
        else:
            print("      no divergence above 0.5 rad on any arm joint")
        if e < 3:
            print("      worst step %d: action_clipped=%s  contacts=%d"
                  % (worst_step, worst_clip, sum(n for _, n in worst_pairs)))
            for key, n in worst_pairs:
                print("         %3d x %s" % (n, key))
            if worst_cmd is not None:
                print("         arm cmd  : %s" % np.round(worst_cmd[:6], 3))
                print("         arm aimed: %s"
                      % np.round(worst_cmd[:6] + worst_vec[:6], 3))
        if e < 3 and worst_vec is not None:
            print("      worst tracking error per dim:")
            for i in range(16):
                mark = "  <== ARM" if i < 6 else ""
                print("         %-24s %+8.4f%s"
                      % (layout.DATASET_NAMES[i], worst_vec[i], mark))
        print("  %-4d (%5.1f,%5.1f) %5.1f  %8.4f %8.3f %8.3f %9s %s"
              % (e, px_cm, py_cm, np.degrees(pyaw), worst, peak, track, ok,
                 len(actions)), flush=True)
        rows.append({"episode": e, "pack_cm": [round(px_cm, 2), round(py_cm, 2)],
                     "yaw_deg": round(float(np.degrees(pyaw)), 2),
                     "ik_err": round(worst, 4), "peak_lift": round(peak, 4),
                     "tracking": round(track, 4), "success": ok,
                     "arm_tracking_rad": round(arm_track, 6),
                     "arm_command_step_rad": round(arm_command_step, 6),
                     "command_joint_travel_rad": command_travel.tolist(),
                     "actual_joint_travel_rad": actual_travel.tolist(),
                     "actual_joint_travel_total_rad": float(actual_travel.sum()),
                     "pregrasp_palm_path_deg": float(np.degrees(pregrasp_palm_path)),
                     "pregrasp_palm_max_from_start_deg": float(np.degrees(pregrasp_palm_max)),
                     "motion_ok": motion_ok,
                     "placement_margins": placement_margins,
                     "obstacle_contact_frames": obstacle_contact_frames,
                     "rate_limited_frames": rate_limited_frames, "phase_command_steps": phase_command_steps,
                     "home_ok": home_ok, "home_error_rad": home_error, "return_contacts": return_contacts,
                     "phases": [name for name, _, _ in seg_bounds],
                     "release_arm_command_span_rad": float(np.ptp(np.asarray(release_arm_commands), axis=0).max()),
                     "end_contacts": end_contacts, "segment_centers": segment_end,
                     "continuous_carry": bool(carry_contacts and min(carry_contacts) > 0),
                     "settled_in_tray": bool(len(final_success) >= 15 and all(final_success[-15:])),
                     "frames": len(actions), "speed_multiplier": args.speed, **episode_meta})
        (OUT / "result.json").write_text(json.dumps(rows, indent=2))
        (OUT / f'thumb_contacts_{e:03d}.json').write_text(json.dumps(thumb_contact_trace,indent=2))
        if round_manager is not None:
            round_manager.finish_pick(ok)
        if args.successes is not None and sum(r["success"] for r in rows) >= args.successes:
            break
    env.close()
    if writer is not None and writer.episodes:
        print("  wrote LeRobot dataset: %s" % writer.write())
    elif writer is not None:
        print("  no successful episodes; no dataset written")
    good = sum(r["success"] for r in rows)
    print()
    print("  SCRIPTED CONTROLLER: success %d/%d   mean ik err %.4f m   max tracking %.4f rad"
          % (good, len(rows), float(np.mean([r["ik_err"] for r in rows])),
             float(np.max([r["tracking"] for r in rows]))))
    (OUT / "result.json").write_text(json.dumps(rows, indent=2))
    return 0 if args.successes is None or good >= args.successes else 2


if __name__ == "__main__":
    raise SystemExit(main())
