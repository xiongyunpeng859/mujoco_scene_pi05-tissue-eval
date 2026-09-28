#!/usr/bin/env python3
"""Add --collect DIR: record successful episodes into a LeRobot v2.1 dataset."""
from pathlib import Path

SRC = Path("/workspace/shared/mujoco_tissue_scene/reports/tools/scripted_pick_place.py")
t = SRC.read_text()
done = []


def sub(tag, old, new):
    global t
    n = t.count(old)
    if n == 1:
        t = t.replace(old, new)
        done.append(tag)
    else:
        done.append("%s SKIP(%d)" % (tag, n))


sub("import",
    "from scripted_pick_place import" if False else "import sim_env",
    "import sim_env\nsys.path.insert(0, str(ROOT / \"reports/tools\"))\nfrom lerobot_writer import LerobotWriter")

sub("arg",
    '    parser.add_argument("--descend-dz", type=float, default=-0.20)',
    '    parser.add_argument("--descend-dz", type=float, default=-0.20)\n'
    '    parser.add_argument("--collect", type=str, default=None)')

sub("writer-init",
    "    boxes = config[\"boxes\"]\n    bx_world, by_world = tray[\"center\"][0], tray[\"center\"][1]",
    "    boxes = config[\"boxes\"]\n    bx_world, by_world = tray[\"center\"][0], tray[\"center\"][1]\n"
    "    writer = LerobotWriter(args.collect) if args.collect else None")

sub("ep-buffers",
    '        fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, tgt["name"] + "_soft")',
    '        fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, tgt["name"] + "_soft")\n'
    '        ep_frames, ep_states, ep_actions = [], [], []')

sub("capture",
    """            if step_i % 24 == 0 or step_i == len(actions) - 1:
                frames.append((step_i, np.asarray(obs["observation.images.top"]).copy()))""",
    """            if step_i % 24 == 0 or step_i == len(actions) - 1:
                frames.append((step_i, np.asarray(obs["observation.images.top"]).copy()))
            if writer is not None:
                ep_frames.append({_c: np.asarray(obs[_c]).copy()
                                  for _c in ("observation.images.top",
                                             "observation.images.left") if _c in obs})
                ep_states.append(np.asarray(mdata.qpos[env.joint_adr]).copy())
                ep_actions.append(np.asarray(value).copy())""")

sub("add-episode",
    "        ok = bool(env.success() and peak > 0.03)",
    "        ok = bool(env.success() and peak > 0.03)\n"
    "        if writer is not None and ok:\n"
    "            writer.add_episode(ep_frames, ep_states, ep_actions,\n"
    "                               {\"pack_cm\": [px_cm, py_cm]})")

sub("write-out",
    "    env.close()",
    "    env.close()\n"
    "    if writer is not None:\n"
    "        print(\"  wrote LeRobot dataset: %s\" % writer.write())")

SRC.write_text(t)
import ast
ast.parse(t)
print("patch:", done)
