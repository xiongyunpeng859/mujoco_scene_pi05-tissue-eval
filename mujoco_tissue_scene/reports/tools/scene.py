"""Photo-inspired MuJoCo scene. Defaults to CPU-only, headless rendering."""
from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

# Set these before importing MuJoCo/PyOpenGL. EGL is opt-in via --backend.
import sys
if "--backend" in sys.argv:
    os.environ["MUJOCO_GL"] = sys.argv[sys.argv.index("--backend") + 1]
else:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
if os.environ["MUJOCO_GL"] in {"osmesa", "egl"}:
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("LP_NUM_THREADS", "2")

import mujoco
import numpy as np
from PIL import Image
import yaml
import hand_control

ROOT = Path(__file__).resolve().parent


def vector(values):
    return " ".join(str(float(value)) for value in values)


def element(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {key: str(value) for key, value in attrs.items()})


def geom(parent, name, position, halfsize, color, **attrs):
    return element(parent, "geom", name=name, type="box", pos=vector(position),
                   size=vector(halfsize), rgba=vector(color), **attrs)


def camera_axes(position, target):
    backward = np.asarray(position) - np.asarray(target)
    backward = backward / np.linalg.norm(backward)
    right = np.cross([0, 0, 1], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    return vector(np.concatenate([right, up]))


def camera_intrinsics(config):
    """MuJoCo principalpixel uses signed image-center offsets, not OpenCV cx/cy."""
    if "intrinsics" not in config:
        return {"fovy": config["fovy"]}
    calibration = config["intrinsics"]
    width, height = calibration["resolution"]
    return {"resolution": f"{width} {height}", "sensorsize": "1 1",
            "focalpixel": vector([calibration["fx"], calibration["fy"]]),
            "principalpixel": vector([width/2-calibration["cx"], height/2-calibration["cy"]])}


def table_corner_cm_to_world_m(point_cm, table_size_cm):
    """Convert real tabletop left-bottom cm coordinates to centered world XY."""
    x_cm, y_cm = point_cm
    width_cm, depth_cm = table_size_cm
    return [x_cm / 100.0 - width_cm / 200.0,
            y_cm / 100.0 - depth_cm / 200.0]


def apply_measured_layout(config):
    """Apply optional measured tabletop layout while preserving Z/yaw estimates."""
    layout = config.get("measured_layout", {})
    if not layout.get("enabled", False):
        return config
    table_size_cm = layout.get("table_size_cm")
    if not table_size_cm:
        table_size_cm = [config["table"]["size"][0] * 100.0,
                         config["table"]["size"][1] * 100.0]
    # Keep the MuJoCo table size consistent with the measured tabletop.
    config["table"]["size"][:2] = [table_size_cm[0] / 100.0, table_size_cm[1] / 100.0]
    if "tray_size_cm" in layout:
        config["tray"]["size"][:2] = [layout["tray_size_cm"][0] / 100.0,
                                      layout["tray_size_cm"][1] / 100.0]
    if "tray_lower_left_xy_cm_from_left_bottom" in layout:
        lower_left = layout["tray_lower_left_xy_cm_from_left_bottom"]
        tray_size_cm = layout.get(
            "tray_size_cm",
            [config["tray"]["size"][0] * 100.0, config["tray"]["size"][1] * 100.0],
        )
        center_cm = [lower_left[0] + tray_size_cm[0] / 2.0,
                     lower_left[1] + tray_size_cm[1] / 2.0]
        config["tray"]["center"] = table_corner_cm_to_world_m(center_cm, table_size_cm)
    elif "tray_center_xy_cm_from_left_bottom" in layout:
        config["tray"]["center"] = table_corner_cm_to_world_m(
            layout["tray_center_xy_cm_from_left_bottom"], table_size_cm
        )
    if "arm_mount_xy_cm_from_left_bottom" in layout:
        x, y = table_corner_cm_to_world_m(
            layout["arm_mount_xy_cm_from_left_bottom"], table_size_cm
        )
        config["arm"]["position"][:2] = [x, y]
    return config


def tray_left_edge_cm(config):
    """Leftmost x of the tray footprint in table centimetres, honouring its yaw."""
    tray = config["tray"]
    centre_cm = [(tray["center"][0] + config["table"]["size"][0] / 2.0) * 100.0,
                 (tray["center"][1] + config["table"]["size"][1] / 2.0) * 100.0]
    half_x = tray["size"][0] * 100.0 / 2.0
    half_y = tray["size"][1] * 100.0 / 2.0
    yaw = float(tray.get("yaw", 0.0))
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return min((rotation @ np.array([sx * half_x, sy * half_y]) + np.array(centre_cm))[0]
               for sx in (-1, 1) for sy in (-1, 1))


def sample_box_positions(config, rng):
    """Draw one (x_cm, y_cm, yaw) per box from the measured spawn region.

    Shared by the scene builder and by the environment's reset, so a randomised
    episode and a rebuilt XML can never disagree about where bags may appear.
    """
    spec = config.get("box_randomization", {})
    region = spec.get("region_xy_cm_from_left_bottom") or {}
    x_range, y_range = region.get("x"), region.get("y")
    if not x_range or not y_range:
        raise ValueError("box_randomization needs region_xy_cm_from_left_bottom.x and .y")
    table_size_cm = [config["table"]["size"][0] * 100.0,
                     config["table"]["size"][1] * 100.0]
    tray_left = tray_left_edge_cm(config)
    yaw_jitter = float(spec.get("yaw_jitter_deg", 0.0))
    separation = float(spec.get("min_separation_cm", 0.0))
    placed, draws = [], []
    for box in config["boxes"]:
        reach = max(box["size"][0], box["size"][1]) * 100.0 / 2.0
        for _ in range(500):
            x_cm = rng.uniform(*x_range)
            y_cm = rng.uniform(*y_range)
            if x_cm + reach > tray_left - 1.0:
                continue
            if x_cm - reach < 1.0 or y_cm + reach > table_size_cm[1] - 1.0:
                continue
            if any(np.hypot(x_cm - px, y_cm - py) < separation for px, py in placed):
                continue
            placed.append((x_cm, y_cm))
            break
        else:
            raise ValueError(f"could not place {box['name']} inside the spawn region")
        yaw = float(box.get("yaw", 0.0)) + float(
            np.radians(rng.uniform(-yaw_jitter, yaw_jitter)))
        draws.append((x_cm, y_cm, yaw))
    return draws, table_size_cm


def apply_box_randomization(config, seed=None):
    """Sample the blue tissue bags inside the measured spawn rectangle.

    The bags are re-placed by hand every real episode, always to the left of the
    green box, so the simulation samples them from the same region instead of
    pinning three fixed coordinates.
    """
    spec = config.get("box_randomization", {})
    if not spec.get("enabled", False):
        return config
    chosen_seed = spec.get("seed") if seed is None else seed
    if chosen_seed is None:
        chosen_seed = int(np.random.SeedSequence().entropy % (2 ** 31))
    draws, table_size_cm = sample_box_positions(config, np.random.default_rng(chosen_seed))
    for box, (x_cm, y_cm, yaw) in zip(config["boxes"], draws):
        box["position"] = [x_cm / 100.0 - table_size_cm[0] / 200.0,
                           y_cm / 100.0 - table_size_cm[1] / 200.0]
        box["yaw"] = yaw
    spec["last_seed"] = chosen_seed
    return config


def target_on_table_xy(config):
    """Place the pre-grasp target beside the tray, not inside it."""
    tray = config["tray"]
    if "target_on_table_position" in tray:
        return tray["target_on_table_position"]
    tx, ty = tray["center"]
    tw, td = tray["size"][:2]
    return [tx - 0.75 * tw, ty + 0.10 * td]


def import_urdf(root, config, prefix="", arm_only=False):
    """Convert vendor models in a temporary directory, without editing sources."""
    urdf_path = Path(config["urdf"])
    mesh_dir = Path(config["mesh_dir"])
    robot = ET.parse(urdf_path).getroot()
    if arm_only:
        keep_links = {"base_link", *(f"link{i}" for i in range(1, 7))}
        for child in list(robot):
            if child.tag == "link" and child.get("name") not in keep_links:
                robot.remove(child)
            elif child.tag == "joint" and child.get("name") not in {f"joint{i}" for i in range(1,7)}:
                robot.remove(child)
    for mesh in robot.iter("mesh"):
        path = mesh_dir / Path(mesh.attrib["filename"]).name
        if not path.is_file():
            raise FileNotFoundError(path)
        # Some vendor binary STL meshes exceed MuJoCo's 200k-face STL limit.
        # Convert to OBJ in our output directory, never mutate vendor assets.
        if arm_only:
            header = path.read_bytes()[:84]
            faces = struct.unpack("<I", header[80:84])[0]
            if faces > 200000:
                destination = Path(config["mesh_cache"]) / (path.stem + ".obj")
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    dtype=np.dtype([("normal","<f4",(3,)),("vertices","<f4",(3,3)),("attribute","<u2")])
                    triangles=np.fromfile(path,dtype=dtype,offset=84,count=faces)
                    vertices,indices=np.unique(triangles["vertices"].reshape(-1,3),axis=0,return_inverse=True)
                    with destination.open("w") as stream:
                        for vertex in vertices:
                            stream.write("v " + vector(vertex) + "\n")
                        for face in indices.reshape(-1,3)+1:
                            stream.write("f " + " ".join(str(int(index)) for index in face) + "\n")
                path = destination.resolve()
        mesh.set("filename", str(path))
    extension = element(robot, "mujoco")
    element(extension, "compiler", discardvisual="false", fusestatic="false",
            strippath="false", balanceinertia="true")
    with tempfile.TemporaryDirectory(prefix="tissue_hand_") as temp:
        source = Path(temp) / "hand.urdf"
        converted = Path(temp) / "hand.xml"
        ET.ElementTree(robot).write(source)
        model = mujoco.MjModel.from_xml_path(str(source))
        mujoco.mj_saveLastXML(str(converted), model)
        imported = ET.parse(converted).getroot()
    assets = imported.find("asset")
    if prefix:
        for node in imported.iter():
            if "name" in node.attrib:
                node.set("name", prefix + node.get("name"))
            for reference in ["mesh", "material"]:
                if reference in node.attrib:
                    node.set(reference, prefix + node.get(reference))
    if assets is not None:
        for asset in assets:
            if asset.tag == "mesh":
                asset_file=Path(asset.attrib["file"])
                if not asset_file.is_absolute():
                    asset.set("file", str(mesh_dir / asset_file.name))
            root.find("asset").append(asset)
    return imported.find("worldbody")


def add_static_hand(root, world, config):
    """Hand base is fixed to flange; finger joints retain SDK motion/mimic."""
    imported_world = import_urdf(root, config, prefix="hand_")
    mount = element(world, "body", name="static_omnihand_reference",
                    pos=vector(config["position"]), euler=vector(config["euler"]))
    for body in imported_world:
        mount.append(body)
    for parent in mount.iter():
        if parent.tag == "joint":
            parent.set("damping","0.01")
            parent.set("armature","0.001")
        if parent.tag == "geom":
            visual=parent.get("group")=="1"
            parent.set("contype", "0" if visual else "2")
            parent.set("conaffinity", "0" if visual else "1")
            if not visual:
                parent.set("group","3")
                parent.set("condim","4")
            # Collision primitives stay invisible. Visual SDK meshes are silver.
            if parent.get("type") == "mesh":
                parent.set("rgba", "0.83 0.84 0.82 1")
    # Mechanical coupling between finger segments is real and must stay.
    equality=element(root,"equality")
    robot=ET.parse(config["urdf"]).getroot()
    for joint in robot.findall("joint"):
        mimic=joint.find("mimic")
        if mimic is not None:
            element(equality,"joint",name="mimic_"+joint.get("name"),
                    joint1="hand_"+joint.get("name"),joint2="hand_"+mimic.get("joint"),
                    polycoef=vector([float(mimic.get("offset","0")),float(mimic.get("multiplier","1")),0,0,0]))
    # The five spread/thumb-base joints share one value in both recorded gestures,
    # so they can be frozen at that constant -- but the dataset shows each episode
    # starts them somewhere slightly different and then holds them.  Freezing them
    # would make the recorded action unrepresentable, so by default they are left
    # free and simply held by their own actuator at whatever reset chose.
    if config.get("hand", {}).get("lock_constant_joints", False):
        real_poses=hand_control.poses()
        for index,joint in enumerate(hand_control.HAND_JOINTS):
            if real_poses["open"][index]==real_poses["closed"][index]:
                element(equality,"joint",name="fixed_gesture_"+joint,joint1="hand_"+joint,
                        polycoef=vector([real_poses["open"][index],0,0,0,0]))


def add_arm_and_hand(root, world, config):
    arm = config["arm"]
    imported_world = import_urdf(root, arm, arm_only=True)
    mount = element(world, "body", name="qiuzhi_arm_mount",
                    pos=vector(arm["position"]), euler=vector(arm["euler"]))
    for body in imported_world:
        mount.append(body)
    for node in mount.iter():
        if node.tag == "body":
            node.set("gravcomp", "1")  # Preview control, not calibrated motor dynamics.
        elif node.tag == "joint":
            # Joint damping is a config value: replaying real trajectories showed
            # 0.1 N.m.s/rad tracks the recorded flange path better than 1.0.
            node.set("damping", str(arm.get("joint_damping", 0.1)))
        elif node.tag == "geom":
            visual=node.get("group")=="1"
            node.set("contype", "0" if visual else "2")
            node.set("conaffinity", "0" if visual else "1")
            if not visual:
                node.set("group","3")
            # The real arm's links are dark grey (RGB ~0.29 0.31 0.31 measured
            # from a 40-frame consensus of real first frames); only the hand is
            # silver.  Getting this wrong poisons every image-based comparison.
            elif node.get("type") == "mesh":
                node.set("rgba", vector(arm.get("link_rgba", [0.29, 0.30, 0.31, 1.0])))
    flange = next(node for node in mount.iter("body") if node.get("name") == "link6")
    adapter = element(flange, "body", name="estimated_hand_adapter",
                      pos=vector(arm["hand_mount_position"]),
                      euler=vector(arm["hand_mount_euler"]))
    element(adapter,"geom",name="hand_adapter_visual",type="cylinder",size="0.028 0.012",
            rgba="0.25 0.25 0.25 1",contype="0",conaffinity="0")
    hand = dict(config["hand"],position=[0,0,0],euler=[0,0,0])
    add_static_hand(root,adapter,hand)
    for node in adapter.iter("body"):
        node.set("gravcomp", "1")
    camera_config = config["wrist_camera"]
    camera_parent = next(node for node in mount.iter("body")
                         if node.get("name") == camera_config["parent_body"])
    anchor = camera_config["bracket_anchor"]
    geom(camera_parent,"wrist_camera_mount_plate",anchor,[.009,.019,.015],
         [.65,.67,.65,1],contype="0",conaffinity="0",mass="0")
    # The strut meets both the wrist plate and camera housing: no floating case.
    element(camera_parent,"geom",name="wrist_camera_support",type="capsule",
            fromto=vector([*anchor,*camera_config["position"]]),size="0.007",
            rgba="0.72 0.74 0.72 1",contype="0",conaffinity="0",mass="0")
    camera_mount = element(camera_parent,"body",name="wrist_camera_mount",
                           pos=vector(camera_config["position"]),
                           gravcomp="1",**({"xyaxes":vector(camera_config["xyaxes"])} if "xyaxes" in camera_config else {"euler":vector(camera_config["euler"])}))
    # Lightweight visual-only housing and two lenses; no uncalibrated sensor mass.
    # The calibrated hand-eye transform gives the OPTICAL CENTRE, so the camera
    # sits at the mount origin and the housing is pushed back along +z instead.
    housing = 0.016
    geom(camera_mount,"wrist_camera_silver_case",[0,0,housing],[.023,.020,.012],
         [.82,.83,.81,1],contype="0",conaffinity="0",mass="0")
    geom(camera_mount,"wrist_camera_black_face",[0,0,housing-.0125],[.020,.017,.001],
         [.018,.021,.024,1],contype="0",conaffinity="0",mass="0")
    for index,x in enumerate([-.012,.012]):
        element(camera_mount,"geom",name=f"wrist_camera_lens_{index}",type="cylinder",
                pos=vector([x,0,housing-.014]),size="0.005 0.001",rgba="0.08 0.15 0.20 1",
                contype="0",conaffinity="0",mass="0")
    element(camera_mount,"camera",name="left_wrist",pos="0 0 0",
            **camera_intrinsics(camera_config))
    actuators = element(root,"actuator")
    # Gains are configuration, not magic numbers: they were chosen by replaying
    # real dataset trajectories and comparing the flange path (see replay_check.py).
    arm_gains = config.get("arm", {}).get("actuator", {})
    hand_gains = config.get("hand", {}).get("actuator", {})
    arm_kp, arm_kv = float(arm_gains.get("kp", 100.0)), float(arm_gains.get("kv", 20.0))
    arm_force = float(arm_gains.get("force", 80.0))
    hand_kp, hand_kv = float(hand_gains.get("kp", 3.0)), float(hand_gains.get("kv", 0.1))
    hand_force = float(hand_gains.get("force", 1.5))
    for index in range(1,7):
        element(actuators,"position",name=f"arm_joint{index}_position",joint=f"joint{index}",
                kp=str(arm_kp),kv=str(arm_kv),forcerange=f"{-arm_force} {arm_force}")
    for joint in hand_control.HAND_JOINTS:
        element(actuators,"position",name="hand_"+joint+"_position",joint="hand_"+joint,
                kp=str(hand_kp),kv=str(hand_kv),forcerange=f"{-hand_force} {hand_force}")


def build(config_path=ROOT / "configs/scene.yaml", output=ROOT / "outputs/scene.xml",
          with_hand=True, target_on_table=False, seed=None, randomize=True):
    config = yaml.safe_load(Path(config_path).read_text())
    config = apply_measured_layout(config)
    if randomize:
        config = apply_box_randomization(config, seed)
    root = ET.Element("mujoco", model="photo_reference_tissue_table")
    element(root, "compiler", angle="radian", autolimits="true")
    element(root, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast")
    visual = element(root, "visual")
    element(visual, "global", offwidth="640", offheight="480")
    element(visual, "quality", shadowsize="1024", offsamples="2")
    default = element(root, "default")
    element(default, "geom", friction=vector(config["friction"]), solref="0.01 1",
            solimp="0.95 0.99 0.001")
    element(root, "asset")
    world = element(root, "worldbody")
    # Appearance is configuration: the values below were read off a real central
    # frame and then tuned until the render matched it (see match_appearance.py).
    lighting = config.get("lighting", {})
    lights = lighting.get("lights") or [{"position": [-0.4, -0.3, 2.5], "direction": [0, 0, -1],
                                         "diffuse": [0.8, 0.8, 0.8]}]
    for light in lights:
        element(world, "light",
                pos=vector(light["position"]), dir=vector(light["direction"]),
                diffuse=vector(light.get("diffuse", [0.8, 0.8, 0.8])),
                ambient=vector(light.get("ambient", [0.0, 0.0, 0.0])),
                specular=vector(light.get("specular", [0.1, 0.1, 0.1])),
                castshadow=str(bool(light.get("castshadow", False))).lower())
    materials = config.get("materials", {})
    geom(world, "floor", [0, 0, -0.04], [2, 2, .04], materials.get("floor", [.3, .3, .3, 1]))
    width, depth, thickness = config["table"]["size"]
    surface = config["table"]["surface_z"]
    geom(world, "black_table", [0, 0, surface-thickness/2],
         [width/2, depth/2, thickness/2], materials.get("table", [.025, .028, .028, 1]))
    geom(world, "background_wall", [0, depth/2+.035, 1.3],
         [1.5, .025, .7], materials.get("wall", [.66, .67, .61, 1]))
    for x in [-width/2+.07, width/2-.07]:
        for y in [-depth/2+.07, depth/2-.07]:
            geom(world, f"leg_{x}_{y}", [x, y, surface/2], [.025, .025, surface/2],
                 materials.get("leg", [.06, .06, .06, 1]))
    tray = config["tray"]
    tx, ty = tray["center"]
    tw, td, th = tray["size"]
    wall = tray["wall_thickness"]
    tray_yaw = float(tray.get("yaw", 0.0))
    # Tray is a static welded body so the measured yaw rotates the whole box.
    tray_body = element(world, "body", name="tray",
                        pos=vector([tx, ty, surface]), euler=vector([0, 0, tray_yaw]))
    geom(tray_body, "tray_green_bottom", [0,0,.005], [tw/2,td/2,.005],
         materials.get("tray_floor", [.02,.48,.10,1]))
    for sign in [-1, 1]:
        geom(tray_body, f"tray_x_{sign}", [sign*(tw-wall)/2,0,th/2],
             [wall/2,td/2,th/2], materials.get("tray_wall", [.88,.89,.78,1]))
        geom(tray_body, f"tray_y_{sign}", [0,sign*(td-wall)/2,th/2],
             [tw/2,wall/2,th/2], materials.get("tray_wall", [.88,.89,.78,1]))
    cos_tray, sin_tray = np.cos(-tray_yaw), np.sin(-tray_yaw)
    deformable = None
    for box in config["boxes"]:
        x, y = box["position"]
        if box["name"] == "target" and target_on_table:
            x, y = target_on_table_xy(config)
        sx,sy,sz = box["size"]
        local_x = cos_tray*(x-tx) - sin_tray*(y-ty)
        local_y = sin_tray*(x-tx) + cos_tray*(y-ty)
        in_tray = abs(local_x)<tw/2 and abs(local_y)<td/2
        bottom = surface + (.01 if in_tray else 0)
        body = element(world, "body", name=box["name"],
                       pos=vector([x,y,bottom+sz/2+.001]), euler=vector([0,0,box["yaw"]]))
        element(body,"freejoint",name=box["name"]+"_free")
        bag_rgba = materials.get("bag", [.22,.53,.70,1])
        # The real object is a SOFT tissue pack: the hand keeps its fingers nearly
        # straight and the pack deforms into the grip.  A rigid box cannot reproduce
        # that (measured: 139 contact samples but zero lift), so the bag becomes a
        # volumetric deformable grid when `soft_body` is set.
        if box.get("soft_body", False):
            # A tissue pack only compresses a centimetre or so, but that is exactly
            # what lets a hand whose closed aperture is ~5.4 cm hold a 12x8.5x6.5 cm
            # pack.  A rigid box cannot do that, and <flexcomp> cannot carry the
            # elastic constants in MuJoCo 3.11, so the <flex> element is written out
            # by hand: a grid of vertices plus a 6-tetrahedra-per-cell decomposition.
            count = box.get("flex_count", [5, 4, 3])
            nx, ny, nz = (int(v) for v in count)
            vertices = []
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        vertices.extend([
                            (i / (nx - 1.0) - 0.5) * sx,
                            (j / (ny - 1.0) - 0.5) * sy,
                            (k / (nz - 1.0) - 0.5) * sz])

            def vid(i, j, k):
                return i * ny * nz + j * nz + k

            elements = []
            for i in range(nx - 1):
                for j in range(ny - 1):
                    for k in range(nz - 1):
                        v000, v100 = vid(i, j, k), vid(i + 1, j, k)
                        v010, v110 = vid(i, j + 1, k), vid(i + 1, j + 1, k)
                        v001, v101 = vid(i, j, k + 1), vid(i + 1, j, k + 1)
                        v011, v111 = vid(i, j + 1, k + 1), vid(i + 1, j + 1, k + 1)
                        elements.extend([v000, v100, v110, v111,
                                         v000, v110, v010, v111,
                                         v000, v010, v011, v111,
                                         v000, v011, v001, v111,
                                         v000, v001, v101, v111,
                                         v000, v101, v100, v111])

            # MuJoCo's <flex> carries contact properties in a <contact> child and
            # elasticity in an <elasticity> child; it has no mass attribute, the
            # attached body's mass is distributed over the vertices instead.
            inertia = [box["mass"] * (sy * sy + sz * sz) / 12.0,
                       box["mass"] * (sx * sx + sz * sz) / 12.0,
                       box["mass"] * (sx * sx + sy * sy) / 12.0]
            element(body, "inertial", pos="0 0 0", mass=str(box["mass"]),
                    diaginertia=vector(inertia))
            if deformable is None:
                deformable = element(root, "deformable")
            flex = element(deformable, "flex", body=box["name"],
                           name=box["name"] + "_soft", dim="3",
                           vertex=vector(vertices),
                           element=" ".join(str(v) for v in elements),
                           radius=str(box.get("flex_radius", 0.005)),
                           rgba=vector(bag_rgba), group="3")
            element(flex, "contact",
                    contype="1", conaffinity="1", condim=str(box.get("condim", 4)),
                    friction=vector(box.get("friction", [1.5, 0.05, 0.001])),
                    solref=vector(box.get("solref", [0.02, 1.0])),
                    solimp=vector(box.get("solimp", [0.90, 0.95, 0.002])),
                    margin=str(box.get("margin", 0.0)), gap=str(box.get("gap", 0.0)))
            element(flex, "elasticity",
                    young=str(box.get("young", 100000.0)),
                    poisson=str(box.get("poisson", 0.30)),
                    damping=str(box.get("flex_damping", 1.0)),
                    thickness=str(box.get("flex_thickness", 0.002)))
            element(flex, "edge",
                    stiffness=str(box.get("edge_stiffness", 0.0)),
                    damping=str(box.get("edge_damping", 0.0)))
            drop = {box["name"] + "_collision", box["name"] + "_bag_visual",
                    box["name"] + "_bag_visual_cross"}
            for sign_x in (-1, 1):
                for sign_y in (-1, 1):
                    drop.add(f"{box['name']}_bag_corner_{sign_x}_{sign_y}")
            for node in list(body):
                if node.get("name") in drop:
                    body.remove(node)
            continue
        scale = box.get("collision_scale", [1.0, 1.0, 1.0])
        contact = {"condim": str(box.get("condim", 4)),
                   "priority": str(box.get("priority", 1))}
        if "solref" in box:
            contact["solref"] = vector(box["solref"])
        if "solimp" in box:
            contact["solimp"] = vector(box["solimp"])
        if "friction" in box:
            contact["friction"] = vector(box["friction"])
        geom(body,box["name"]+"_collision",[0,0,0],
             [sx*scale[0]/2,sy*scale[1]/2,sz*scale[2]/2],
             bag_rgba,mass=box["mass"],group="3",**contact)
        radius=.004
        geom(body,box["name"]+"_bag_visual",[0,0,0],[sx/2-radius,sy/2,sz/2],
             bag_rgba,contype="0",conaffinity="0",mass="0")
        geom(body,box["name"]+"_bag_visual_cross",[0,0,0],[sx/2,sy/2-radius,sz/2],
             bag_rgba,contype="0",conaffinity="0",mass="0")
        for sign_x in [-1,1]:
            for sign_y in [-1,1]:
                x,y=sign_x*(sx/2-radius),sign_y*(sy/2-radius)
                element(body,"geom",name=f"{box['name']}_bag_corner_{sign_x}_{sign_y}",type="capsule",
                        fromto=vector([x,y,-sz/2+radius,x,y,sz/2-radius]),size=str(radius),
                        rgba=vector(bag_rgba),contype="0",conaffinity="0",mass="0")
        geom(body,box["name"]+"_white_top",[0,0,sz/2+.0002],
             [sx*.38,sy*.39,.0002],materials.get("bag_label",[.90,.92,.87,1]),
             contype="0",conaffinity="0",mass="0")
        geom(body,box["name"]+"_label",[0,-sy/2-.0002,0],
             [sx*.32,.0002,sz*.22],materials.get("bag_label",[.73,.84,.89,1]),
             contype="0",conaffinity="0",mass="0")
        geom(body,box["name"]+"_opening",[0,0,sz/2+.0006],
             [sx*.20,sy*.025,.0002],materials.get("bag_opening",[.25,.3,.28,1]),
             contype="0",conaffinity="0",mass="0")
    for name, camera in config["cameras"].items():
        element(world,"camera",name=name,pos=vector(camera["position"]),
                xyaxes=vector(camera["xyaxes"]) if "xyaxes" in camera else camera_axes(camera["position"],camera["target"]),
                **camera_intrinsics(camera))
    if with_hand:
        config["arm"]["mesh_cache"] = str(Path(output).parent / "converted_meshes")
        add_arm_and_hand(root,world,config)
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(output,encoding="unicode")
    if with_hand:
        assembled = mujoco.MjModel.from_xml_path(str(output))
        positions = assembled.qpos0.copy()
        initial = config["arm"]["initial_qpos"]
        for index, value in enumerate(initial,1):
            joint = assembled.joint(f"joint{index}")
            if not (assembled.jnt_range[joint.id,0] <= value <= assembled.jnt_range[joint.id,1]):
                raise ValueError(f"joint{index} initial position outside joint range")
            positions[joint.qposadr[0]] = value
        opening=hand_control.poses()["open"]
        for name,value in zip(hand_control.HAND_JOINTS,opening):
            positions[assembled.joint("hand_"+name).qposadr[0]]=value
        keyframes = element(root,"keyframe")
        element(keyframes,"key",name="preview_start",qpos=vector(positions),ctrl=vector([*initial,*opening]))
        ET.indent(root)
        ET.ElementTree(root).write(output,encoding="unicode")
    return output


def validate(model, seconds=3):
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model,data,0)
    mujoco.mj_forward(model,data)
    for _ in range(round(seconds/model.opt.timestep)):
        mujoco.mj_step(model,data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("Simulation became non-finite")
    target = model.body("target").id
    return data, {"mujoco_version":mujoco.__version__,"backend":os.environ["MUJOCO_GL"],
                  "seconds":seconds,"nq":model.nq,"nu":model.nu,
                  "contacts":data.ncon,"target_position":data.xpos[target].tolist(),
                  "robot_control_ready":False,"calibrated":False,
                  "binary_hand_control":bool(model.nu==16),
                  "robot_environment_collision":bool(model.nu==16),
                  "robot_self_collision":False}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/scene.yaml")
    parser.add_argument("--output-dir",type=Path,default=None)
    parser.add_argument("--backend",choices=["osmesa","egl","glfw"],default="osmesa")
    parser.add_argument("--no-hand",action="store_true")
    parser.add_argument("--target-on-table",action="store_true")
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--render",action="store_true")
    parser.add_argument("--viewer",action="store_true")
    parser.add_argument("--hand-state",choices=["open","closed"],default="open")
    parser.add_argument("--seed",type=int,default=None,
                        help="override box_randomization.seed for reproducible bag placement")
    parser.add_argument("--no-randomize",action="store_true",
                        help="keep the boxes at their configured positions (measured samples)")
    args=parser.parse_args()
    # Desktop users can view read-only shared sources without overwriting another
    # user's output (or root-owned files left by a previous sudo invocation).
    temporary_output = None
    if args.output_dir is None:
        if args.viewer:
            temporary_output = tempfile.TemporaryDirectory(prefix="mujoco_tissue_viewer_")
            args.output_dir = Path(temporary_output.name)
        else:
            args.output_dir = ROOT / "outputs"
    path=build(args.config,args.output_dir/"scene.xml",not args.no_hand,args.target_on_table,
               args.seed,not args.no_randomize)
    model=mujoco.MjModel.from_xml_path(str(path))
    data,report=validate(model)
    if not args.no_hand:
        hand_control.command(model,data,args.hand_state)
        for _ in range(1000):
            mujoco.mj_step(model,data)
        mujoco.mj_forward(model,data)
        report["hand_state"]=args.hand_state
        report["hand_actual_rad"]=[float(data.qpos[model.joint("hand_"+name).qposadr[0]])
                                   for name in hand_control.HAND_JOINTS]
        report["hand_pose_source"]=str(hand_control.POSE_SOURCE)
        report["simulation_time"]=float(data.time)
        print("Hand actual pose:",report["hand_actual_rad"])
    print(json.dumps(report,indent=2))
    (args.output_dir/"validation.json").write_text(json.dumps(report,indent=2))
    if args.render:
        options=mujoco.MjvOption()
        options.geomgroup[3]=0
        with mujoco.Renderer(model,height=480,width=640) as renderer:
            cameras=["central","overview"]
            if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,"left_wrist") >= 0:
                cameras.append("left_wrist")
            for camera in cameras:
                renderer.update_scene(data,camera=camera,scene_option=options)
                Image.fromarray(renderer.render()).save(args.output_dir/f"{camera}.png")
    if args.viewer:
        import mujoco.viewer
        requested={"state":args.hand_state}
        def on_key(key):
            if key in {ord("O"),ord("o")}:
                requested["state"]="open"
            elif key in {ord("C"),ord("c")}:
                requested["state"]="closed"
        print("Viewer controls: O = open hand, C = close hand. Finger spread remains fixed.")
        with mujoco.viewer.launch_passive(model,data,key_callback=on_key) as viewer:
            viewer.opt.geomgroup[3]=0
            while viewer.is_running():
                started=time.monotonic()
                with viewer.lock():
                    if not args.no_hand:
                        hand_control.command(model,data,requested["state"])
                    for _ in range(10):
                        mujoco.mj_step(model,data)
                viewer.sync()
                time.sleep(max(0,10*model.opt.timestep-(time.monotonic()-started)))
    if temporary_output is not None:
        temporary_output.cleanup()
