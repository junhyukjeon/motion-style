# ================== Visualize joints (T,J,3) as an armature & export FBX ==================
# Run in Blender (Text Editor → Run Script) or: blender --background --python this.py

import bpy, numpy as np, os

# ---------------------- CONFIG ----------------------
NPY_PATH = "./test_ric.npy"            # shape (T,J,3)
FBX_OUT  = "./test_ric_out.fbx"        # where to save; set None to skip export
ARM_NAME = "JointsVis"

FPS         = 20
START_FRAME = 1

SOURCE_IS_LOCAL_TO_PARENT = False  # True if npy stores child = parent + local_offset (world axes)
CONVERT_YUP_TO_ZUP        = True   # If npy is Y-up (+Z forward), rotate into Blender Z-up
GLOBAL_SCALE              = 1.0     # overall scale

# Your joint order & parents (*root = -1*) — default: T2M/SMPL-22
JOINT_NAMES = [
    "pelvis","left_hip","right_hip","spine1","left_knee","right_knee","spine2",
    "left_ankle","right_ankle","spine3","left_foot","right_foot","neck",
    "left_collar","right_collar","head","left_shoulder","right_shoulder",
    "left_elbow","right_elbow","left_wrist","right_wrist",
]
PARENTS = [-1,0,0,0, 1,2,3, 4,5,6, 7,8, 9,9,9, 12,13,14, 16,17, 18,19]

# Root “aim” child to define root rotation (pick a child of the root)
ROOT_AIM_CHILD_NAME = "spine2"

# Put root translation on the armature object (True) or on the root bone (False)
ROOT_MOTION_ON_OBJECT = True

# FBX axes for most DCCs (Unity/UE): -Z forward, +Y up
FBX_AXIS_FORWARD = '-Z'
FBX_AXIS_UP      = 'Y'

# ---------------------- math helpers ----------------------
def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + 1e-8)

def qnormalize(q):
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / (n + 1e-8)

def qbetween(a, b):
    """Quaternion rotating vector a -> b. a,b: (N,3). Returns (N,4) as (w,x,y,z)."""
    a = _unit(a); b = _unit(b)
    c = np.cross(a, b)
    d = np.sum(a*b, axis=1, keepdims=True)
    w = np.sqrt((1.0 + np.clip(d, -1.0, 1.0)) * 2.0)
    q = np.concatenate([w*0.5, c/(w + 1e-8)], axis=1)
    # Handle a ~ -b (180°)
    mask = (w[:,0] < 1e-6)
    if np.any(mask):
        aa = a[mask]
        t  = np.array([1,0,0], np.float32)
        alt = np.cross(aa, t)
        small = np.linalg.norm(alt, axis=1) < 1e-6
        if np.any(small):
            t2 = np.array([0,1,0], np.float32)
            alt[small] = np.cross(aa[small], t2)
        alt = _unit(alt)
        q[mask] = np.concatenate([np.zeros((alt.shape[0],1),np.float32), alt], axis=1)
    return qnormalize(q)

def qmul(a, b):
    """(N,4)x(N,4)->(N,4)"""
    aw,ax,ay,az = a[:,0],a[:,1],a[:,2],a[:,3]
    bw,bx,by,bz = b[:,0],b[:,1],b[:,2],b[:,3]
    return np.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    ], axis=1)

def qinv(q):
    return np.stack([q[:,0], -q[:,1], -q[:,2], -q[:,3]], axis=1)

def Rx_pos90(V):  # rotate +90° about X (Y-up -> Z-up): (x,y,z)->(x,-z,y)
    x = V[...,0:1]; y = V[...,1:2]; z = V[...,2:3]
    return np.concatenate([x, -z, y], axis=-1)

def _safe_head_tail(head, tail, fallback_axis=(0, 0.1, 0)):
    """Ensure non-zero bone length; nudge tail if head≈tail."""
    head = np.asarray(head, np.float32)
    tail = np.asarray(tail, np.float32)
    if np.linalg.norm(tail - head) < 1e-6:
        tail = head + np.asarray(fallback_axis, np.float32)
    return tuple(head), tuple(tail)

# ---------------------- data prep ----------------------
assert os.path.exists(NPY_PATH), f"Missing npy: {NPY_PATH}"
X = np.load(NPY_PATH).astype(np.float32)  # (T,J,3)
assert X.ndim == 3 and X.shape[1] == len(JOINT_NAMES), f"Expected (T,{len(JOINT_NAMES)},3), got {X.shape}"

T, J, _ = X.shape
root = PARENTS.index(-1)
name_to_idx = {n:i for i,n in enumerate(JOINT_NAMES)}

# If source stores local (child-parent) offsets in world axes, accumulate to world
if SOURCE_IS_LOCAL_TO_PARENT:
    order = [root]
    kids = [[] for _ in range(J)]
    for i,p in enumerate(PARENTS):
        if p != -1: kids[p].append(i)
    stack = kids[root][::-1]
    while stack:
        j = stack.pop(); order.append(j); stack.extend(kids[j][::-1])

    W = np.zeros_like(X)
    W[:, root] = 0.0
    for j in order[1:]:
        p = PARENTS[j]
        W[:, j] = W[:, p] + X[:, j]
    X = W

if CONVERT_YUP_TO_ZUP:
    X = Rx_pos90(X)

X *= GLOBAL_SCALE
P0 = X[0]

# ---------------------- build armature from frame 0 ----------------------
# Clean up any previous armature with the same name
old = bpy.data.objects.get(ARM_NAME)
if old:
    bpy.data.objects.remove(old, do_unlink=True)

def new_armature(name=ARM_NAME):
    for o in bpy.context.scene.objects:
        o.select_set(False)

    arm_data = bpy.data.armatures.new(name)
    arm_obj  = bpy.data.objects.new(name, arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')

    ebones = {}
    bone_names_in_order = [None] * J

    aim_child = name_to_idx.get(ROOT_AIM_CHILD_NAME, None)

    # create bones
    for j, n in enumerate(JOINT_NAMES):
        eb = arm_data.edit_bones.new(n)
        ebones[j] = eb
        bone_names_in_order[j] = eb.name  # capture actual name

    # place & parent with zero-length safety
    for j in range(J):
        p = PARENTS[j]
        eb = ebones[j]
        if p == -1:
            if aim_child is not None:
                h, t = _safe_head_tail(P0[j], P0[aim_child], fallback_axis=(0, 0.1, 0))
            else:
                h, t = tuple(P0[j]), (P0[j][0], P0[j][1] + 0.1, P0[j][2])
            eb.head, eb.tail = h, t
        else:
            h, t = _safe_head_tail(P0[p], P0[j], fallback_axis=(0, 0.1, 0))
            eb.head, eb.tail = h, t
            eb.parent = ebones[p]
            eb.use_connect = True

    bpy.ops.object.mode_set(mode='POSE')
    for pb in arm_obj.pose.bones:
        pb.rotation_mode = 'QUATERNION'

    return arm_obj, bone_names_in_order

arm, BONE_NAMES = new_armature()

pose_bones = arm.pose.bones
missing = [n for n in BONE_NAMES if pose_bones.get(n) is None]
if missing:
    print("[ERROR] Missing pose bones:", missing)
    print("[HINT] Available:", [pb.name for pb in pose_bones])
    raise RuntimeError("Pose bones not found (name mismatch).")

# Precompute unit rest directions u[j] = normalize(rest[j]-rest[parent])
udirs = np.zeros((J,3), np.float32)
for j in range(J):
    p = PARENTS[j]
    if p == -1:
        if ROOT_AIM_CHILD_NAME in name_to_idx:
            c = name_to_idx[ROOT_AIM_CHILD_NAME]
            udirs[j] = _unit((P0[c] - P0[j])[None,:])[0]
        else:
            ch = [k for k in range(J) if PARENTS[k] == j]
            if ch:
                v = np.mean([P0[k]-P0[j] for k in ch], axis=0)
                udirs[j] = _unit(v[None,:])[0]
            else:
                udirs[j] = np.array([0,0,1], np.float32)
    else:
        udirs[j] = _unit((P0[j]-P0[p])[None,:])[0]

# Topo order
order = [root]
kids = [[] for _ in range(J)]
for i,p in enumerate(PARENTS):
    if p != -1: kids[p].append(i)
stack = kids[root][::-1]
while stack:
    j = stack.pop(); order.append(j); stack.extend(kids[j][::-1])

# ---------------------- animate ----------------------
bpy.context.scene.render.fps = FPS

# Optional: put root translation on the armature object
if ROOT_MOTION_ON_OBJECT:
    base = X[0, root].copy()
    for t in range(T):
        frame = START_FRAME + t
        arm.location = (X[t, root] - base)
        arm.keyframe_insert(data_path="location", frame=frame)

for t in range(T):
    frame = START_FRAME + t
    # per-frame directions
    vdirs = np.zeros_like(udirs)
    for j in range(J):
        p = PARENTS[j]
        if p == -1:
            if ROOT_AIM_CHILD_NAME in name_to_idx:
                c = name_to_idx[ROOT_AIM_CHILD_NAME]
                vdirs[j] = _unit((X[t, c] - X[t, j])[None,:])[0]
            else:
                vdirs[j] = udirs[j]
        else:
            vdirs[j] = _unit((X[t, j] - X[t, p])[None,:])[0]

    # accumulate world quats
    Rw = np.zeros((J,4), np.float32)
    Rw[root] = qbetween(udirs[root][None,:], vdirs[root][None,:])[0]

    # if root translation goes on the bone (not the object)
    if not ROOT_MOTION_ON_OBJECT:
        pb_root = pose_bones[BONE_NAMES[root]]
        base = X[0, root]
        loc = X[t, root] - base
        pb_root.location = (loc[0], loc[1], loc[2])
        pb_root.keyframe_insert(data_path="location", frame=frame)

    # rotations for children
    for j in order[1:]:
        p = PARENTS[j]
        rot_u_to_v = qbetween(udirs[j][None,:], vdirs[j][None,:])[0]
        Rloc = qmul(qinv(Rw[p][None,:]), rot_u_to_v[None,:])[0]
        Rw[j] = qmul(Rw[p][None,:], Rloc[None,:])[0]

        pb = pose_bones[BONE_NAMES[j]]
        w,x,y,z = Rloc
        pb.rotation_quaternion = (w,x,y,z)
        pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    # root rotation
    pb0 = pose_bones[BONE_NAMES[root]]
    w,x,y,z = Rw[root]
    pb0.rotation_quaternion = (w,x,y,z)
    pb0.keyframe_insert(data_path="rotation_quaternion", frame=frame)

print(f"[INFO] Keyed {T} frames for {J} joints on armature '{ARM_NAME}'.")

# ---------------------- export FBX ----------------------
def export_fbx(out_path, arm_obj):
    scn = bpy.context.scene
    scn.frame_start = START_FRAME
    scn.frame_end   = START_FRAME + T - 1
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    try:
        if arm_obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass

    for o in scn.objects: o.select_set(False)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj

    override = {
        "selected_objects": [arm_obj],
        "active_object": arm_obj,
        "object": arm_obj,
        "scene": scn,
        "view_layer": bpy.context.view_layer,
    }
    with bpy.context.temp_override(**override):
        bpy.ops.export_scene.fbx(
            filepath=out_path,
            use_selection=True,
            add_leaf_bones=False,
            use_armature_deform_only=False,  # keep all bones
            apply_unit_scale=True,
            global_scale=1.0,
            bake_anim=True,
            bake_anim_use_all_bones=True,
            bake_anim_use_nla_strips=False,
            bake_anim_use_all_actions=False,
            bake_anim_force_startend_keying=True,
            bake_anim_step=1.0,
            bake_anim_simplify_factor=0.0,
            armature_nodetype='NULL',
            axis_forward=FBX_AXIS_FORWARD,
            axis_up=FBX_AXIS_UP,
            path_mode='AUTO',
        )
    print(f"[INFO] Exported FBX -> {out_path}")

if FBX_OUT:
    export_fbx(FBX_OUT, arm)
