# ======================= Retarget 22 -> SMPL-X & Export FBX =======================
# - Loads (T,22,3) .npy motion
# - Computes SMPL-X local joint quats by matching bone directions
# - Applies to Blender armature & keys frames
# - Exports baked FBX
#
# Notes:
# * Make sure the 22-joint order & parents match your data (T2M/SMPL-22 default below).
# * If your .npy is Y-up, set CONVERT_YUP_TO_ZUP=True to convert into Blender Z-up.
# * Ensure names in MAP55_TO_22 match your SMPL-X FBX bone names.

import bpy
import numpy as np
import os

# ======================= CONFIG =======================

FBX_PATH       = "./fbx/smpl.fbx"
NPY_PATH       = "./test_gt.npy"  # (T,22,3)
ARMATURE_NAME  = "SMPLX-male"
START_FRAME    = 1
FPS            = 20

# If your npy is Y-up (+Z forward), convert into Blender Z-up
CONVERT_YUP_TO_ZUP = False

# Scale npy positions into armature space (e.g., if your npy is in meters but FBX imported at 0.01)
NPY_TO_ARMATURE_SCALE = 1.0

# Apply pelvis trajectory onto the armature OBJECT location?
APPLY_ROOT_TRAJ = True

# FBX export path (optional; set to None to skip export)
FBX_OUT_PATH    = "./test_gt.fbx"
FBX_AXIS_FORWARD = '-Z'  # typical for Unity/UE
FBX_AXIS_UP      = 'Y'

# 22-joint layout (T2M/SMPL-22 style) & parents (root = -1)
NAMES22 = [
    "pelvis","left_hip","right_hip","spine1","left_knee","right_knee","spine2",
    "left_ankle","right_ankle","spine3","left_foot","right_foot","neck",
    "left_collar","right_collar","head","left_shoulder","right_shoulder",
    "left_elbow","right_elbow","left_wrist","right_wrist",
]
PARENTS22 = [-1,0,0,0, 1,2,3, 4,5,6, 7,8, 9,9,9, 12,13,14, 16,17, 18,19]

# Map: SMPL-X bone name -> your 22-name (edit keys to match your FBX bone names)
MAP55_TO_22 = {
    "pelvis":"pelvis",
    "left_hip":"left_hip",       "right_hip":"right_hip",
    "left_knee":"left_knee",     "right_knee":"right_knee",
    "left_ankle":"left_ankle",   "right_ankle":"right_ankle",
    "left_foot":"left_foot",     "right_foot":"right_foot",
    "spine1":"spine1", "spine2":"spine2", "spine3":"spine3",
    "neck":"neck", "head":"head",
    "left_collar":"left_collar", "right_collar":"right_collar",
    "left_shoulder":"left_shoulder","right_shoulder":"right_shoulder",
    "left_elbow":"left_elbow",   "right_elbow":"right_elbow",
    "left_wrist":"left_wrist",   "right_wrist":"right_wrist",
}

# ======================= Quat helpers =======================

# Try to use your own quaternion utils; else fallback to light NumPy impls
try:
    from common.quaternion import qbetween_np as qbetween, qmul_np as qmul, qinv_np as qinv, qrot_np as qrot
    HAVE_OWN_QUATS = True
except Exception:
    HAVE_OWN_QUATS = False
    EPS = 1e-8
    def _unit(v):
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / (n + EPS)
    def qnormalize(q):
        n = np.linalg.norm(q, axis=-1, keepdims=True)
        return q / (n + EPS)
    def qbetween(a, b):
        a = _unit(a); b = _unit(b)
        c = np.cross(a, b)
        d = np.sum(a*b, axis=1, keepdims=True)
        w = np.sqrt((1.0 + np.clip(d, -1.0, 1.0)) * 2.0)
        q = np.concatenate([w/2.0, c / (w + EPS)], axis=1)
        mask = (w[:,0] < 1e-6)
        if np.any(mask):
            aa = a[mask]
            t = np.array([1,0,0], np.float32)
            alt = np.cross(aa, t)
            small = np.linalg.norm(alt, axis=1) < 1e-6
            if np.any(small):
                t2 = np.array([0,1,0], np.float32)
                alt[small] = np.cross(aa[small], t2)
            alt = _unit(alt)
            rep = np.concatenate([np.zeros((alt.shape[0],1), np.float32), alt], axis=1)
            q[mask] = rep
        return qnormalize(q)
    def qmul(a, b):
        aw, ax, ay, az = a[:,0], a[:,1], a[:,2], a[:,3]
        bw, bx, by, bz = b[:,0], b[:,1], b[:,2], b[:,3]
        return np.stack([
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw
        ], axis=1)
    def qinv(q):
        return np.stack([q[:,0], -q[:,1], -q[:,2], -q[:,3]], axis=1)
    def qrot(q, v):
        qvec = q[:,1:]
        uv   = np.cross(qvec, v)
        uuv  = np.cross(qvec, uv)
        return v + 2*(q[:, :1]*uv + uuv)

def Rx_pos90(V):  # rotate +90° about X (Y-up -> Z-up)
    x = V[...,0:1]; y = V[...,1:2]; z = V[...,2:3]
    return np.concatenate([x, -z, y], axis=-1)

# ======================= Armature helpers =======================

def import_fbx_if_needed(fbx_path):
    if not fbx_path:
        return
    if not os.path.exists(fbx_path):
        raise RuntimeError(f"FBX not found: {fbx_path}")
    print(f"[INFO] Importing FBX: {fbx_path}")
    bpy.ops.import_scene.fbx(filepath=fbx_path)

def get_armature(name_hint=None):
    if name_hint:
        obj = bpy.data.objects.get(name_hint)
        if obj and obj.type == 'ARMATURE':
            return obj
    # fallback: first armature in the scene
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            print(f"[WARN] Using armature '{obj.name}' (hint '{name_hint}' not found).")
            return obj
    raise RuntimeError("No ARMATURE object found in the scene.")

def smplx_names_parents(arm):
    bones = list(arm.data.bones)
    names = [b.name for b in bones]
    index = {n:i for i,n in enumerate(names)}
    parents = []
    for b in bones:
        parents.append(-1 if b.parent is None else index[b.parent.name])
    return names, parents

def rest_offsets(arm, names55, parents55):
    bones = {b.name:b for b in arm.data.bones}
    P = np.zeros((len(names55), 3), np.float32)
    for i, n in enumerate(names55):
        h = bones[n].head_local
        P[i] = (h.x, h.y, h.z)
    off = np.zeros_like(P)
    root = parents55.index(-1)
    for j,p in enumerate(parents55):
        off[j] = 0 if p == -1 else (P[j]-P[p])
    off[root] = 0
    return off

def topo_order(parents):
    parents = list(parents)
    root = parents.index(-1)
    kids = [[] for _ in range(len(parents))]
    for i,p in enumerate(parents):
        if p != -1: kids[p].append(i)
    order, stack = [root], kids[root][::-1]
    while stack:
        j = stack.pop(); order.append(j); stack.extend(kids[j][::-1])
    return order, root

# ======================= Retarget core =======================

def retarget_quats(j22, names22, parents22, names55, parents55, offsets55, map55_to22, up_vec):
    T = j22.shape[0]
    J55 = len(names55)
    quats = np.zeros((T, J55, 4), np.float32)
    Rw    = np.zeros((T, J55, 4), np.float32)

    udirs = offsets55.copy()
    udirs[1:] = udirs[1:] / (np.linalg.norm(udirs[1:], axis=1, keepdims=True) + 1e-8)
    order55, root55 = topo_order(parents55)

    name2idx22 = {n:i for i,n in enumerate(names22)}
    name2idx55 = {n:i for i,n in enumerate(names55)}
    mapped = {name2idx55[k]: name2idx22[v]
              for k,v in map55_to22.items()
              if (k in name2idx55 and v in name2idx22)}

    # Root orientation from hips+shoulders
    rhip = name2idx22["right_hip"]; lhip = name2idx22["left_hip"]
    rsho = name2idx22["right_shoulder"]; lsho = name2idx22["left_shoulder"]
    across = (j22[:, rhip]-j22[:, lhip]) + (j22[:, rsho]-j22[:, lsho])
    across = across / (np.linalg.norm(across, axis=1, keepdims=True) + 1e-8)
    up = np.tile(np.asarray(up_vec, np.float32), (T,1))
    fwd = np.cross(up, across); fwd = fwd / (np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-8)
    target = np.tile(np.array([0,0,1], np.float32), (T,1))
    root_world = qbetween(fwd, target)

    quats[:, root55] = root_world
    Rw[:, root55]    = root_world

    mapped_set = set(mapped.keys())
    for j in order55[1:]:
        p = parents55[j]
        if (j in mapped_set) and (p in mapped_set):
            cj22 = mapped[j]; pj22 = mapped[p]
            v = j22[:, cj22] - j22[:, pj22]
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
            u = np.tile(udirs[j], (T,1))
            rot_u_v = qbetween(u, v)                 # world rot to aim rest dir at observed dir
            R_loc   = qmul(qinv(Rw[:, p]), rot_u_v)  # local to parent
            quats[:, j] = R_loc
            Rw[:, j]    = qmul(Rw[:, p], R_loc)
        else:
            quats[:, j] = np.array([1,0,0,0], np.float32)    # keep rest
            Rw[:, j]    = qmul(Rw[:, p], quats[:, j])

    return quats

def fk_verify(quats, parents55, offsets55):
    T, J, _ = quats.shape
    order55, root55 = topo_order(parents55)
    joints = np.zeros((T, J, 3), np.float32)
    Rw = np.zeros((T, J, 4), np.float32)
    Rw[:, root55] = quats[:, root55]
    for j in order55[1:]:
        p = parents55[j]
        Rw[:, j] = qmul(Rw[:, p], quats[:, j])
        off = np.tile(offsets55[j], (T,1))
        joints[:, j] = qrot(Rw[:, j], off) + joints[:, p]
    return joints

# ======================= Apply & Export =======================

def apply_quats(arm, names55, quats, start_frame, fps, root_traj=None, parents55=None):
    bpy.context.scene.render.fps = fps
    T = quats.shape[0]

    if bpy.context.object != arm:
        bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')

    for pb in arm.pose.bones:
        pb.rotation_mode = 'QUATERNION'

    # --- put root motion on the ROOT BONE instead of the object ---
    if root_traj is not None and parents55 is not None:
        root_idx = parents55.index(-1)
        root_bone_name = names55[root_idx]
        pb_root = arm.pose.bones[root_bone_name]

        base = root_traj[0].copy()
        for t in range(T):
            frame = start_frame + t
            # translation in armature space
            loc = (root_traj[t] - base)
            pb_root.location = (loc[0], loc[1], loc[2])
            pb_root.keyframe_insert(data_path="location", frame=frame)

    # rotations for all bones
    for t in range(T):
        frame = start_frame + t
        for i, name in enumerate(names55):
            pb = arm.pose.bones.get(name)
            if pb is None:
                continue
            w, x, y, z = quats[t, i]
            pb.rotation_quaternion = (w, x, y, z)
            pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)

def export_fbx(out_path, armature_name, frame_start, frame_end,
               axis_forward='-Z', axis_up='Y', use_selection=True):
    import bpy, os
    scn = bpy.context.scene
    scn.frame_start = frame_start
    scn.frame_end   = frame_end

    # ensure directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Resolve armature
    arm = bpy.data.objects.get(armature_name)
    if arm is None:
        raise RuntimeError(f"Armature '{armature_name}' not found")

    # Make sure we're in OBJECT mode
    try:
        if arm.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass  # in background this can be harmless

    # Clear selection using data API (no operator)
    for o in scn.objects:
        o.select_set(False)

    # Select only the armature (and optionally its children meshes)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm

    if use_selection:
        # Optional: include meshes parented to the armature
        for child in arm.children:
            if child.type in {'MESH', 'EMPTY'}:
                child.select_set(True)

    # Use a safe context override for the exporter
    override = {
        "selected_objects": [o for o in scn.objects if o.select_get()],
        "active_object": arm,
        "object": arm,
        "scene": scn,
        "view_layer": bpy.context.view_layer,
    }

    with bpy.context.temp_override(**override):
        bpy.ops.export_scene.fbx(
            filepath=out_path,
            use_selection=use_selection,
            add_leaf_bones=False,
            use_armature_deform_only=False,   # <--- keep ALL bones
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
            axis_forward=axis_forward,
            axis_up=axis_up,
            path_mode='AUTO',
        )
    print(f"[INFO] Exported FBX -> {out_path}")

# ======================= Main =======================

def main():
    # 0) Import FBX if requested
    import_fbx_if_needed(FBX_PATH)

    # 1) Get/resolve armature
    arm = get_armature(ARMATURE_NAME)

    # 2) Load motion
    assert os.path.exists(NPY_PATH), f"Missing NPY: {NPY_PATH}"
    j22 = np.load(NPY_PATH).astype(np.float32)  # (T,22,3)
    assert j22.ndim == 3 and j22.shape[1:] == (22,3), f"Expected (T,22,3), got {j22.shape}"

    # 3) Axis & scale into Blender armature space
    if CONVERT_YUP_TO_ZUP:
        j22 = Rx_pos90(j22)   # Y-up -> Z-up
        up_vec = (0,0,1)
    else:
        up_vec = (0,1,0)
    j22 *= NPY_TO_ARMATURE_SCALE

    # 4) SMPL-X structure from the armature
    names55, parents55 = smplx_names_parents(arm)
    offsets55 = rest_offsets(arm, names55, parents55)

    # 5) Retarget to local quats
    quats55 = retarget_quats(
        j22, NAMES22, PARENTS22,
        names55, parents55,
        offsets55,
        MAP55_TO_22,
        up_vec=up_vec
    )

    # 6) Optional root trajectory (from pelvis of the 22-data)
    root_traj = None
    if APPLY_ROOT_TRAJ:
        pelvis_idx_22 = NAMES22.index("pelvis")
        root_traj = j22[:, pelvis_idx_22]

    # 7) Apply & key
    apply_quats(arm, names55, quats55, START_FRAME, FPS,
            root_traj=root_traj, parents55=parents55)

    # 8) Export FBX
    if FBX_OUT_PATH:
        T = j22.shape[0]
        export_fbx(FBX_OUT_PATH, arm.name, START_FRAME, START_FRAME + T - 1,
                   axis_forward=FBX_AXIS_FORWARD, axis_up=FBX_AXIS_UP)
        print(f"[INFO] Exported FBX -> {FBX_OUT_PATH}")

    print("[INFO] Retarget done. Frames:", j22.shape[0])

if __name__ == "__main__":
    main()