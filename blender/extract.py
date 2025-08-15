
import bpy
import numpy as np
from difflib import get_close_matches


# -------------------- CONFIG --------------------

ARMATURE_NAME = "SMPLX-male"       # <-- change to your armature object name
OUTPUT_PATH   = "./smpl22_offsets.npz"  # where to save the .npz

# Convert Blender Z-up to Y-up? (Many ML pipelines expect Y-up, +Z forward)
Y_UP = False

# Apply object scale from the armature object? (FBX often imports at scale 0.01)
APPLY_OBJECT_SCALE = True

# Your desired joint order (example: T2M/SMPL-22 style; edit as you like)
JOINT_NAMES = [
    "pelvis",        # 0
    "left_hip",      # 1
    "right_hip",     # 2
    "spine1",        # 3
    "left_knee",     # 4
    "right_knee",    # 5
    "spine2",        # 6
    "left_ankle",    # 7
    "right_ankle",   # 8
    "spine3",        # 9
    "left_foot",     # 10
    "right_foot",    # 11
    "neck",          # 12
    "left_collar",   # 13
    "right_collar",  # 14
    "head",          # 15
    "left_shoulder", # 16
    "right_shoulder",# 17
    "left_elbow",    # 18
    "right_elbow",   # 19
    "left_wrist",    # 20
    "right_wrist",   # 21
]

# Your desired parent array for the above order (root = -1)
PARENTS = [
    -1,  # pelvis
     0,  # left_hip
     0,  # right_hip
     0,  # spine1
     1,  # left_knee
     2,  # right_knee
     3,  # spine2
     4,  # left_ankle
     5,  # right_ankle
     6,  # spine3
     7,  # left_foot
     8,  # right_foot
     9,  # neck
     9,  # left_collar
     9,  # right_collar
    12,  # head
    13,  # left_shoulder
    14,  # right_shoulder
    16,  # left_elbow
    17,  # right_elbow
    18,  # left_wrist
    19,  # right_wrist
]

# Optional: map your target names -> FBX bone names (fill in if they differ)
# Common SMPL-X naming quirks included (wrists -> middle finger base as proxy).
NAME_MAP = {
    "pelvis":         "pelvis",
    "left_hip":       "left_hip",
    "right_hip":      "right_hip",
    "spine1":         "spine1",
    "left_knee":      "left_knee",
    "right_knee":     "right_knee",
    "spine2":         "spine2",
    "left_ankle":     "left_ankle",
    "right_ankle":    "right_ankle",
    "spine3":         "spine3",
    "left_foot":      "left_foot",
    "right_foot":     "right_foot",
    "neck":           "neck",
    "left_collar":    "left_collar",
    "right_collar":   "right_collar",
    "head":           "head",
    "left_shoulder":  "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_elbow":     "left_elbow",
    "right_elbow":    "right_elbow",
    "left_wrist":     "left_wrist",
    "right_wrist":    "right_wrist",
}

# Loose aliases to help matching (case/underscore tolerant)
ALIASES = {
    "l_hip": "left_hip", "r_hip": "right_hip",
    "l_knee": "left_knee", "r_knee": "right_knee",
    "l_ankle": "left_ankle", "r_ankle": "right_ankle",
    "l_foot": "left_foot", "r_foot": "right_foot",
    "l_collar": "left_collar", "r_collar": "right_collar",
    "l_shoulder": "left_shoulder", "r_shoulder": "right_shoulder",
    "l_elbow": "left_elbow", "r_elbow": "right_elbow",
    "l_wrist": "left_wrist", "r_wrist": "right_wrist",
    "clavicle_l": "left_collar", "clavicle_r": "right_collar",
}

# -------------------- HELPERS --------------------

def _norm_key(s: str) -> str:
    return s.lower().replace(" ", "").replace("-", "").replace("_", "")

def _build_bone_maps(bones):
    by_exact = {b.name: b for b in bones}
    by_norm  = {_norm_key(b.name): b for b in bones}
    return by_exact, by_norm

def _resolve_bone(bones_exact, bones_norm, target_name: str):
    want = NAME_MAP.get(target_name, target_name)
    cand = bones_exact.get(want) or bones_norm.get(_norm_key(want))
    if cand: return cand
    alias = ALIASES.get(target_name)
    if alias:
        want2 = NAME_MAP.get(alias, alias)
        cand = bones_exact.get(want2) or bones_norm.get(_norm_key(want2))
        if cand: return cand
    names = list(bones_exact.keys())
    guess = get_close_matches(want, names, n=3, cutoff=0.6)
    raise ValueError(f"Bone not found for target '{target_name}' -> '{want}'. Closest in FBX: {guess}")

def _extract_heads_in_object_space(arm_obj, joint_names):
    bones = arm_obj.data.bones
    bx, bn = _build_bone_maps(bones)
    S = np.array(arm_obj.scale[:], dtype=np.float32) if APPLY_OBJECT_SCALE else np.array([1,1,1], np.float32)
    P = []
    for tname in joint_names:
        b = _resolve_bone(bx, bn, tname)
        h = b.head_local
        P.append([h.x * S[0], h.y * S[1], h.z * S[2]])
    return np.asarray(P, dtype=np.float32)  # (J,3)

def _apply_y_up_rotation(X):
    Rx = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float32)
    return (X @ Rx.T)

def _lengths_from_positions(P, parents):   # <<< new: compute magnitudes only
    J = len(parents)
    L = np.zeros(J, dtype=np.float32)
    root = parents.index(-1)
    for j in range(J):
        p = parents[j]
        if p != -1:
            L[j] = np.linalg.norm(P[j] - P[p])
    L[root] = 0.0
    return L  # (J,)

# -------------------- MAIN --------------------

def main():
    arm = bpy.data.objects.get(ARMATURE_NAME)
    if arm is None:
        raise RuntimeError(f"Armature '{ARMATURE_NAME}' not found. "
                           f"Available: {[o.name for o in bpy.data.objects if o.type=='ARMATURE']}")
    if arm.type != 'ARMATURE':
        raise RuntimeError(f"Object '{ARMATURE_NAME}' is not an ARMATURE (type={arm.type})")

    print(f"[INFO] Using armature: {arm.name}")
    print(f"[INFO] Object scale: {tuple(arm.scale)}  (APPLY_OBJECT_SCALE={APPLY_OBJECT_SCALE})")
    print(f"[INFO] Target joints: {len(JOINT_NAMES)}; root index: {PARENTS.index(-1)}")

    # 1) absolute rest positions
    P = _extract_heads_in_object_space(arm, JOINT_NAMES)  # (J,3)

    # 2) optional axis conversion
    if Y_UP:
        P = _apply_y_up_rotation(P)

    # 3) bone lengths in YOUR hierarchy
    L = _lengths_from_positions(P, PARENTS)  # (J,)

    # 4) save magnitudes
    np.savez(
        OUTPUT_PATH,
        lengths=L.astype(np.float32),                  # <<<
        parents=np.asarray(PARENTS, np.int32),
        joint_names=np.asarray(JOINT_NAMES, dtype=object),
        rest_positions=P.astype(np.float32),
        meta=np.asarray([f"Y_UP={Y_UP}", f"APPLY_OBJECT_SCALE={APPLY_OBJECT_SCALE}"], dtype=object),
    )

    # 5) console preview
    nz = L[1:]
    print("[INFO] Saved:", OUTPUT_PATH)
    print("[INFO] Bone length stats (non-root): min={:.6f}, max={:.6f}, mean={:.6f}".format(
        nz.min(), nz.max(), nz.mean()
    ))
    for i,(name,len_i) in enumerate(zip(JOINT_NAMES, L)):
        p = PARENTS[i]
        print(f"  {i:2d} {name:16s} parent={p:2d}  length={len_i:.6f}")

if __name__ == "__main__":
    # If you haven't imported your FBX yet, uncomment and set the path:
    bpy.ops.import_scene.fbx(filepath="./fbx/smpl.fbx")
    main()