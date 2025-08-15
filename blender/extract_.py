import bpy
import numpy as np

SMPL_22_NAMES = [
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
    "right_shoulder",# 13  ← right shoulder first (so chain [9,13,16,18,20])
    "left_shoulder", # 14
    "head",          # 15
    "right_elbow",   # 16
    "left_elbow",    # 17
    "right_wrist",   # 18
    "left_wrist",    # 19
    "right_hand",    # 20
    "left_hand",     # 21
]

# Parents aligned to those chains
SMPL_22_PARENTS = [
    -1,  # 0 pelvis
     0,  # 1 left_hip    ← [0,1,4,7,10]
     0,  # 2 right_hip   ← [0,2,5,8,11]
     0,  # 3 spine1      ← [0,3,6,9,12,15]
     1,  # 4 left_knee
     2,  # 5 right_knee
     3,  # 6 spine2
     4,  # 7 left_ankle
     5,  # 8 right_ankle
     6,  # 9 spine3
     7,  # 10 left_foot
     8,  # 11 right_foot
     9,  # 12 neck
     9,  # 13 right_shoulder  ← arms start from spine3 (9)
     9,  # 14 left_shoulder
    12,  # 15 head (from neck)
    13,  # 16 right_elbow
    14,  # 17 left_elbow
    16,  # 18 right_wrist
    17,  # 19 left_wrist
    18,  # 20 right_hand
    19,  # 21 left_hand
]

HAND_MAP = {
    "left_hand": "left_middle1",
    "right_hand": "right_middle1"
}

def import_fbx(fbx_path):
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    print(f"[INFO] Imported FBX: {fbx_path}")

def extract_skeleton(armature_name="SMPLX-male", output_path="skeleton_smpl22.npz"):
    armature = bpy.data.objects.get(armature_name)
    if armature is None:
        raise ValueError(f"Armature '{armature_name}' not found.")

    bones = armature.data.bones
    bone_map = {bone.name: bone for bone in bones}

    joint_names = []
    offsets = []
    parents = []

    for idx, name in enumerate(SMPL_22_NAMES):
        target_name = HAND_MAP.get(name, name)  # Replace hands with middle1 joints
        bone = bone_map.get(target_name)
        if bone is None:
            raise ValueError(f"Missing joint in armature: {target_name}")

        joint_names.append(name)  # Keep SMPL naming
        parent_idx = SMPL_22_PARENTS[idx]

        if parent_idx == -1:
            offset = bone.head_local
        else:
            parent_name = HAND_MAP.get(SMPL_22_NAMES[parent_idx], SMPL_22_NAMES[parent_idx])
            offset = bone.head_local - bone_map[parent_name].head_local

        offsets.append([offset.x, offset.y, offset.z])
        parents.append(parent_idx)

    # Build kinematic tree
    kinematic_tree = []
    visited = set()
    for i, p in enumerate(parents):
        if p == -1 and i not in visited:
            chain = [i]
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                children = [j for j, pj in enumerate(parents) if pj == curr and j not in visited]
                queue.extend(children)
                chain.extend(children)
                visited.update(children)
            kinematic_tree.append(np.array(chain, dtype=np.int32))

    np.savez(
        output_path,
        offsets=np.array(offsets, dtype=np.float32),
        parents=np.array(parents, dtype=np.int32),
        joint_names=np.array(joint_names, dtype=object),
        kinematic_tree=np.array(kinematic_tree, dtype=object),
    )

    print(f"[INFO] Saved SMPL-22 skeleton to: {output_path}")
    print(f"[INFO] left_hand → {HAND_MAP['left_hand']}, right_hand → {HAND_MAP['right_hand']}")


# === Main === #
if __name__ == "__main__":
    fbx_path = "./fbx/smpl.fbx"
    output_path = "./skeleton.npz"
    import_fbx(fbx_path)
    extract_skeleton(armature_name="SMPLX-male", output_path=output_path)
