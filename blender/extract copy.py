# save as make_t2m_skeleton.py
import numpy as np

# --- Joint names in the standard T2M/SMPL-22 order ---
SMPL_22_NAMES = [
    "pelvis",        # 0
    "left_hip",      # 1
    "right_hip",     # 2
    "spine1",        # 3
    "left_knee",     # 4
    "right_knee",    # 5
    "spine2",        # 6
    "spine3",        # 9 in some papers, but we keep the index 9 below
    "left_ankle",    # 7
    "right_ankle",   # 8
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
# NOTE: The comments show semantic mapping; array indices below are authoritative.

# --- Parents (rooted at pelvis=0) ---
T2M_PARENTS = np.array([
    -1,  # 0 pelvis
     0,  # 1 left_hip
     0,  # 2 right_hip
     0,  # 3 spine1
     1,  # 4 left_knee
     2,  # 5 right_knee
     3,  # 6 spine2
     4,  # 7 left_ankle
     5,  # 8 right_ankle
     6,  # 9 spine3
     7,  # 10 left_foot
     8,  # 11 right_foot
     9,  # 12 neck
     9,  # 13 left_collar
     9,  # 14 right_collar
    12,  # 15 head
    13,  # 16 left_shoulder
    14,  # 17 right_shoulder
    16,  # 18 left_elbow
    17,  # 19 right_elbow
    18,  # 20 left_wrist
    19,  # 21 right_wrist
], dtype=np.int32)

# --- Canonical 5 chains (consistent with parents above) ---
HML3D_CHAINS = [
    [0, 1, 4, 7, 10],          # left leg
    [0, 2, 5, 8, 11],          # right leg
    [0, 3, 6, 9, 12, 15],      # spine -> neck -> head
    [9, 13, 16, 18, 20],       # left arm
    [9, 14, 17, 19, 21],       # right arm
]

# --- Canonical axis-aligned, unit-length offsets (root is zero) ---
t2m_raw_offsets = np.array(
    [
        [0, 0, 0],   # 0  pelvis
        [1, 0, 0],   # 1  left_hip
        [-1, 0, 0],  # 2  right_hip
        [0, 1, 0],   # 3  spine1
        [0, -1, 0],  # 4  left_knee
        [0, -1, 0],  # 5  right_knee
        [0, 1, 0],   # 6  spine2
        [0, -1, 0],  # 7  left_ankle
        [0, -1, 0],  # 8  right_ankle
        [0, 1, 0],   # 9  spine3
        [0, 0, 1],   # 10 left_foot
        [0, 0, 1],   # 11 right_foot
        [0, 1, 0],   # 12 neck
        [1, 0, 0],   # 13 left_collar
        [-1, 0, 0],  # 14 right_collar
        [0, 0, 1],   # 15 head (forward)
        [0, -1, 0],  # 16 left_shoulder
        [0, -1, 0],  # 17 right_shoulder
        [0, -1, 0],  # 18 left_elbow
        [0, -1, 0],  # 19 right_elbow
        [0, -1, 0],  # 20 left_wrist
        [0, -1, 0],  # 21 right_wrist
    ],
    dtype=np.float32,
)

def save_t2m_skeleton(output_path="skeleton_t2m22_hml3d.npz"):
    offsets = t2m_raw_offsets.astype(np.float32).copy()
    offsets[0] = 0.0  # ensure root is exactly zero

    # Optional: if your pipeline expects a different up-axis, you can rotate here.
    # T2M convention is typically +Y up, +Z forward, so we keep as-is.

    kinematic_tree = np.array([np.array(c, dtype=np.int32) for c in HML3D_CHAINS], dtype=object)

    np.savez(
        output_path,
        offsets=offsets,
        parents=T2M_PARENTS,
        joint_names=np.array(SMPL_22_NAMES, dtype=object),
        kinematic_tree=kinematic_tree,
        # (Optional extras: adjacency, template flag, etc.)
    )
    print(f"[INFO] Wrote {output_path}")
    print("[INFO] Chains:", HML3D_CHAINS)

if __name__ == "__main__":
    save_t2m_skeleton("./skeleton_t2m22_hml3d.npz")
