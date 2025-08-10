import numpy as np
from scipy.spatial.transform import Rotation as R

SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle",
    "spine1", "spine2", "spine3",
    "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_toe", "right_toe",  # dummy toe joints
    "left_hand", "right_hand" # dummy hand joints
]

SMPL_PARENTS = [
    -1,  # pelvis
    0, 1, 2,         # left leg
    0, 4, 5,         # right leg
    0, 7, 8,         # spine
    9, 10,           # neck, head
    9, 13, 14,       # left arm
    9, 16, 17,       # right arm
    3, 6,            # left/right toe
    15, 18           # left/right hand (dummy)
]

SMPL_OFFSETS = np.array([
    [ 0.0000,  0.0000,  0.0000],  # pelvis
    [ 0.0000, -0.0850,  0.0170],  # left_hip
    [ 0.0000, -0.4160,  0.0000],  # left_knee
    [ 0.0000, -0.4210,  0.0000],  # left_ankle
    [ 0.0000, -0.0850, -0.0170],  # right_hip
    [ 0.0000, -0.4160,  0.0000],  # right_knee
    [ 0.0000, -0.4210,  0.0000],  # right_ankle
    [ 0.0000,  0.0850,  0.0000],  # spine1
    [ 0.0000,  0.1250,  0.0000],  # spine2
    [ 0.0000,  0.1200,  0.0000],  # spine3
    [ 0.0000,  0.1400,  0.0000],  # neck
    [ 0.0000,  0.1100,  0.0000],  # head
    [ 0.0450,  0.1000,  0.0000],  # left_shoulder
    [ 0.1500,  0.0000,  0.0000],  # left_elbow
    [ 0.1600,  0.0000,  0.0000],  # left_wrist
    [-0.0450,  0.1000,  0.0000],  # right_shoulder
    [-0.1500,  0.0000,  0.0000],  # right_elbow
    [-0.1600,  0.0000,  0.0000],  # right_wrist
    [ 0.0000, -0.0500,  0.1200],  # left_toe (dummy)
    [ 0.0000, -0.0500,  0.1200],  # right_toe (dummy)
    [ 0.0600,  0.0000,  0.0000],  # left_hand (dummy)
    [-0.0600,  0.0000,  0.0000],  # right_hand (dummy)
])


def write_bvh(save_path, parsed_data, framerate=20):
    joint_names = parsed_data['joint_names']
    parents = parsed_data['parents']
    root_trans = parsed_data['root_trans']        # [T, 3]
    euler_angles = parsed_data['euler_angles']    # [T, J, 3]

    # Use predefined offsets
    assert len(joint_names) == len(SMPL_OFFSETS)
    offsets = SMPL_OFFSETS

    T = root_trans.shape[0]
    J = len(joint_names)

    joint_channels = {name: "Zrotation Yrotation Xrotation" for name in joint_names}
    joint_channels[joint_names[0]] = "Xposition Yposition Zposition Zrotation Yrotation Xrotation"

    child_map = {i: [] for i in range(J)}
    for i, p in enumerate(parents):
        if p != -1:
            child_map[p].append(i)

    def write_joint(f, idx, indent=0):
        name = joint_names[idx]
        tabs = "  " * indent
        offset = offsets[idx]
        joint_type = "ROOT" if parents[idx] == -1 else "JOINT"

        f.write(f"{tabs}{joint_type} {name}\n")
        f.write(f"{tabs}{{\n")
        f.write(f"{tabs}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")

        f.write(f"{tabs}  CHANNELS {len(joint_channels[name].split())} {joint_channels[name]}\n")
        for child_idx in child_map[idx]:
            write_joint(f, child_idx, indent + 1)

        f.write(f"{tabs}}}\n")

    with open(save_path, "w") as f:
        f.write("HIERARCHY\n")
        write_joint(f, 0)
        f.write("MOTION\n")
        f.write(f"Frames: {T}\n")
        f.write(f"Frame Time: {1.0 / framerate:.8f}\n")

        for t in range(T):
            frame_vals = []
            for j in range(J):
                if j == 0:
                    pos = root_trans[t]
                    frame_vals += [pos[0], pos[1], pos[2]]

                rot = euler_angles[t, j]  # [Z, Y, X]
                frame_vals += [rot[2], rot[1], rot[0]]  # X, Y, Z

            f.write(" ".join(f"{v:.6f}" for v in frame_vals) + "\n")

    print(f"✅ BVH saved to: {save_path}")