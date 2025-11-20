import numpy as np
import torch
from utils.quaternion import *

T2M_TEMPLATE = np.array([
    [0,0,0],
    [1,0,0],  [-1,0,0], [0,1,0],
    [0,-1,0], [0,-1,0], [0, 1,0],
    [0,-1,0], [0,-1,0], [0, 1,0],
    [0,0,1],  [0,0,1],  [0, 1,0],
    [1,0,0],  [-1,0,0], [0,0,1],
    [0,-1,0], [0,-1,0], [0,-1,0],
    [0,-1,0], [0,-1,0], [0,-1,0],
], dtype=np.float32)

def _unit_dirs(template: np.ndarray) -> np.ndarray:
    u = template.astype(np.float32).copy()
    u[0] = 0.0
    for j in range(1, len(u)):
        n = np.linalg.norm(u[j])
        u[j] = u[j] / (n + 1e-8) if n > 0 else 0.0
    return u

class Skeleton:
    def __init__(self, parents, lengths, *,
                 face_indices=(1,2,11,12),
                 up_axis=(0,1,0),
                 target_forward=(0,0,1),
                 smooth_forward_sigma=0.0,
                 first_frame_root_identity=True,
                 zero_root_length=True):
        
        self.parents = [int(p) for p in parents]
        self.J = len(self.parents)
        self.root = self.parents.index(-1)

        # ============ #
        # Policy knobs #
        # ============ #
        self.face_indices              = tuple(int(i) for i in face_indices)
        self.up_axis                   = np.asarray(up_axis, dtype=np.float32)
        self.target_forward            = np.asarray(target_forward, dtype=np.float32)
        self.smooth_forward_sigma      = float(smooth_forward_sigma)
        self.first_frame_root_identity = bool(first_frame_root_identity)
        
        # T2M unit directions
        dirs_np = _unit_dirs(T2M_TEMPLATE)
        dirs    = torch.tensor(dirs_np, dtype=torch.float32)

        # Bone lengths -> (J,1) and optional zero root length
        L = torch.as_tensor(lengths, dtype=torch.float32).view(self.J, 1).clone()
        if zero_root_length:
            L[self.root] = 0.0

        # Final offsets (J,3) = dir * length
        self.offsets = dirs * L 

        # Parent-first joint order
        kids = [[] for _ in range(self.J)]
        for j, p in enumerate(self.parents):
            if p != -1:
                kids[p].append(j)
        order, stack = [self.root], kids[self.root][::-1]
        while stack:
            j = stack.pop()
            order.append(j)
            stack.extend(kids[j][::-1])
        self.order = order

    def forward_kinematics_cont6d(self, cont6d_params, root_pos, do_root_R=True):
        B, J = cont6d_params.shape[:2]
        device, dtype = cont6d_params.device, cont6d_params.dtype
        joints = torch.zeros((B, J, 3), device=device, dtype=dtype)
        joints[:, self.root] = root_pos

        # local -> matrix once per joint
        Rloc = [cont6d_to_matrix(cont6d_params[:, j]) for j in range(J)]

        # world rotations
        Rw = [None]*J
        Rw[self.root] = Rloc[self.root] if do_root_R else torch.eye(3, device=device, dtype=dtype).expand(B,3,3)

        # offsets to right device/dtype
        off = self.offsets.to(device=device, dtype=dtype)

        for j in self.order[1:]:
            p = self.parents[j]
            Rw[j] = Rw[p] @ Rloc[j]
            joints[:, j] = (Rw[j] @ off[j].unsqueeze(-1)).squeeze(-1) + joints[:, p]
        return joints

    def inverse_kinematics(self, joints, root_quat):
        joints = np.asarray(joints, dtype=np.float32)
        T, J = joints.shape[:2]

        def _normalize(v):
            return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)

        # Root rotation
        if root_quat is None:
            r_hip, l_hip, r_sh, l_sh = self.face_indices
            across = _normalize((joints[:, r_hip] - joints[:, l_hip]) +
                                (joints[:, r_sh] - joints[:, l_sh]))
            up = np.broadcast_to(self.up_axis[None, :], across.shape)
            forward = _normalize(np.cross(up, across, axis=-1))
            if self.smooth_forward_sigma > 0:
                try:
                    import scipy.ndimage as ndi
                    forward = _normalize(ndi.gaussian_filter1d(
                        forward, self.smooth_forward_sigma, axis=0, mode="nearest"))
                except Exception:
                    pass
            tgt = np.broadcast_to(self.target_forward[None, :], forward.shape)
            root_quat = qbetween_np(forward, tgt).astype(np.float32)
            if self.first_frame_root_identity:
                root_quat[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            root_quat = np.asarray(root_quat, dtype=np.float32)
            assert root_quat.shape == (T, 4), f"root_quat should be (T,4), got {root_quat.shape}"

        # Template unit directions u_j from offsets (I have no idea what this means)
        dirs = self.offsets.detach().cpu().numpy().astype(np.float32)
        nrm  = np.linalg.norm(dirs, axis=-1, keepdims=True)
        u_dirs = np.where(nrm > 0, dirs / np.maximum(nrm, 1e-8), 0.0)

        # One-pass parent-first IK
        quat_local = np.zeros((T, J, 4), dtype=np.float32)
        quat_world = np.zeros((T, J, 4), dtype=np.float32)
        quat_local[:, self.root] = root_quat
        quat_world[:, self.root] = root_quat

        for j in self.order[1:]:
            p = self.parents[j]
            v = _normalize(joints[:, j] - joints[:, p])
            u = np.broadcast_to(u_dirs[j], (T, 3))
            Rwv = qbetween_np(u, v)
            Rpl = qinv_np(quat_world[:, p])
            Rlj = qmul_np(Rpl, Rwv)
            quat_local[:, j] = Rlj
            quat_world[:, j] = qmul_np(quat_world[:, p], Rlj)
        return quat_local

#     # ---------- basic accessors ----------

#     def njoints(self) -> int:
#         return self._J

#     def parents(self) -> List[int]:
#         return self._parents

#     def template_dirs(self) -> torch.Tensor:
#         """(J,3) unit directions used as canonical local axes for IK/FK."""
#         return self._template_dirs

#     def offsets(self) -> torch.Tensor:
#         """(J,3) current FK offsets (length * template_dir)."""
#         return self._offsets

#     def set_offsets(self, offsets: torch.Tensor) -> None:
#         """Directly set FK offsets (root will be forced to zero)."""
#         assert offsets.shape == (self._J, 3)
#         off = offsets.clone().detach().to(self.device).float()
#         off[0] = 0.0
#         self._offsets = off

#     # ---------- deriving offsets (lengths) from joints ----------

#     def set_offsets_from_joints(self, joints: torch.Tensor) -> None:
#         """
#         Derive bone lengths from a rest pose `joints` (J,3), but keep canonical directions.
#         offset[j] = ||joints[j] - joints[parent[j]]|| * template_dir[j]
#         """
#         assert joints.shape == (self._J, 3)
#         joints = joints.to(self.device).float()
#         lengths = torch.zeros((self._J, 1), device=self.device, dtype=torch.float32)
#         for j in range(1, self._J):
#             p = self._parents[j]
#             v = joints[j] - joints[p]
#             lengths[j, 0] = torch.linalg.norm(v)
#         self._offsets = self._template_dirs * lengths  # (J,3)
#         self._offsets[0] = 0.0

#     # ---------- IK: compute local rotations per joint (NP) ----------

#     @staticmethod
#     def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
#         n = np.linalg.norm(v, axis=-1, keepdims=True)
#         return v / np.maximum(n, eps)

#     @staticmethod
#     def _normalize_dirs(off: np.ndarray) -> np.ndarray:
#         """Make per-joint unit directions from offsets, keep root zero."""
#         dirs = off.copy()
#         dirs[0] = 0.0
#         for j in range(1, off.shape[0]):
#             n = np.linalg.norm(off[j])
#             if n > 0:
#                 dirs[j] = off[j] / n
#             else:
#                 # Fallback: if zero, leave as zero; caller should ensure valid dirs
#                 dirs[j] = off[j]
#         return dirs.astype(np.float32)

#     def inverse_kinematics_np(
#         self,
#         joints: np.ndarray,                    # (T,J,3) world joints
#         face_indices: Tuple[int, int, int, int],  # (r_hip, l_hip, r_shoulder, l_shoulder)
#         smooth_forward_sigma: Optional[float] = None,
#         up_axis: np.ndarray = np.array([0, 1, 0], dtype=np.float32),
#         target_forward: np.ndarray = np.array([0, 0, 1], dtype=np.float32),
#         first_frame_root_identity: bool = True,
#     ) -> np.ndarray:
#         """
#         Returns local quaternions (T,J,4).
#         Root quat aligns 'forward' toward `target_forward` via yaw-only heuristic:
#           forward = normalize(cross(up_axis, across)), across = (rhip-lhip) + (rshoulder-lshoulder)
#         Then for each joint we solve R_loc s.t. R_accum * R_loc maps template_dir[j] -> measured bone direction.
#         """
#         T = joints.shape[0]
#         assert joints.shape[1] == self._J and joints.shape[2] == 3

#         r_hip, l_hip, sdr_r, sdr_l = face_indices  # **order clarified**
#         across = (joints[:, r_hip] - joints[:, l_hip]) + (joints[:, sdr_r] - joints[:, sdr_l])
#         across = self._normalize(across)

#         up = np.broadcast_to(up_axis[None, :], across.shape)
#         forward = np.cross(up, across, axis=-1)               # yaw-only “forward”
#         if smooth_forward_sigma and smooth_forward_sigma > 0:
#             import scipy.ndimage
#             forward = scipy.ndimage.gaussian_filter1d(forward, smooth_forward_sigma, axis=0, mode="nearest")
#         forward = self._normalize(forward)

#         tgt = np.broadcast_to(target_forward[None, :], forward.shape)
#         root_quat = qbetween_np(forward, tgt)                 # (T,4)

#         if first_frame_root_identity:
#             root_quat[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

#         quat_params = np.zeros((T, self._J, 4), dtype=np.float32)
#         quat_params[:, 0] = root_quat

#         # prefetch template unit directions as numpy
#         u_dirs = self._template_dirs_np  # (J,3), unit; root is zero

#         for chain in self._chains_from_parents():
#             R_acc = root_quat  # (T,4)
#             for idx in range(1, len(chain)):
#                 j = chain[idx]
#                 p = chain[idx - 1]

#                 u = np.broadcast_to(u_dirs[j], (T, 3))
#                 v = joints[:, j] - joints[:, p]
#                 v = self._normalize(v)

#                 rot_u_to_v = qbetween_np(u, v)               # (T,4)
#                 R_loc = qmul_np(qinv_np(R_acc), rot_u_to_v)   # local rotation at j

#                 quat_params[:, j] = R_loc
#                 R_acc = qmul_np(R_acc, R_loc)

#         return quat_params  # (T,J,4)

#     # ---------- FK (Quat / 6D) ----------

#     def _ensure_offsets(self) -> None:
#         if self._offsets is None:
#             raise RuntimeError("Offsets are not set. Call set_offsets(...) or set_offsets_from_joints(...).")

#     def _chains_from_parents(self) -> List[List[int]]:
#         """Build root→leaf chains from parents for readable loops."""
#         children = [[] for _ in range(self._J)]
#         for j in range(1, self._J):
#             children[self._parents[j]].append(j)
#         # DFS from root
#         chains = []
#         stack = [[0]]
#         while stack:
#             path = stack.pop()
#             last = path[-1]
#             if not children[last]:
#                 chains.append(path)
#             else:
#                 for c in children[last]:
#                     stack.append(path + [c])
#         return chains

#     def forward_kinematics_np(
#         self,
#         quat_params: np.ndarray,               # (T,J,4)
#         root_pos: np.ndarray,                  # (T,3)
#         do_root_R: bool = True,
#     ) -> np.ndarray:
#         """FK with quaternions (numpy)."""
#         self._ensure_offsets()
#         T = quat_params.shape[0]
#         joints = np.zeros((T, self._J, 3), dtype=np.float32)
#         joints[:, 0] = root_pos

#         # cache offsets (constant across T)
#         off = self._offsets.detach().cpu().numpy().astype(np.float32)

#         for chain in self._chains_from_parents():
#             R_acc = quat_params[:, 0] if do_root_R else np.array([1, 0, 0, 0], dtype=np.float32)[None, :].repeat(T, 0)
#             for idx in range(1, len(chain)):
#                 j = chain[idx]
#                 R_acc = qmul_np(R_acc, quat_params[:, j])
#                 joints[:, j] = qrot_np(R_acc, off[j]) + joints[:, chain[idx - 1]]
#         return joints

#     def forward_kinematics_cont6d_np(
#         self,
#         cont6d_params: np.ndarray,             # (T,J,6)
#         root_pos: np.ndarray,                  # (T,3)
#         do_root_R: bool = True,
#     ) -> np.ndarray:
#         """FK with 6D rotations (numpy)."""
#         self._ensure_offsets()
#         T = cont6d_params.shape[0]
#         joints = np.zeros((T, self._J, 3), dtype=np.float32)
#         joints[:, 0] = root_pos

#         off = self._offsets.detach().cpu().numpy().astype(np.float32)
#         if do_root_R:
#             R_acc = cont6d_to_matrix_np(cont6d_params[:, 0])          # (T,3,3)
#         else:
#             R_acc = np.eye(3, dtype=np.float32)[None, ...].repeat(T, axis=0)

#         for chain in self._chains_from_parents():
#             R = R_acc.copy()
#             for idx in range(1, len(chain)):
#                 j = chain[idx]
#                 R = np.matmul(R, cont6d_to_matrix_np(cont6d_params[:, j]))
#                 joints[:, j] = (R @ off[j][..., None]).squeeze(-1) + joints[:, chain[idx - 1]]
#         return joints

#     # ---------- Torch FK (6D) for training loops ----------

#     def forward_kinematics_cont6d(
#         self,
#         cont6d_params: torch.Tensor,           # (T,J,6) or (B,J,6)
#         root_pos: torch.Tensor,                # (T,3) or (B,3)
#         do_root_R: bool = True,
#     ) -> torch.Tensor:
#         """FK with 6D rotations (torch). Batch/time-first agnostic on leading dim."""
#         self._ensure_offsets()
#         off = self._offsets  # (J,3)
#         B = cont6d_params.shape[0]
#         joints = torch.zeros((B, self._J, 3), device=cont6d_params.device, dtype=cont6d_params.dtype)
#         joints[:, 0, :] = root_pos

#         if do_root_R:
#             R_acc = cont6d_to_matrix(cont6d_params[:, 0])             # (B,3,3)
#         else:
#             R_acc = torch.eye(3, device=cont6d_params.device).expand(B, 3, 3)

#         for chain in self._chains_from_parents():
#             R = R_acc.clone()
#             for idx in range(1, len(chain)):
#                 j = chain[idx]
#                 R = R @ cont6d_to_matrix(cont6d_params[:, j])         # (B,3,3)
#                 joints[:, j] = (R @ off[j].unsqueeze(-1)).squeeze(-1) + joints[:, chain[idx - 1]]
#         return joints


# class Skeleton(object):
#     def __init__(self, offset, kinematic_tree, device):
#         self.device = device
#         self._raw_offset_np = offset.numpy()
#         self._raw_offset = offset.clone().detach().to(device).float()
#         self._kinematic_tree = kinematic_tree
#         self._offset = None
#         self._parents = [0] * len(self._raw_offset)
#         self._parents[0] = -1
#         for chain in self._kinematic_tree:
#             for j in range(1, len(chain)):
#                 self._parents[chain[j]] = chain[j-1]

#     def njoints(self):
#         return len(self._raw_offset)

#     def offset(self):
#         return self._offset

#     def set_offset(self, offsets):
#         self._offset = offsets.clone().detach().to(self.device).float()

#     def kinematic_tree(self):
#         return self._kinematic_tree

#     def parents(self):
#         return self._parents

#     # joints (batch_size, joints_num, 3)
#     def get_offsets_joints_batch(self, joints):
#         assert len(joints.shape) == 3
#         _offsets = self._raw_offset.expand(joints.shape[0], -1, -1).clone()
#         for i in range(1, self._raw_offset.shape[0]):
#             _offsets[:, i] = torch.norm(joints[:, i] - joints[:, self._parents[i]], p=2, dim=1)[:, None] * _offsets[:, i]

#         self._offset = _offsets.detach()
#         return _offsets

#     # joints (joints_num, 3)
#     def get_offsets_joints(self, joints):
#         assert len(joints.shape) == 2
#         _offsets = self._raw_offset.clone()
#         for i in range(1, self._raw_offset.shape[0]):
#             # print(joints.shape)
#             _offsets[i] = torch.norm(joints[i] - joints[self._parents[i]], p=2, dim=0) * _offsets[i]

#         self._offset = _offsets.detach()
#         return _offsets

#     # face_joint_idx should follow the order of right hip, left hip, right shoulder, left shoulder
#     # joints (batch_size, joints_num, 3)
#     def inverse_kinematics_np(self, joints, face_joint_idx, smooth_forward=False):
#         assert len(face_joint_idx) == 4
#         '''Get Forward Direction'''
#         l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
#         across1 = joints[:, r_hip] - joints[:, l_hip]
#         across2 = joints[:, sdr_r] - joints[:, sdr_l]
#         across = across1 + across2
#         across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]
#         # print(across1.shape, across2.shape)

#         # forward (batch_size, 3)
#         forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
#         if smooth_forward:
#             forward = filters.gaussian_filter1d(forward, 20, axis=0, mode='nearest')
#             # forward (batch_size, 3)
#         forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

#         '''Get Root Rotation'''
#         target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
#         root_quat = qbetween_np(forward, target)

#         '''Inverse Kinematics'''
#         # quat_params (batch_size, joints_num, 4)
#         # print(joints.shape[:-1])
#         quat_params = np.zeros(joints.shape[:-1] + (4,))
#         # print(quat_params.shape)
#         root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
#         quat_params[:, 0] = root_quat
#         # quat_params[0, 0] = np.array([[1.0, 0.0, 0.0, 0.0]])
#         for chain in self._kinematic_tree:
#             R = root_quat
#             for j in range(len(chain) - 1):
#                 # (batch, 3)
#                 u = self._raw_offset_np[chain[j+1]][np.newaxis,...].repeat(len(joints), axis=0)
#                 # print(u.shape)
#                 # (batch, 3)
#                 v = joints[:, chain[j+1]] - joints[:, chain[j]]
#                 v = v / np.sqrt((v**2).sum(axis=-1))[:, np.newaxis]
#                 # print(u.shape, v.shape)
#                 rot_u_v = qbetween_np(u, v)

#                 R_loc = qmul_np(qinv_np(R), rot_u_v)

#                 quat_params[:,chain[j + 1], :] = R_loc
#                 R = qmul_np(R, R_loc)

#         return quat_params

#     # Be sure root joint is at the beginning of kinematic chains
#     def forward_kinematics(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
#         # quat_params (batch_size, joints_num, 4)
#         # joints (batch_size, joints_num, 3)
#         # root_pos (batch_size, 3)
#         if skel_joints is not None:
#             offsets = self.get_offsets_joints_batch(skel_joints)
#         if len(self._offset.shape) == 2:
#             offsets = self._offset.expand(quat_params.shape[0], -1, -1)
#         joints = torch.zeros(quat_params.shape[:-1] + (3,)).to(self.device)
#         joints[:, 0] = root_pos
#         for chain in self._kinematic_tree:
#             if do_root_R:
#                 R = quat_params[:, 0]
#             else:
#                 R = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(len(quat_params), -1).detach().to(self.device)
#             for i in range(1, len(chain)):
#                 R = qmul(R, quat_params[:, chain[i]])
#                 offset_vec = offsets[:, chain[i]]
#                 joints[:, chain[i]] = qrot(R, offset_vec) + joints[:, chain[i-1]]
#         return joints

#     # Be sure root joint is at the beginning of kinematic chains
#     def forward_kinematics_np(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
#         # quat_params (batch_size, joints_num, 4)
#         # joints (batch_size, joints_num, 3)
#         # root_pos (batch_size, 3)
#         if skel_joints is not None:
#             skel_joints = torch.from_numpy(skel_joints)
#             offsets = self.get_offsets_joints_batch(skel_joints)
#         if len(self._offset.shape) == 2:
#             offsets = self._offset.expand(quat_params.shape[0], -1, -1)
#         offsets = offsets.numpy()
#         joints = np.zeros(quat_params.shape[:-1] + (3,))
#         joints[:, 0] = root_pos
#         for chain in self._kinematic_tree:
#             if do_root_R:
#                 R = quat_params[:, 0]
#             else:
#                 R = np.array([[1.0, 0.0, 0.0, 0.0]]).repeat(len(quat_params), axis=0)
#             for i in range(1, len(chain)):
#                 R = qmul_np(R, quat_params[:, chain[i]])
#                 offset_vec = offsets[:, chain[i]]
#                 joints[:, chain[i]] = qrot_np(R, offset_vec) + joints[:, chain[i - 1]]
#         return joints

#     def forward_kinematics_cont6d_np(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
#         # cont6d_params (batch_size, joints_num, 6)
#         # joints (batch_size, joints_num, 3)
#         # root_pos (batch_size, 3)
#         if skel_joints is not None:
#             skel_joints = torch.from_numpy(skel_joints)
#             offsets = self.get_offsets_joints_batch(skel_joints)
#         if len(self._offset.shape) == 2:
#             offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
#         offsets = offsets.numpy()
#         joints = np.zeros(cont6d_params.shape[:-1] + (3,))
#         joints[:, 0] = root_pos
#         for chain in self._kinematic_tree:
#             if do_root_R:
#                 matR = cont6d_to_matrix_np(cont6d_params[:, 0])
#             else:
#                 matR = np.eye(3)[np.newaxis, :].repeat(len(cont6d_params), axis=0)
#             for i in range(1, len(chain)):
#                 matR = np.matmul(matR, cont6d_to_matrix_np(cont6d_params[:, chain[i]]))
#                 offset_vec = offsets[:, chain[i]][..., np.newaxis]
#                 # print(matR.shape, offset_vec.shape)
#                 joints[:, chain[i]] = np.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i-1]]
#         return joints

#     def forward_kinematics_cont6d(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
#         # cont6d_params (batch_size, joints_num, 6)
#         # joints (batch_size, joints_num, 3)
#         # root_pos (batch_size, 3)
#         if skel_joints is not None:
#             # skel_joints = torch.from_numpy(skel_joints)
#             offsets = self.get_offsets_joints_batch(skel_joints)
#         if len(self._offset.shape) == 2:
#             offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
#         joints = torch.zeros(cont6d_params.shape[:-1] + (3,)).to(cont6d_params.device)
#         joints[..., 0, :] = root_pos
#         for chain in self._kinematic_tree:
#             if do_root_R:
#                 matR = cont6d_to_matrix(cont6d_params[:, 0])
#             else:
#                 matR = torch.eye(3).expand((len(cont6d_params), -1, -1)).detach().to(cont6d_params.device)
#             for i in range(1, len(chain)):
#                 matR = torch.matmul(matR, cont6d_to_matrix(cont6d_params[:, chain[i]]))
#                 offset_vec = offsets[:, chain[i]].unsqueeze(-1)
#                 # print(matR.shape, offset_vec.shape)
#                 joints[:, chain[i]] = torch.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i-1]]
#         return joints




# class Skeleton(object):
#     def __init__(self, offset, kinematic_tree, device):
#         self.device = device
#         self._raw_offset_np = offset.numpy()
#         self._raw_offset = offset.clone().detach().to(device).float()
#         self._kinematic_tree = kinematic_tree
#         self._offset = None
#         self._parents = [0] * len(self._raw_offset)
#         self._parents[0] = -1
#         for chain in self._kinematic_tree:
#             for j in range(1, len(chain)):
#                 self._parents[chain[j]] = chain[j-1]

#     def njoints(self):
#         return len(self._raw_offset)

#     def offset(self):
#         return self._offset

#     def set_offset(self, offsets):
#         self._offset = offsets.clone().detach().to(self.device).float()

#     def kinematic_tree(self):
#         return self._kinematic_tree

#     def parents(self):
#         return self._parents

#     # joints (batch_size, joints_num, 3)
#     def get_offsets_joints_batch(self, joints):
#         assert len(joints.shape) == 3
#         _offsets = self._raw_offset.expand(joints.shape[0], -1, -1).clone()
#         for i in range(1, self._raw_offset.shape[0]):
#             _offsets[:, i] = torch.norm(joints[:, i] - joints[:, self._parents[i]], p=2, dim=1)[:, None] * _offsets[:, i]

#         self._offset = _offsets.detach()
#         return _offsets

#     # joints (joints_num, 3)
#     def get_offsets_joints(self, joints):
#         assert len(joints.shape) == 2
#         _offsets = self._raw_offset.clone()
#         for i in range(1, self._raw_offset.shape[0]):
#             # print(joints.shape)
#             _offsets[i] = torch.norm(joints[i] - joints[self._parents[i]], p=2, dim=0) * _offsets[i]

#         self._offset = _offsets.detach()
#         return _offsets

#     # face_joint_idx should follow the order of right hip, left hip, right shoulder, left shoulder
#     # joints (batch_size, joints_num, 3)
#     def inverse_kinematics_np(self, joints, face_joint_idx, smooth_forward=False):
#         assert len(face_joint_idx) == 4
#         '''Get Forward Direction'''
#         l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
#         across1 = joints[:, r_hip] - joints[:, l_hip]
#         across2 = joints[:, sdr_r] - joints[:, sdr_l]
#         across = across1 + across2
#         across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]
#         # print(across1.shape, across2.shape)

#         # forward (batch_size, 3)
#         forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
#         if smooth_forward:
#             forward = filters.gaussian_filter1d(forward, 20, axis=0, mode='nearest')
#             # forward (batch_size, 3)
#         forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

#         '''Get Root Rotation'''
#         target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
#         root_quat = qbetween_np(forward, target)

#         '''Inverse Kinematics'''
#         # quat_params (batch_size, joints_num, 4)
#         # print(joints.shape[:-1])
#         quat_params = np.zeros(joints.shape[:-1] + (4,))
#         # print(quat_params.shape)
#         root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
#         quat_params[:, 0] = root_quat
#         # quat_params[0, 0] = np.array([[1.0, 0.0, 0.0, 0.0]])
#         for chain in self._kinematic_tree:
#             R = root_quat
#             for j in range(len(chain) - 1):
#                 # (batch, 3)
#                 u = self._raw_offset_np[chain[j+1]][np.newaxis,...].repeat(len(joints), axis=0)
#                 # print(u.shape)
#                 # (batch, 3)
#                 v = joints[:, chain[j+1]] - joints[:, chain[j]]
#                 v = v / np.sqrt((v**2).sum(axis=-1))[:, np.newaxis]
#                 # print(u.shape, v.shape)
#                 rot_u_v = qbetween_np(u, v)

#                 R_loc = qmul_np(qinv_np(R), rot_u_v)

#                 quat_params[:,chain[j + 1], :] = R_loc
#                 R = qmul_np(R, R_loc)

#         return quat_params

#     # Be sure root joint is at the beginning of kinematic chains
#     def forward_kinematics(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
#         # quat_params (batch_size, joints_num, 4)
#         # joints (batch_size, joints_num, 3)
#         # root_pos (batch_size, 3)
#         if skel_joints is not None:
#             offsets = self.get_offsets_joints_batch(skel_joints)
#         if len(self._offset.shape) == 2:
#             offsets = self._offset.expand(quat_params.shape[0], -1, -1)
#         joints = torch.zeros(quat_params.shape[:-1] + (3,)).to(self.device)
#         joints[:, 0] = root_pos
#         for chain in self._kinematic_tree:
#             if do_root_R:
#                 R = quat_params[:, 0]
#             else:
#                 R = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(len(quat_params), -1).detach().to(self.device)
#             for i in range(1, len(chain)):
#                 R = qmul(R, quat_params[:, chain[i]])
#                 offset_vec = offsets[:, chain[i]]
#                 joints[:, chain[i]] = qrot(R, offset_vec) + joints[:, chain[i-1]]
#         return joints

#     # Be sure root joint is at the beginning of kinematic chains
#     def forward_kinematics_np(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
#         # quat_params (batch_size, joints_num, 4)
#         # joints (batch_size, joints_num, 3)
#         # root_pos (batch_size, 3)
#         if skel_joints is not None:
#             skel_joints = torch.from_numpy(skel_joints)
#             offsets = self.get_offsets_joints_batch(skel_joints)
#         if len(self._offset.shape) == 2:
#             offsets = self._offset.expand(quat_params.shape[0], -1, -1)
#         offsets = offsets.numpy()
#         joints = np.zeros(quat_params.shape[:-1] + (3,))
#         joints[:, 0] = root_pos
#         for chain in self._kinematic_tree:
#             if do_root_R:
#                 R = quat_params[:, 0]
#             else:
#                 R = np.array([[1.0, 0.0, 0.0, 0.0]]).repeat(len(quat_params), axis=0)
#             for i in range(1, len(chain)):
#                 R = qmul_np(R, quat_params[:, chain[i]])
#                 offset_vec = offsets[:, chain[i]]
#                 joints[:, chain[i]] = qrot_np(R, offset_vec) + joints[:, chain[i - 1]]
#         return joints

#     def forward_kinematics_cont6d_np(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
#         # cont6d_params (batch_size, joints_num, 6)
#         # joints (batch_size, joints_num, 3)
#         # root_pos (batch_size, 3)
#         if skel_joints is not None:
#             skel_joints = torch.from_numpy(skel_joints)
#             offsets = self.get_offsets_joints_batch(skel_joints)
#         if len(self._offset.shape) == 2:
#             offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
#         offsets = offsets.numpy()
#         joints = np.zeros(cont6d_params.shape[:-1] + (3,))
#         joints[:, 0] = root_pos
#         for chain in self._kinematic_tree:
#             if do_root_R:
#                 matR = cont6d_to_matrix_np(cont6d_params[:, 0])
#             else:
#                 matR = np.eye(3)[np.newaxis, :].repeat(len(cont6d_params), axis=0)
#             for i in range(1, len(chain)):
#                 matR = np.matmul(matR, cont6d_to_matrix_np(cont6d_params[:, chain[i]]))
#                 offset_vec = offsets[:, chain[i]][..., np.newaxis]
#                 # print(matR.shape, offset_vec.shape)
#                 joints[:, chain[i]] = np.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i-1]]
#         return joints

#     def forward_kinematics_cont6d(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
#         # cont6d_params (batch_size, joints_num, 6)
#         # joints (batch_size, joints_num, 3)
#         # root_pos (batch_size, 3)
#         if skel_joints is not None:
#             # skel_joints = torch.from_numpy(skel_joints)
#             offsets = self.get_offsets_joints_batch(skel_joints)
#         if len(self._offset.shape) == 2:
#             offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
#         joints = torch.zeros(cont6d_params.shape[:-1] + (3,)).to(cont6d_params.device)
#         joints[..., 0, :] = root_pos
#         for chain in self._kinematic_tree:
#             if do_root_R:
#                 matR = cont6d_to_matrix(cont6d_params[:, 0])
#             else:
#                 matR = torch.eye(3).expand((len(cont6d_params), -1, -1)).detach().to(cont6d_params.device)
#             for i in range(1, len(chain)):
#                 matR = torch.matmul(matR, cont6d_to_matrix(cont6d_params[:, chain[i]]))
#                 offset_vec = offsets[:, chain[i]].unsqueeze(-1)
#                 # print(matR.shape, offset_vec.shape)
#                 joints[:, chain[i]] = torch.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i-1]]
#         return joints