# Pure-PyTorch reimplementation of the pytorch3d.transform functions used by
# EDGE (quaternion wxyz convention, 6D rotation = first two matrix columns).
# This lets the repo run on Windows without building pytorch3d.
import torch
import torch.nn.functional as F

__all__ = [
    "axis_angle_to_quaternion",
    "quaternion_to_axis_angle",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "quaternion_multiply",
    "quaternion_apply",
    "axis_angle_to_matrix",
    "matrix_to_axis_angle",
    "rotation_6d_to_matrix",
    "matrix_to_rotation_6d",
    "RotateAxisAngle",
]


def axis_angle_to_quaternion(axisangle):
    angle = torch.norm(axisangle, p=2, dim=-1, keepdim=True)
    axis = axisangle / torch.clamp(angle, min=1e-8)
    cos = torch.cos(angle / 2.0)
    sin = torch.sin(angle / 2.0)
    return torch.cat([cos, axis * sin], dim=-1)  # (w, x, y, z)


def quaternion_to_axis_angle(quaternion):
    norm = torch.norm(quaternion[..., 1:], p=2, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(norm, quaternion[..., :1])
    axis = quaternion[..., 1:] / torch.clamp(norm, min=1e-8)
    return axis * angle


def quaternion_to_matrix(quaternion):
    w, x, y, z = torch.unbind(quaternion, -1)
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    r00 = 1.0 - (tyy + tzz)
    r01 = txy - twz
    r02 = txz + twy
    r10 = txy + twz
    r11 = 1.0 - (txx + tzz)
    r12 = tyz - twx
    r20 = txz - twy
    r21 = tyz + twx
    r22 = 1.0 - (txx + tyy)
    return torch.stack(
        [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1
    ).reshape(quaternion.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix):
    """Branch-free Shepperd; matrix (..., 3, 3) -> quaternion (..., 4) wxyz."""
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(matrix.shape[:-2] + (9,)), dim=-1
    )
    kx = 1.0 + m00 - m11 - m22
    ky = 1.0 - m00 + m11 - m22
    kz = 1.0 - m00 - m11 + m22
    kw = 1.0 + m00 + m11 + m22

    which = torch.stack([kx, ky, kz, kw], dim=-1).argmax(dim=-1)

    def _sqrt(x):
        return torch.sqrt(torch.clamp(x, min=1e-12))

    # each case must yield (w, x, y, z) to match the wxyz convention used by the
    # rest of this file; the scalar is unsqueezed so it broadcasts over the last axis
    case_x = (0.5 / _sqrt(kx)).unsqueeze(-1) * torch.stack([m21 - m12, kx, m01 + m10, m02 + m20], dim=-1)
    case_y = (0.5 / _sqrt(ky)).unsqueeze(-1) * torch.stack([m02 - m20, m01 + m10, ky, m12 + m21], dim=-1)
    case_z = (0.5 / _sqrt(kz)).unsqueeze(-1) * torch.stack([m10 - m01, m02 + m20, m12 + m21, kz], dim=-1)
    case_w = (0.5 / _sqrt(kw)).unsqueeze(-1) * torch.stack([kw, m21 - m12, m02 - m20, m10 - m01], dim=-1)

    quat = torch.stack([case_x, case_y, case_z, case_w], dim=-2)
    quat = quat.gather(
        -2, which.reshape(which.shape + (1, 1)).expand(which.shape + (1, 4))
    ).squeeze(-2)
    return quat / torch.clamp(torch.norm(quat, dim=-1, keepdim=True), min=1e-12)


def quaternion_multiply(a, b):
    w1, x1, y1, z1 = torch.unbind(a, -1)
    w2, x2, y2, z2 = torch.unbind(b, -1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quaternion_apply(quaternion, point):
    qvec = quaternion[..., 1:]
    t = torch.linalg.cross(qvec, point, dim=-1) * 2.0
    return point + quaternion[..., :1] * t + torch.linalg.cross(qvec, t, dim=-1)


def axis_angle_to_matrix(axisangle):
    quat = axis_angle_to_quaternion(axisangle)
    return quaternion_to_matrix(quat)


def matrix_to_axis_angle(matrix):
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rotation_6d(matrix):
    return matrix[..., :2].transpose(-1, -2).reshape(*matrix.shape[:-2], 6)


class RotateAxisAngle:
    def __init__(self, angle, axis="X", degrees=True, **kwargs):
        if degrees:
            angle = angle * torch.pi / 180.0
        axis = axis.upper()
        axis_map = {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}
        device = torch.device(kwargs.get("device", "cpu"))
        axisangle = torch.tensor(axis_map[axis], device=device) * angle
        self._matrix = axis_angle_to_matrix(axisangle)

    def transform_points(self, points):
        m = self._matrix.to(points.device)
        return torch.einsum("...ij,...j->...i", m, points)
