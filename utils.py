import colorsys
import numpy as np
import torch

EPS = 1e-4


def gen_colors(N):
    """
    Generate N colors as far from each other as possible
    """
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    RGB_tuples = [[int(x[0]*255), int(x[1]*255), int(x[2]*255)] for x in RGB_tuples]
    return RGB_tuples


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    # assert q.shape[-1] == 4
    # assert v.shape[-1] == 3
    # assert q.shape[:-1] == v.shape[:-1]
    #
    original_shape = list(v.shape)
    # q = q.view(-1, 4)
    # v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def depth2cloud(depth, cc2world, image_size=64, sil=None, data_type='VON'):
    if sil is not None:
        fg_mask = sil > 0.5
    else:
        fg_mask = depth > -1. + EPS

    if data_type == 'ortho':
        uv = fg_mask.nonzero()
        Y = (uv[:, 1] - (image_size / 2)) / image_size
        Z = ((image_size / 2) - uv[:, 0]) / image_size
        P_C_object = torch.stack([depth[fg_mask], Z, Y])

    else:
        X = 2.5-depth
        uv = fg_mask.nonzero()
        distance_factor = 2.5 if data_type == 'blender' else 1.5
        Y = (uv[:, 1] - (image_size / 2)) * X[fg_mask] / ((image_size / 2) * distance_factor)
        Z = ((image_size / 2) - uv[:, 0]) * X[fg_mask] / ((image_size / 2) * distance_factor)
        X = 2.5-X if data_type == 'blender' else X-2.5
        Y = torch.clamp(Y, -1, 1)
        Z = torch.clamp(Z, -1, 1)
        P_C_object = torch.stack([X[fg_mask], Y, Z])
        permutation = torch.LongTensor([0, 2, 1]) if data_type == 'blender' else torch.LongTensor([0, 1, 2])
        P_C_object = P_C_object[permutation]

    rotated_P_C = qrot(cc2world.expand(P_C_object.size(1), -1), torch.transpose(P_C_object, 0, 1))
    P_C_world = rotated_P_C  # if data_type == 'blender' else torch.transpose(rotated_P_C, 0, 1)
    return P_C_world


def axis_angle2quat(axis, angle):
    return torch.from_numpy(
        np.array([np.cos(angle / 2),
                  axis[0] * np.sin(angle / 2),
                  axis[1] * np.sin(angle / 2),
                  axis[2] * np.sin(angle / 2)])).float()


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    # Compute outer product
    terms = torch.mm(r.view(4, -1), q.view(-1, 4))

    w = terms[0, 0] - terms[1, 1] - terms[2, 2] - terms[3, 3]
    x = terms[0, 1] + terms[1, 0] - terms[2, 3] + terms[3, 2]
    y = terms[0, 2] + terms[1, 3] + terms[2, 0] - terms[3, 1]
    z = terms[0, 3] - terms[1, 2] + terms[2, 1] + terms[3, 0]

    return torch.from_numpy(np.array([w, x, y, z])).float()


def quaternion_matrix(camera_pos, data_type='VON'):
    az, el = camera_pos
    if data_type != 'VON':
        qel = axis_angle2quat((0, 0, 1), -el)
        qaz = axis_angle2quat((0, 1, 0), az)
        q2object = qmul(qel, qaz)
    else:
        q0 = axis_angle2quat((0, 0, 1), -np.pi / 2)
        qel = axis_angle2quat((1, 0, 0), el)
        qaz = axis_angle2quat((0, 0, 1), az)
        q2object = qmul(qmul(q0, qel), qaz)
    q2world = torch.stack([q2object[0], q2object[1] * -1, q2object[2] * -1, q2object[3] * -1])
    return q2object, q2world


def read_camera_positions(path, device=None, data_type='VON'):
    r"""
    Reads the elevation and azimuthal angles
    :param path:
    :return:
    """
    wc2cc = []       # world coordinates to camera coordinates
    cc2wc = []
    with open(path) as f:
        for vp, line in enumerate(f.readlines()):
            line = line.strip().split()
            az, el = line
            az, el = float(az), float(el)
            q2object, q2world = quaternion_matrix([az, el], data_type=data_type)
            cc2wc.append(q2world.to(device))
            wc2cc.append(q2object.to(device))

    return wc2cc, cc2wc
