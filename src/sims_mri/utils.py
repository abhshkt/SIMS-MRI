import numpy as np
import torch
import torch.nn.functional as F

def get_rotation_matrix_numpy(theta_rot_x, theta_rot_y, theta_rot_z):
    x_rot_mat = np.array([[1, 0, 0, 0],
                       [0, np.cos(theta_rot_x), -np.sin(theta_rot_x), 0],
                       [0, np.sin(theta_rot_x), np.cos(theta_rot_x), 0],
                       [0, 0, 0, 1]])

    y_rot_mat = np.array([[np.cos(theta_rot_y), 0, np.sin(theta_rot_y), 0],
                       [0, 1, 0, 0],
                       [-np.sin(theta_rot_y), 0, np.cos(theta_rot_y), 0],
                       [0, 0, 0, 1]])

    z_rot_mat = np.array([[np.cos(theta_rot_z), -np.sin(theta_rot_z), 0, 0],
                       [np.sin(theta_rot_z), np.cos(theta_rot_z), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

    return x_rot_mat.dot(y_rot_mat).dot(z_rot_mat)

def get_rotation_matrix_torch(theta_rot_x, theta_rot_y, theta_rot_z, device=None):
    x_rot_mat = torch.tensor([[[1, 0, 0],
                       [0, torch.cos(theta_rot_x), -torch.sin(theta_rot_x)],
                       [0, torch.sin(theta_rot_x), torch.cos(theta_rot_x)],
                       ]])

    y_rot_mat = torch.tensor([[[torch.cos(theta_rot_y), 0, torch.sin(theta_rot_y)],
                       [0, 1, 0],
                       [-torch.sin(theta_rot_y), 0, torch.cos(theta_rot_y)],
                       ]])

    z_rot_mat = torch.tensor([[[torch.cos(theta_rot_z), -torch.sin(theta_rot_z), 0],
                       [torch.sin(theta_rot_z), torch.cos(theta_rot_z), 0],
                       [0, 0, 1],
                       ]])

    rotation_mat = torch.mm(torch.mm(x_rot_mat[0], y_rot_mat[0]), z_rot_mat[0])

    return rotation_mat


def denorm_grid(normalized_grid, x_min, x_max, s_min=-1, s_max=1):
    return (normalized_grid - s_min)/(s_max - s_min)*(x_max - x_min) + x_min


# From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/
def quaternion_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]

# From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/
def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)



# From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/
def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

# From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/
def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")



# from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)

# from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling


def count_parameters(model):
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EarlyStopping:
    """Early stopping to stop training when loss doesn't improve after patience epochs."""

    def __init__(self, patience=10, delta=0.0001, delta_fraction=0.0001, mode='min', verbose=False):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
            delta (float): Legacy absolute threshold retained for compatibility.
            delta_fraction (float): Relative improvement threshold applied to best_loss.
            mode (str): One of 'min' or 'max'. In 'min' mode, training will stop when the
                        monitored quantity stops decreasing.
            verbose (bool): If True, prints a message for each improvement.
        """
        self.patience = patience
        self.delta = delta
        self.delta_fraction = delta_fraction
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.best_loss = None
        self.old_best_loss = None
        self.improved = False  # True if last __call__ resulted in an improvement
        self.best_epoch = None  # Epoch number when best_loss was recorded
        self._current_epoch = -1  # Internal counter for epoch tracking
        
    def __call__(self, current_loss, model=None, path=None):
        """
        Check if early stopping criteria is met.

        Args:
            current_loss (float): Current epoch's loss.
            model (torch.nn.Module): Model to save if this is the best loss.
            path (str): Path to save the best model.

        Returns:
            bool: True if early stopping criteria is met.
        """
        self._current_epoch += 1
        self.improved = False  # Reset for this call

        if self.best_loss is None:
            self.old_best_loss = current_loss
            self.best_loss = current_loss
            self.best_epoch = self._current_epoch
            self.improved = True  # First call is always an "improvement"
            if model is not None and path is not None:
                self.save_checkpoint(current_loss, model, path)
        else:
            # Check if current loss is better than best loss
            if self.mode == 'min':
                # For minimization: improvement means current < best - delta
                is_improvement = current_loss < self.best_loss - self.best_loss * self.delta_fraction
            else:
                # For maximization: improvement means current > best + delta
                is_improvement = current_loss > self.best_loss + self.best_loss * self.delta_fraction

            if is_improvement:
                self.old_best_loss = self.best_loss
                self.best_loss = current_loss
                self.best_epoch = self._current_epoch
                self.improved = True
                if model is not None and path is not None:
                    self.save_checkpoint(current_loss, model, path)
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True

        return self.early_stop
    
    def save_checkpoint(self, loss, model, path):
        """Save model when loss improves."""
        if self.verbose:
            print(f'Loss improved ({self.old_best_loss:.6f} --> {loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)
        
    def reset(self):
        """Reset the early stopping counter and best loss."""
        self.counter = 0
        self.early_stop = False
        self.best_loss = None
        self.improved = False
        self.best_epoch = None
        self._current_epoch = -1


