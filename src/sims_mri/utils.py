import numpy as np

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


class EarlyStopping:
    """Early stopping to stop training when loss doesn't improve after patience epochs."""

    def __init__(self, patience=10, delta_fraction=0.0001, mode='min', verbose=False):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
            delta_fraction (float): Relative improvement threshold applied to best_loss.
            mode (str): One of 'min' or 'max'. In 'min' mode, training will stop when the
                        monitored quantity stops decreasing.
            verbose (bool): If True, prints a message for each improvement.
        """
        self.patience = patience
        self.delta_fraction = delta_fraction
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.best_loss = None
        self.improved = False  # True if last __call__ resulted in an improvement
        self.best_epoch = None  # Epoch number when best_loss was recorded
        self._current_epoch = -1  # Internal counter for epoch tracking

    def __call__(self, current_loss):
        """
        Check if early stopping criteria is met.

        Args:
            current_loss (float): Current epoch's loss.

        Returns:
            bool: True if early stopping criteria is met.
        """
        self._current_epoch += 1
        self.improved = False  # Reset for this call

        if self.best_loss is None:
            self.best_loss = current_loss
            self.best_epoch = self._current_epoch
            self.improved = True  # First call is always an "improvement"
        else:
            # Check if current loss is better than best loss
            if self.mode == 'min':
                # For minimization: improvement means current < best - delta
                is_improvement = current_loss < self.best_loss - self.best_loss * self.delta_fraction
            else:
                # For maximization: improvement means current > best + delta
                is_improvement = current_loss > self.best_loss + self.best_loss * self.delta_fraction

            if is_improvement:
                self.best_loss = current_loss
                self.best_epoch = self._current_epoch
                self.improved = True
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True

        return self.early_stop

    def reset(self):
        """Reset the early stopping counter and best loss."""
        self.counter = 0
        self.early_stop = False
        self.best_loss = None
        self.improved = False
        self.best_epoch = None
        self._current_epoch = -1
