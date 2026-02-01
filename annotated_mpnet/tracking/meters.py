"""
Helper class borrowed from fairseq to help store values as we train across multiple steps and epochs
"""


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        """Initialize the meter and reset state."""
        self.reset()

    def reset(self) -> None:
        """Reset all stored values to 0."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update the meter with a new value.

        :param float val: Value to add.
        :param int n: Weight/count for the value, defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
