# Adapted from https://github.com/pytorch/ignite/blob/master/ignite/handlers/early_stopping.py

class EarlyStopping:
    """EarlyStopping handler can be used to stop the training if no improvement after a given number of events.
    Args:
        patience (int):
            Number of events to wait if no improvement and then stop the training.
        min_delta (float, optional):
            A minimum increase in the score to qualify as an improvement,
            i.e. an increase of less than or equal to `min_delta`, will count as no improvement.
        cumulative_delta (bool, optional):
            It True, `min_delta` defines an increase since the last `patience` reset, otherwise,
            it defines an increase after the last event. Default value is False.
    """

    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        cumulative_delta: bool = False,
        logger=None,
    ):
        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if min_delta < 0.0:
            raise ValueError("Argument min_delta should not be a negative number.")

        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.counter = 0
        self.best_score = None
        self.logger = logger

    def step(self, score) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta:
            if not self.cumulative_delta and score > self.best_score:
                self.best_score = score
            self.counter += 1
            if self.logger: self.logger.info("EarlyStopping: %i / %i" % (self.counter, self.patience))
            print("EarlyStopping: %i / %i" % (self.counter, self.patience))
            if self.counter >= self.patience:
                if self.logger: self.logger.info("EarlyStopping: Stop training")
                print("EarlyStopping: Stop training")
                return False
        else:
            self.best_score = score
            self.counter = 0
        return True