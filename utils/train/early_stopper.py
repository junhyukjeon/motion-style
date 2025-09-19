class EarlyStopper:
    def __init__(self, cfg):
        assert cfg['mode'] in ("min", "max")
        self.cfg = cfg
        self.best = float("inf") if cfg['mode'] == "min" else -float("inf")
        self.bad_epochs = 0
        self.best_epoch = -1

    def is_improvement(self, metric: float) -> bool:
        if self.cfg['mode'] == "min":
            return metric < (self.best - self.cfg['min_delta'])
        else:
            return metric > (self.best + self.cfg['min_delta'])

    def step(self, metric: float, epoch: int) -> bool:
        if self.is_improvement(metric):
            self.best = metric
            self.best_epoch = epoch
            self.bad_epochs = 0 
            return False  # do not stop
        else:
            self.bad_epochs += 1
            return self.bad_epochs > self.cfg['patience']
