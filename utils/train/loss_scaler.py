from collections import defaultdict
import torch


def get_weight(loss_name, loss_cfg):
    if loss_name in loss_cfg:
        return loss_cfg[loss_name]["weight"]
    if loss_name in ("style", "content") and "stylecon" in loss_cfg:
        return loss_cfg["stylecon"]["weight"]
    return 1.0


class LossScaler:
    def __init__(self, loss_cfg, alpha=0.01, eps=1e-8, warmup=100):
        self.loss_cfg = loss_cfg
        self.alpha = alpha
        self.eps = eps
        self.warmup = warmup
        self.step = 0
        self.running = defaultdict(lambda: 1.0)  # EMA of RAW values

    def normalize_and_weight(self, raw_losses: dict, train: bool):
        use_norm = self.step >= self.warmup
        new, total = {}, None

        for name, (_, raw) in raw_losses.items():
            if train:
                prev = self.running[name]
                rm = (1 - self.alpha) * prev + self.alpha * float(raw.detach())
                self.running[name] = rm
            else:
                rm = self.running[name]

            nf = rm + self.eps
            w = get_weight(name, self.loss_cfg)
            scaled = (w / nf) * raw if use_norm else w * raw

            new[name] = (scaled, raw)
            total = scaled if total is None else total + scaled

        if train:
            self.step += 1
        return new, total

    # For checkpoints
    def state_dict(self):
        return {"running": dict(self.running), "step": self.step}

    def load_state_dict(self, state):
        self.running = defaultdict(lambda: 1.0, state.get("running", {}))
        self.step = state.get("step", 0)