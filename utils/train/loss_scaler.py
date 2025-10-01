from collections import defaultdict
import math
import torch

class LossScaler:
<<<<<<< HEAD
    def __init__(self, config):
        self.tau = config['tau']
        self.rms = {}

    @ torch.no_grad()
    def calibrate(self, raw_losses_list):
        sum_sq = defaultdict(float)
        n = 0
        for raw_losses in raw_losses_list:
            for name, raw in raw_losses.items():
                v = float(raw.detach())
                sum

    def accumulate_calibration(self, raw_losses):

    def end_calibration(self):

    def normalize(self, raw_losses):

# class LossScaler:
#     def __init__(self, loss_cfg: dict, alpha=0.01, eps=1e-8, warmup=100):
#         self.loss_cfg = loss_cfg or {}
#         g = self.loss_cfg.get("scaler", {})

#         # honor YAML global scaler; fall back to ctor args
#         self.alpha  = g.get("alpha", alpha)
#         self.eps    = g.get("eps",   eps)
#         self.warmup = g.get("warmup", warmup)

#         self.step = 0
#         # EMAs for stable normalization
#         self.ema_mean = defaultdict(lambda: 1.0)  # EMA(|raw|)
#         self.ema_sq   = defaultdict(lambda: 1.0)  # EMA(raw^2)

#         self.freeze_after_warmup = bool(self._global_scaler().get("freeze_after_warmup", False))
#         self.frozen_denom = {}           # name -> scalar
#         self._calibrating = False
#         self._cal_sum_abs, self._cal_sum_sq, self._cal_count = None, None, 0

#     # ---------- config lookups (compatible with your old usage) ----------

#     def begin_calibration(self):
#         from collections import defaultdict
#         self._calibrating = True
#         self._cal_sum_abs = defaultdict(float)
#         self._cal_sum_sq  = defaultdict(float)
#         self._cal_count   = 0

#     @torch.no_grad()
#     def accumulate_calibration(self, raw_losses: dict):
#         # raw_losses: {name: (anything, raw_tensor)}
#         for name, (_, raw) in raw_losses.items():
#             v = float(raw.detach())
#             self._cal_sum_abs[name] += abs(v)
#             self._cal_sum_sq[name]  += v * v
#         self._cal_count += 1

#     def end_calibration(self, use="rms", tau=1e-3):
#         assert self._calibrating, "begin_calibration() first"
#         for name in self._cal_sum_sq.keys():
#             if use == "mean":
#                 d = self._cal_sum_abs[name] / max(1, self._cal_count)
#             else:  # rms
#                 d = math.sqrt(self._cal_sum_sq[name] / max(1, self._cal_count))
#             self.frozen_denom[name] = max(d, tau)
#         self._calibrating = False
#         self._cal_sum_abs = self._cal_sum_sq = None
#         self._cal_count   = 0

#     def _per_loss_block(self, name: str) -> dict:
#         return self.loss_cfg.get(name, {}) if isinstance(self.loss_cfg.get(name, {}), dict) else {}

#     def _per_loss_scaler(self, name: str) -> dict:
#         blk = self._per_loss_block(name)
#         sc  = blk.get("scaler", {})
#         return sc if isinstance(sc, dict) else {}

#     def _global_scaler(self) -> dict:
#         sc = self.loss_cfg.get("scaler", {})
#         return sc if isinstance(sc, dict) else {}

#     def _weight(self, name: str) -> float:
#         # Keep your special mapping: ("style","content") -> "stylecon"
#         if name in self.loss_cfg and "weight" in self.loss_cfg[name]:
#             return float(self.loss_cfg[name]["weight"])
#         if name in ("style", "content") and "stylecon" in self.loss_cfg:
#             return float(self.loss_cfg["stylecon"].get("weight", 1.0))
#         # global default
#         return float(self._global_scaler().get("default_weight", 1.0))

#     def _cfg(self, name: str, key: str, default):
#         # per-loss scaler override -> global scaler -> default
#         per = self._per_loss_scaler(name)
#         if key in per: return per[key]
#         g = self._global_scaler()
#         if key in g: return g[key]
#         return default

#     # ---------- EMA updates / denominators ----------

#     @staticmethod
#     def _to_floats(x):
#         v = float(x.detach())
#         return abs(v), v

#     def _update_stats(self, name, raw):
#         a = self.alpha
#         av, v = self._to_floats(raw)
#         self.ema_mean[name] = (1 - a) * self.ema_mean[name] + a * av
#         self.ema_sq[name]   = (1 - a) * self.ema_sq[name]   + a * (v * v)

#     def _denom(self, name):
#         # Prefer frozen denominators after burn-in
#         if name in self.frozen_denom:
#             return self.frozen_denom[name]
#         # fallback to live EMA if you want (or return None to skip)
#         mode = self._cfg(name, "mode", "rms")
#         if mode == "off": return None
#         if mode == "mean": return self.ema_mean[name]
#         if mode == "rms":  return math.sqrt(self.ema_sq[name] + self.eps)
#         raise ValueError(f"Unknown scaler mode: {mode}")

#     # ---------- main API (keeps your signature/behavior) ----------

#     def normalize_and_weight(self, raw_losses: dict, train: bool):
#         """
#         raw_losses: {loss_name: (anything, raw_tensor)}
#         returns: (new_dict, total) where new_dict[name] = (scaled, raw)
#         """
#         use_norm = self.step >= self.warmup
#         new, total = {}, None

#         for name, (_, raw) in raw_losses.items():
#             if train:
#                 self._update_stats(name, raw)

#             w   = float(self._weight(name))
#             mode = self._cfg(name, "mode", "rms")
#             tau  = float(self._cfg(name, "tau", 1e-3))
#             cap  = self._cfg(name, "cap", None)
#             cap  = float(cap) if cap is not None else None

#             if (not use_norm) or mode == "off":
#                 scaled = w * raw
#             else:
#                 d = self._denom(name)
#                 d = max(d, tau)
#                 scale = w / d
#                 if cap is not None:
#                     scale = min(scale, w * cap)  # cap relative to w
#                 scaled = scale * raw

#             new[name] = (scaled, raw)
#             total = scaled if total is None else total + scaled

#         if train:
#             self.step += 1
#         return new, total

#     # ---------- checkpoints ----------

#     def state_dict(self):
#         return {
#             "ema_mean": dict(self.ema_mean),
#             "ema_sq":   dict(self.ema_sq),
#             "step":     self.step,
#             "alpha":    self.alpha,
#             "eps":      self.eps,
#             "warmup":   self.warmup,
#             "frozen_denom": dict(self.frozen_denom),
#         }

#     def load_state_dict(self, state):
#         self.ema_mean = defaultdict(lambda: 1.0, state.get("ema_mean", {}))
#         self.ema_sq   = defaultdict(lambda: 1.0, state.get("ema_sq", {}))
#         self.step     = state.get("step", 0)
#         self.alpha    = state.get("alpha", self.alpha)
#         self.eps      = state.get("eps",   self.eps)
#         self.warmup   = state.get("warmup", self.warmup)
#         self.frozen_denom = state.get("frozen_denom", {})
=======
    def __init__(self, loss_cfg: dict, alpha=0.01, eps=1e-8, warmup=100):
        self.loss_cfg = loss_cfg or {}
        g = self.loss_cfg.get("scaler", {})

        # honor YAML global scaler; fall back to ctor args
        self.alpha  = g.get("alpha", alpha)
        self.eps    = g.get("eps",   eps)
        self.warmup = g.get("warmup", warmup)

        self.step = 0
        # EMAs for stable normalization
        self.ema_mean = defaultdict(lambda: 1.0)  # EMA(|raw|)
        self.ema_sq   = defaultdict(lambda: 1.0)  # EMA(raw^2)

        self.freeze_after_warmup = bool(self._global_scaler().get("freeze_after_warmup", False))
        self.frozen_denom = {}           # name -> scalar
        self._calibrating = False
        self._cal_sum_abs, self._cal_sum_sq, self._cal_count = None, None, 0

    # ---------- config lookups (compatible with your old usage) ----------

    def begin_calibration(self):
        from collections import defaultdict
        self._calibrating = True
        self._cal_sum_abs = defaultdict(float)
        self._cal_sum_sq  = defaultdict(float)
        self._cal_count   = 0

    @torch.no_grad()
    def accumulate_calibration(self, raw_losses: dict):
        # raw_losses: {name: (anything, raw_tensor)}
        for name, (_, raw) in raw_losses.items():
            v = float(raw.detach())
            self._cal_sum_abs[name] += abs(v)
            self._cal_sum_sq[name]  += v * v
        self._cal_count += 1

    def end_calibration(self, use="rms", tau=1e-3):
        assert self._calibrating, "begin_calibration() first"
        for name in self._cal_sum_sq.keys():
            if use == "mean":
                d = self._cal_sum_abs[name] / max(1, self._cal_count)
            else:  # rms
                d = math.sqrt(self._cal_sum_sq[name] / max(1, self._cal_count))
            self.frozen_denom[name] = max(d, tau)
        self._calibrating = False
        self._cal_sum_abs = self._cal_sum_sq = None
        self._cal_count   = 0

    def _per_loss_block(self, name: str) -> dict:
        return self.loss_cfg.get(name, {}) if isinstance(self.loss_cfg.get(name, {}), dict) else {}

    def _per_loss_scaler(self, name: str) -> dict:
        blk = self._per_loss_block(name)
        sc  = blk.get("scaler", {})
        return sc if isinstance(sc, dict) else {}

    def _global_scaler(self) -> dict:
        sc = self.loss_cfg.get("scaler", {})
        return sc if isinstance(sc, dict) else {}

    def _weight(self, name: str) -> float:
        # Keep your special mapping: ("style","content") -> "stylecon"
        if name in self.loss_cfg and "weight" in self.loss_cfg[name]:
            return float(self.loss_cfg[name]["weight"])
        if name in ("style", "content") and "stylecon" in self.loss_cfg:
            return float(self.loss_cfg["stylecon"].get("weight", 1.0))
        # global default
        return float(self._global_scaler().get("default_weight", 1.0))

    def _cfg(self, name: str, key: str, default):
        # per-loss scaler override -> global scaler -> default
        per = self._per_loss_scaler(name)
        if key in per: return per[key]
        g = self._global_scaler()
        if key in g: return g[key]
        return default

    # ---------- EMA updates / denominators ----------

    @staticmethod
    def _to_floats(x):
        v = float(x.detach())
        return abs(v), v

    def _update_stats(self, name, raw):
        a = self.alpha
        av, v = self._to_floats(raw)
        self.ema_mean[name] = (1 - a) * self.ema_mean[name] + a * av
        self.ema_sq[name]   = (1 - a) * self.ema_sq[name]   + a * (v * v)

    def _denom(self, name):
        # Prefer frozen denominators after burn-in
        if name in self.frozen_denom:
            return self.frozen_denom[name]
        # fallback to live EMA if you want (or return None to skip)
        mode = self._cfg(name, "mode", "rms")
        if mode == "off": return None
        if mode == "mean": return self.ema_mean[name]
        if mode == "rms":  return math.sqrt(self.ema_sq[name] + self.eps)
        raise ValueError(f"Unknown scaler mode: {mode}")

    # ---------- main API (keeps your signature/behavior) ----------

    def normalize_and_weight(self, raw_losses: dict, train: bool):
        """
        raw_losses: {loss_name: (anything, raw_tensor)}
        returns: (new_dict, total) where new_dict[name] = (scaled, raw)
        """
        use_norm = self.step >= self.warmup
        new, total = {}, None

        for name, (_, raw) in raw_losses.items():
            if train:
                self._update_stats(name, raw)

            w   = float(self._weight(name))
            mode = self._cfg(name, "mode", "off")
            tau  = float(self._cfg(name, "tau", 1e-3))
            cap  = self._cfg(name, "cap", None)
            cap  = float(cap) if cap is not None else None

            if (not use_norm) or mode == "off":
                scaled = w * raw
            else:
                d = self._denom(name)
                d = max(d, tau)
                scale = w / d
                if cap is not None:
                    scale = min(scale, w * cap)  # cap relative to w
                scaled = scale * raw

            new[name] = (scaled, raw)
            total = scaled if total is None else total + scaled

        if train:
            self.step += 1
        return new, total

    # ---------- checkpoints ----------

    def state_dict(self):
        return {
            "ema_mean": dict(self.ema_mean),
            "ema_sq":   dict(self.ema_sq),
            "step":     self.step,
            "alpha":    self.alpha,
            "eps":      self.eps,
            "warmup":   self.warmup,
            "frozen_denom": dict(self.frozen_denom),
        }

    def load_state_dict(self, state):
        self.ema_mean = defaultdict(lambda: 1.0, state.get("ema_mean", {}))
        self.ema_sq   = defaultdict(lambda: 1.0, state.get("ema_sq", {}))
        self.step     = state.get("step", 0)
        self.alpha    = state.get("alpha", self.alpha)
        self.eps      = state.get("eps",   self.eps)
        self.warmup   = state.get("warmup", self.warmup)
        self.frozen_denom = state.get("frozen_denom", {})
>>>>>>> a259b41980b31d02bb0dee1bdc08f3f7b7844c9e

# Optional standalone helper if you still call get_weight elsewhere
# def get_weight(loss_name, loss_cfg):
#     if loss_name in loss_cfg and "weight" in loss_cfg[loss_name]:
#         return float(loss_cfg[loss_name]["weight"])
#     if loss_name in ("style", "content") and "stylecon" in loss_cfg:
#         return float(loss_cfg["stylecon"].get("weight", 1.0))
#     return float(loss_cfg.get("scaler", {}).get("default_weight", 1.0))