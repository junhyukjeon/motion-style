import os
from os.path import join as pjoin

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import yaml
import time
import random
from tqdm import tqdm

from eval2.dataset import Text2MotionTestDataset #, build_dict_from_txt
from eval2.metrics import TM2TMetrics
from eval2.evaluator_wrapper import StyleClassification

from utils.motion import recover_from_ric
from model.t2sm import Text2StylizedMotion

from salad.models.t2m_eval_wrapper import build_evaluators
from salad.utils.word_vectorizer import WordVectorizer
from salad.utils.plot_script import plot_3d_motion_
from salad.utils.paramUtil import t2m_kinematic_chain

import warnings
from matplotlib import MatplotlibDeprecationWarning

def build_dict_from_txt(filename):
    result_dict = {}
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) >= 3:
                key = parts[2]
                value = parts[1].split("_")[0]
                result_dict[key] = value
                
    return result_dict

# padding to max length in one batch
def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def collate_fn(batch):
    notnone_batches = [b for b in batch if b is not None]
    batch_len = len(notnone_batches[0])
    notnone_batches.sort(key=lambda x: x[3], reverse=True)

    adapted_batch = {
        "motion":
        collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
        "text": [b[2] for b in notnone_batches],
        "length": [b[5] for b in notnone_batches],
        "word_embs":
        collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
        "pos_ohot":
        collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
        "text_len":
        collate_tensors([torch.tensor(b[3]) for b in notnone_batches]),
        "tokens": [b[6] for b in notnone_batches],

        "reference_motion": collate_tensors([torch.tensor(b[8]).float() for b in notnone_batches]),
        "text2": [b[7] for b in notnone_batches],
        "style_text": [b[11] for b in notnone_batches],
        "label": [torch.tensor(int(b[10])) for b in notnone_batches],
        "text_len2":collate_tensors([torch.tensor(b[9]) for b in notnone_batches]),
    }
    return adapted_batch

class SmoodiEval():
    def __init__(self, config, device="cuda:0"):

        self.device = device
        
        # model
        self.model = Text2StylizedMotion(config["model"]).to(device)
        self.model.load_state_dict(
            torch.load(pjoin(config["checkpoint_dir"], "best.ckpt"), map_location=device), strict=False
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # style classifier
        self.classifier = StyleClassification(nclasses=47).to(device)
        self.classifier.load_state_dict(
            torch.load("./checkpoints/style_classifier/style_classifier.pt", map_location=device)
        )
        self.label_to_motion = build_dict_from_txt("./dataset/100style/100STYLE_name_dict_Filter.txt")

        # metrics
        self.metrics = TM2TMetrics().to(device)

        # evaluation models
        self.text_enc, self.motion_enc, self.movement_enc = build_evaluators({
            "dataset_name": "t2m",
            "device": device,
            "dim_word": 300,
            "dim_pos_ohot": 15,
            "dim_motion_hidden": 1024,
            "dim_text_hidden": 512,
            "dim_coemb_hidden": 512,
            "dim_pose": 263,
            "dim_movement_enc_hidden": 512,
            "dim_movement_latent": 512,
            "checkpoints_dir": ".",
            "unit_length": 4,
        })
        self.text_enc = self.text_enc.to(device)
        self.motion_enc = self.motion_enc.to(device)
        self.movement_enc = self.movement_enc.to(device)
        self.text_enc.eval()
        self.motion_enc.eval()
        self.movement_enc.eval()
        for p in self.text_enc.parameters():
            p.requires_grad = False
        for p in self.motion_enc.parameters():
            p.requires_grad = False
        for p in self.movement_enc.parameters():
            p.requires_grad = False

        # dataset & dataloader
        data_root = "./dataset/humanml3d"
        mean = np.load(pjoin(data_root, "../100style/Mean.npy"))
        std = np.load(pjoin(data_root, "../100style/Std.npy"))
        mean_eval = np.load("./checkpoints/t2m/Comp_v6_KLD01/meta/mean.npy")
        std_eval = np.load("./checkpoints/t2m/Comp_v6_KLD01/meta/std.npy")
        w_vectorizer = WordVectorizer("./glove", "our_vab")
        dataset = Text2MotionTestDataset(
            mean=mean,
            std=std,
            mean_eval=mean_eval,
            std_eval=std_eval,
            split_file="./dataset/humanml3d/test.txt",
            w_vectorizer=w_vectorizer,
            max_motion_length=196,
            min_motion_length=40,
            max_text_len=20,
            unit_length=4,
            motion_dir1=pjoin(data_root, "new_joint_vecs"),
            text_dir1=pjoin(data_root, "texts"),
            motion_dir2=pjoin(data_root, "../100style/new_joint_vecs"),
            text_dir2=pjoin(data_root, "../100style/texts"),
        )
        # self.mean = torch.from_numpy(mean).float().to(device)
        # self.std = torch.from_numpy(std).float().to(device)
        # self.mean_eval = torch.from_numpy(np.load(pjoin(data_root, "../100style/Mean.npy"))).float().to(device)
        # self.std_eval = torch.from_numpy(np.load(pjoin(data_root, "../100style/Std.npy"))).float().to(device)
        self.data_loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

    def t2m_eval(self, batch):
        texts = batch["text"]
        motions = batch["motion"].detach().clone().cuda()
        lengths = torch.tensor(batch["length"]).detach().clone().cuda()
        
        reference_motion = batch["reference_motion"].detach().clone().cuda()
        label = torch.tensor(batch["label"]).detach().clone().cuda()

        word_embs = batch["word_embs"].detach().clone().cuda()
        pos_ohot = batch["pos_ohot"].detach().clone().cuda()
        text_lengths = batch["text_len"].detach().clone().cuda()

        ref_t2m = self.data_loader.dataset.renorm4t2m(reference_motion)
        start = time.time()
        feats_rst_t2m = self.model.generate2(ref_t2m, texts, motions.shape[1])[0]
        end = time.time()

        feats_rst = self.data_loader.dataset.renorm4style(feats_rst_t2m)
        logits = self.classifier(feats_rst)
        probs = F.softmax(logits, dim=-1)
        predicted = torch.argmax(probs, dim=1)

        motion_name = self.label_to_motion[str(predicted[0].cpu().numpy())]
        base_name = self.label_to_motion[str(label[0].cpu().numpy())]
        print(f"Name: {base_name} -> {motion_name}")


        # joints recover
        joints_rst = self.data_loader.dataset.feats2joints(feats_rst)
        joints_ref = self.data_loader.dataset.feats2joints(motions)
        joints_ref_m = self.data_loader.dataset.feats2joints(reference_motion)

        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
        #     plot_3d_motion_("rst.mp4", t2m_kinematic_chain, joints_rst[0].cpu().numpy(), "rst", fps=20)
        #     plot_3d_motion_("ref.mp4", t2m_kinematic_chain, joints_ref[0].cpu().numpy(), "ref", fps=20)
        #     plot_3d_motion_("ref_m.mp4", t2m_kinematic_chain, joints_ref_m[0].cpu().numpy(), "ref_m", fps=20)

        # import pdb; pdb.set_trace()

        # renorm for t2m eval
        feats_rst = self.data_loader.dataset.renorm4t2m(feats_rst)
        motions = self.data_loader.dataset.renorm4t2m(motions)

        # t2m motion encoder
        align_idx = np.argsort(lengths.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        sample = feats_rst_t2m[align_idx]
        lengths = lengths[align_idx]
        lengths = torch.div(lengths, 4, rounding_mode="floor")

        recons_mov = self.movement_enc(sample[..., :-4]).detach()
        recons_emb = self.motion_enc(recons_mov, lengths)
        motion_mov = self.movement_enc(motions[..., :-4]).detach()
        motion_emb = self.motion_enc(motion_mov, lengths)
        text_emb = self.text_enc(word_embs, pos_ohot, text_lengths)[align_idx]

        return {
            "m_ref": motions,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "predicted": logits,
            "label": label,
            "inference_time": end - start,
            "length": lengths,
        }


    def evaluate(self):
        for i, batch in enumerate(tqdm(self.data_loader, desc="Evaluating")):
            # batch = [word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, tokens, caption2, sent_len2, label, style_text]
            torch.cuda.empty_cache()
            res = self.t2m_eval(batch)
            self.metrics.update(
                res["lat_t"],
                res["lat_rm"],
                res["lat_m"],
                res["length"],
                res["predicted"],
                res["label"],
                res["joints_rst"],
                res["inference_time"],
            )
            # if i > 3:
            #     break

    def compute_metrics(self):
        metrics_dict = self.metrics.compute()
        self.metrics.reset()
        print(metrics_dict)
        for k, v in metrics_dict.items():
            if isinstance(v, float):
                print(f"{k}: {v}")
            elif isinstance(v, torch.Tensor):
                print(f"{k}: {v.item()}")
            elif isinstance(v, np.ndarray):
                print(f"{k}: {v.item()}")
        return metrics_dict


# def load_config(config_path):
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)

#     config_basename = os.path.basename(config_path)
#     config["run_name"] = os.path.splitext(config_basename)[0]
#     config["result_dir"] = os.path.join(config["result_dir"], config["run_name"])
#     config["checkpoint_dir"] = os.path.join(config["checkpoint_dir"], config["run_name"])
#     return config


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file (YAML)')
    parser.add_argument('--style_weight', type=float, default=None,
                        help='Override config.model.style_weight at runtime')
    parser.add_argument('--csv_name', type=str, default=None,
                        help='CSV filename to write results to (e.g., metrics_style0.5.csv)')
    args = parser.parse_args()

    from pathlib import Path
    cfg_path = Path(args.config).resolve()

    with cfg_path.open('r') as f:
        config = yaml.safe_load(f)

    parts = cfg_path.parts
    if "configs" in parts:
        i = parts.index("configs")
        sub = Path(*parts[i+1:]).with_suffix("")
        run_name = str(sub).replace("\\", "/")
    else:
        run_name = cfg_path.stem

    config["run_name"] = run_name
    config["result_dir"]     = os.path.join(config["result_dir"], run_name)
    config["checkpoint_dir"] = os.path.join(config["checkpoint_dir"], run_name)

    if args.style_weight is not None:
        if "model" not in config or not isinstance(config["model"], dict):
            config["model"] = {}
        config["model"]["style_weight"] = args.style_weight

    if args.csv_name is not None:
        config["csv_name"] = args.csv_name  # store for use later

    return config


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, default="./configs/")
    # args = parser.parse_args()

    # config = load_config(args.config)
    config = load_config()
    set_seed(config["random_seed"])
    with torch.no_grad():
        evaluator = SmoodiEval(config)
        evaluator.evaluate()
        metrics = evaluator.compute_metrics()

        # Write csv
        import csv
        row = {"run": config["run_name"]}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            elif isinstance(v, np.ndarray):
                v = float(v)
            row[k] = v

        csv_dir  = './evaluation'
        csv_name = config.get("csv_name", "metrics.csv")
        csv_file = os.path.join(csv_dir, csv_name)
        os.makedirs(csv_dir, exist_ok=True)
        exists = os.path.isfile(csv_file)

        with open(csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["run"] + list(metrics.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)