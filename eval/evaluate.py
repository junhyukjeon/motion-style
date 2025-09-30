# This code is based on https://github.com/neu-vi/SMooDi/blob/37d5a43b151e0b60c52fc4b37bddbb5923f14bb7/mld/models/modeltype/mld.py#L1449

import torch
import json
import time
import os
import argparse
import yaml
from tqdm import tqdm

import dist_util
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
from tm2t import TM2TMetrics
from style_classifier import load_classifier
from data_loaders.humanml.networks.evaluator_wrapper import build_evaluators
from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from data_loaders.tensors import lengths_to_mask

from model.t2sm import Text2StylizedMotion

# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config_basename = os.path.basename(config_path)
    config["run_name"] = os.path.splitext(config_basename)[0]
    config["result_dir"] = os.path.join(config["result_dir"], config["run_name"])
    config["checkpoint_dir"] = os.path.join(config["checkpoint_dir"], config["run_name"])
    return config

class SmoodiEval():
    def __init__(self, args, lora_base_path):
        
        self.args = args
        self.lora_base_path = lora_base_path

        self.classifier, self.cassifier_styles, self.label_to_style  = load_classifier(args.classifier_style_group)
        self.style_to_label = {v:k for k,v in self.label_to_style.items()}

        badly_proccesd_styles = ['Zombie','WiggleHips', 'WhirlArms', 'WildArms', 'WildLegs', 'WideLegs']
        
        self.styles = self.cassifier_styles
        assert len(set(self.cassifier_styles) & set(badly_proccesd_styles)) == 0

        # init model and metrics
        self.metrics = TM2TMetrics().to(dist_util.dev())
        
        self.text_enc, self.motion_enc, self.movement_enc = build_evaluators({
            'dataset_name': 't2m',
            'device': dist_util.dev(),
            'dim_word': 300,
            'dim_pos_ohot': 15,
            'dim_motion_hidden': 1024,
            'dim_text_hidden': 512,
            'dim_coemb_hidden': 512,
            'dim_pose': 263,
            'dim_movement_enc_hidden': 512,
            'dim_movement_latent': 512,
            'checkpoints_dir': '.',
            'unit_length': 4,
            })
                                                                             
        self.data_loader = get_dataset_loader(name='humanml', batch_size=args.batch_size, num_frames=None, split='test', hml_mode='gt')
               
        self.model = Text2StylizedMotion(args.config["model"]).to(dist_util.dev())
        # self.model.load_state_dict(torch.load(os.path.join(args.config["checkpoint_dir"], "best.ckpt"), map_location=dist_util.dev()))
        self.model.eval()
        
        # Save
        self.save_dir = os.path.join(args.config["result_dir"], "samples")
        os.makedirs(self.save_dir, exist_ok=True)
        self.sample_idx = 0  # running counter across batches
  
    # def _style_to_model(self, style):
    #     if not self.args.lora_finetune:
    #         self.style_set=False
    #         return

    #     if self.args.guidance_param != 1:
    #         model = self.model.model
    #     else:
    #         model = self.model

    #     self.style_set=True
        
    def _procees_dataset(self):
        n_styles = len(self.styles)
        replace_every = len(self.data_loader) // n_styles
        style = None
        
        for i, batch in enumerate(tqdm(self.data_loader)):   
            if i % replace_every == 0:
                style_idx = (i // replace_every) % n_styles
                style = self.styles[style_idx]
                # self._style_to_model(style)
                
            # assert self.style_set or not self.args.lora_finetune
            rs_set = self._procces_batch(batch, style)
            self.metrics.update(
                rs_set["text_embeddings"],
                rs_set["gen_motion_embeddings"],
                rs_set["gt_motion_embeddings"],
                rs_set["lengths"],
                rs_set["our_predicted"],
                rs_set["label"],
                rs_set["joints_gen"],
            )

    def _procces_batch(self, batch, gen_style):   
        # batch = [word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, tokens]
        
        word_embs, pos_ohot, texts, text_lengths, gt_motions, lengths, *_ = batch
        # gt_motions = batch[4].permute(0, 2,1) # B J F
        gt_motions = gt_motions.to(dist_util.dev()).float()
        lengths = lengths.to(dist_util.dev())
        # lengths = batch[5]
        # word_embs = batch[0]
        # pos_ohot = batch[1]
        # texts = batch[2]
        # text_lengths = batch[3]

        # sample.shape = # B J 1 F
        sample = self._sample(texts, lengths, gt_motions)

        # >>> save generated motions here <<<
        save_dir = os.path.join(self.args.config["result_dir"], "raw_samples")
        os.makedirs(save_dir, exist_ok=True)

        # detach to cpu numpy
        np.save(
            os.path.join(save_dir, f"sample_{self.sample_idx:06d}.npy"),
            sample.detach().cpu().numpy()
        )
        self.sample_idx += 1
        # <<< end saving >>>

        sample = sample.permute(0, 2, 1).unsqueeze(2) # B J 1 F
        gt_motions = gt_motions.permute(0, 2, 1).unsqueeze(2)

        # style classification
        logits = self.classifier(sample, lengths.to(dist_util.dev()))

        # renorm 
        gt_motions_unnorm  = self.data_loader.dataset.inv_transform(gt_motions.cpu().permute(0, 2, 3, 1)).float()
        gen_motions_unnorm = self.data_loader.dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        
        gt_motions_renorm_t2m  = self.data_loader.dataset.renorm4t2m(gt_motions.cpu().permute(0, 2, 3, 1)).float()
        gen_motions_renorm_t2m = self.data_loader.dataset.renorm4t2m(sample.cpu().permute(0, 2, 3, 1)).float()

        
        # joints recover
        n_joints = 22
        joints_gt = recover_from_ric(gt_motions_unnorm.squeeze(1), n_joints)
        joints_gen = recover_from_ric(gen_motions_unnorm.squeeze(1), n_joints)


        # t2m motion encoder
        m_lens = lengths
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        gt_motions_renorm_t2m = gt_motions_renorm_t2m[align_idx]
        gen_motions_renorm_t2m = gen_motions_renorm_t2m[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens, 4, rounding_mode="floor")
        

        gen_motion_mov = self.movement_enc(gen_motions_renorm_t2m[..., :-4].squeeze(1)).detach()
        gen_motion_embeddings = self.motion_enc(gen_motion_mov, m_lens)
        gt_motion_mov = self.movement_enc(gt_motions_renorm_t2m[..., :-4].squeeze(1)).detach()
        gt_motion_embeddings = self.motion_enc(gt_motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.text_enc(word_embs.float(), pos_ohot.float(), text_lengths)[align_idx] 

        res = {
                "gt_motions": gt_motions_renorm_t2m,
                "gen_motions": gen_motions_renorm_t2m,
                "text_embeddings": text_emb,
                "gt_motion_embeddings": gt_motion_embeddings,
                "gen_motion_embeddings": gen_motion_embeddings, 
                "joints_ref": joints_gt,
                "joints_gen": joints_gen,
                "our_predicted": logits,
                "label": torch.tensor([self.cassifier_styles.index(gen_style)]*len(lengths)),
                'lengths': lengths,
        }
        
        return res
    
    def _sample(self, texts, lengths, gt_motions):
        batch_size = gt_motions.shape[0]
        n_frames = 196
        
        import pdb; pdb.set_trace()
        # batch = [gt_motions, texts, torch.zeros(32), torch.zeros(32)]

        # TODO: generation
        sample = self.model.generate(gt_motions, texts)
        return sample[0]
    
    def finish(self):
        metrics_dict, count_seq = self.metrics.compute()
        print(metrics_dict)
        print(f'num samples={count_seq}')
        return metrics_dict
    
def main():
    args = evaluation_parser()
    args.batch_size = 32
    args.config = load_config(args.config)
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    with torch.no_grad():
        eval = SmoodiEval(args, lora_base_path='save/lora')
        eval._procees_dataset()
        res = eval.finish()  
        
    with open(f"main_eval.json", 'w') as f:
        json.dump(res, f, indent=4)
       
              
if __name__ == '__main__':
    main()