"""
Mostly brought by SMooDi:
- https://github.com/neu-vi/SMooDi/blob/d429bfa744de29ae556fae8ce26881efd9396eca/mld/data/humanml/data/dataset.py
"""

import os
from os.path import join as pjoin

import random
import numpy as np
import codecs as cs
from rich.progress import track

import torch
from torch.utils.data import Dataset

from salad.utils.motion_process import recover_from_ric

def build_dict_from_txt(filename,is_style=True,is_style_text=False):
    result_dict = {}
    
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) >= 3:
                key = parts[0]
                if is_style and is_style_text == False:
                    value = parts[2]
                elif is_style_text:
                    value = parts[1].split("_")[0]
                else:
                    value = parts[3]


                result_dict[key] = value
                
    return result_dict


class Text2MotionTestDataset(Dataset):
    def __init__(
        self,
        mean,
        std,
        mean_eval,
        std_eval,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir1,
        text_dir1,
        motion_dir2,
        text_dir2,
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =="t2m" else 24
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        self.mean = mean
        self.std = std
        self.mean_eval = mean_eval
        self.std_eval = std_eval

        data_dict_1 = {}
        id_list_1 = []

        data_dict_2 = {}
        id_list_2 = []

        split_dir = pjoin(os.path.dirname(split_file), "../100style")
        split_base = os.path.basename(split_file).split(".")[0]
        split_subfile_1 = pjoin(split_dir, split_base + "_humanml.txt")
        split_subfile_2 = pjoin(split_dir, split_base + "_100STYLE_Filter.txt")

        dict_path = "./dataset/100style/100STYLE_name_dict_Filter.txt"
        motion_to_label = build_dict_from_txt(dict_path)
        motion_to_style_text = build_dict_from_txt(dict_path,is_style_text=True)

        with cs.open(split_subfile_1, "r") as f:
            for line in f.readlines():
                id_list_1.append(line.strip())
        
        id_list_1 = np.array(id_list_1)
        # Use random_samples to index id_list_1_np
        # id_list_1 = id_list_1[random_samples]

        self.id_list_1 = id_list_1

        with cs.open(split_subfile_2, "r") as f:
            for line in f.readlines():
                id_list_2.append(line.strip())
        
        id_list_2 = id_list_2
        self.id_list_2 = id_list_2

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        if progress_bar:
            enumerator_1 = enumerate(
                track(
                    id_list_1,
                    f"Loading HumanML3D {split_subfile_1.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator_1 = enumerate(id_list_1)

        count = 0
        bad_count = 0
        new_name_list_1 = []
        length_list_1 = []

        for i, name in enumerator_1:
            if count > maxdata:
                break
            motion = np.load(pjoin(motion_dir1, name + ".npy"))
            if (len(motion)) < self.min_motion_length or (len(motion) >=200):
                bad_count += 1
                continue
            text_data_1 = []
            flag = False

            # name = style_to_neutral[name]
            text_path = pjoin(text_dir1, name + ".txt")
            assert os.path.exists(text_path)
            with cs.open(text_path) as f:
                for line in f.readlines():
                    text_dict_1 = {}
                    line_split = line.strip().split("#")

                    caption = line_split[0]
                    tokens = line_split[1].split(" ")
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict_1["caption"] = caption
                    text_dict_1["tokens"] = tokens
                    text_data_1.append(text_dict_1)
                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data_1.append(text_dict_1)
                    else:
                        try:
                            n_motion = motion[int(f_tag * 20):int(to_tag *20)]
                            if (len(n_motion)) < self.min_motion_length or ((len(n_motion) >= 200)):
                                continue
                            new_name = (
                                random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                "_" + name)
                            while new_name in data_dict_1:
                                new_name = (random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" +name)
                            data_dict_1[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict_1],
                                }
                            new_name_list_1.append(new_name)
                            length_list_1.append(len(n_motion))
                        except:
                            print(line_split)
                            print(line_split[2], line_split[3], f_tag,to_tag, name)

                
                if flag:
                    data_dict_1[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data_1,
                    }
                    new_name_list_1.append(name)
                    length_list_1.append(len(motion))
                    count += 1            

        name_list_1, length_list_1 = zip(
            *sorted(zip(new_name_list_1, length_list_1), key=lambda x: x[1]))

        if progress_bar:
            enumerator_2 = enumerate(
                track(
                    id_list_2,
                    f"Loading 100STYLE {split_subfile_2.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator_2 = enumerate(id_list_2)

        count = 0
        bad_count = 0
        new_name_list_2 = []
        length_list_2 = []

        for i, name in enumerator_2:
            if count > maxdata:
                break
            motion = np.load(pjoin(motion_dir2, name + ".npy"))[1:]
            label_data = motion_to_label[name]
            style_text = motion_to_style_text[name]

            if (len(motion)) < self.min_motion_length or (len(motion) >=200):
                bad_count += 1
                continue
            if np.max(np.abs((motion - self.mean_eval) / self.std_eval)) > 1e3: # filter outliers
                bad_count += 1
                continue
            text_data_2 = []
            flag = True

            text_path = pjoin(text_dir2, name + ".txt")
            assert os.path.exists(text_path)
            with cs.open(text_path) as f:
                for line in f.readlines():
                    text_dict_2 = {}
                    line_split = line.strip().split("#")
                    caption = line_split[0]
                    text_dict_2["caption"] = caption
                    text_data_2.append(text_dict_2)

                if flag:
                    data_dict_2[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data_2,
                        "label": label_data,
                        "style_text":style_text,
                    }
                    new_name_list_2.append(name)
                    length_list_2.append(len(motion))
                    count += 1            

        name_list_2, length_list_2 = zip(
            *sorted(zip(new_name_list_2, length_list_2), key=lambda x: x[1]))

        
        self.length_arr_1 = np.array(length_list_1)
        self.data_dict_1 = data_dict_1
        self.name_list = name_list_1

        self.length_arr_2 = np.array(length_list_2)
        self.data_dict_2 = data_dict_2
        self.name_list_2 = name_list_2

        self.nfeats = motion.shape[1]
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length

        self.pointer_1 = np.searchsorted(self.length_arr_1, length)
        print("Pointer Pointing at %d" % self.pointer_1)

        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def transform(self,data):
        return (data - self.mean) / self.std

    def get_mean_std(self):
        return self.mean, self.std
    
    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx_1 = (self.pointer_1 + item) % len(self.name_list)
        data_1 = self.data_dict_1[self.name_list[idx_1]]
        motion_1, m_length_1, text_list_1 = data_1["motion"], data_1["length"], data_1["text"]

        idx_2 = (self.pointer_1 + item + random.randint(0,len(self.name_list_2)-1)) % len(self.name_list_2)
        data_2 = self.data_dict_2[self.name_list_2[idx_2]]
        motion_2, m_length_2, text_list_2,label,style_text = data_2["motion"], data_2["length"], data_2["text"], data_2["label"], data_2["style_text"]
      
        # Randomly select a caption
        text_data_1 = random.choice(text_list_1)
        caption_1,tokens = text_data_1["caption"], text_data_1["tokens"]

        text_data_2 = random.choice(text_list_2)
        caption_2 = text_data_2["caption"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)


        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length_1 = (m_length_1 // self.unit_length - 1) * self.unit_length
            m_length_2 = (m_length_2 // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length_1 = (m_length_1 // self.unit_length) * self.unit_length
            m_length_2 = (m_length_2 // self.unit_length) * self.unit_length
        
        idx_1 = random.randint(0, len(motion_1) - m_length_1)
        motion_1 = motion_1[idx_1:idx_1 + m_length_1]
        "Z Normalization"
        motion_1 = (motion_1 - self.mean) / self.std

        idx_2 = random.randint(0, len(motion_2) - m_length_2)
        motion_2 = motion_2[idx_2:idx_2 + m_length_2]
        "Z Normalization"
        motion_2 = (motion_2 - self.mean) / self.std

        # debug check nan
        if np.any(np.isnan(motion_1)):
            raise ValueError("nan in motion")

        return (
            word_embeddings,
            pos_one_hots,
            caption_1,
            sent_len,
            motion_1,
            m_length_1,
            "_".join(tokens),

            caption_2,
            motion_2,
            m_length_2,
            label,
            style_text,
        ) # word_embs, pos_ohot, text, text_len, motion, length, tokens, text2, reference_motion, text_len2, label, style_text

    def feats2joints(self, features):
        mean = torch.tensor(self.mean).to(features)
        std = torch.tensor(self.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, 22)

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.mean).to(features)
        ori_std = torch.tensor(self.std).to(features)
        eval_mean = torch.tensor(self.mean_eval).to(features)
        eval_std = torch.tensor(self.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features
    
    def renorm4style(self, features):
        # renorm to style norms for using style evaluators
        ori_mean = torch.tensor(self.mean).to(features)
        ori_std = torch.tensor(self.std).to(features)
        eval_mean = torch.tensor(self.mean_eval).to(features)
        eval_std = torch.tensor(self.std_eval).to(features)
        features = features * eval_std + eval_mean
        features = (features - ori_mean) / ori_std
        return features