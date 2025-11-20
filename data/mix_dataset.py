import os
from os.path import join as pjoin

import random
import codecs as cs
import numpy as np
from tqdm import tqdm

from torch.utils import data
# from torch.utils.data._utils.collate import default_collate


def build_dict_from_txt(filename):
    result_dict = {}
    
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) >= 3:
                key = parts[0]

                style_text = parts[1].split("_")[0]
                style_label = parts[2]

                result_dict[key] = (style_text, style_label)
                
    return result_dict


class MixedTextStyleDataset(data.Dataset):
    def __init__(
        self,
        config,
        styles,
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):
        self.mean = np.load(config["mean_path"])
        self.std  = np.load(config["std_path"])

        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = config["max_frames"]
        self.min_motion_length = config["min_frames"]
        self.unit_length = config["unit_length"]

        data_dict_1 = {}
        id_list_1 = []

        data_dict_2 = {}
        id_list_2 = []

        split_file = config["split_file"]
        split_dir = os.path.dirname(split_file)
        split_base = os.path.basename(split_file).split(".")[0]
        split_subfile_1 = pjoin(split_dir, split_base + "_100STYLE_Full.txt")
        split_subfile_2 = pjoin(split_dir, split_base + "_humanml.txt")
        motion_dir = pjoin(split_dir, "new_joint_vecs")
        text_dir = pjoin(split_dir, "texts")
        motion2label = build_dict_from_txt(pjoin(split_dir, "100STYLE_name_dict.txt"))

        self.style_idx_to_style = {}
        for k, (v1, v2) in motion2label.items():
            if v2 not in self.style_idx_to_style:
                self.style_idx_to_style[int(v2)] = v1
            else:
                assert self.style_idx_to_style[int(v2)] == v1

        with cs.open(split_subfile_1, "r") as f:
            for line in f.readlines():
                id_list_1.append(line.strip())
        self.id_list_1 = id_list_1

        with cs.open(split_subfile_2, "r") as f:
            for line in f.readlines():
                id_list_2.append(line.strip())
        self.id_list_2 = id_list_2

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        if progress_bar:
            enumerator_1 = enumerate(
                tqdm(
                    id_list_1,
                    f"Loading 100STYLE {split_subfile_1.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator_1 = enumerate(id_list_1)

        count = 0
        bad_count = 0
        new_name_list_1 = []
        length_list_1 = []
        style_list_1 = []

        for i, name in enumerator_1:
            if count > maxdata:
                break

            style_text, style_label = motion2label[name]
            if not (style_text in styles):
                continue
            style_label = int(style_label)

            motion = np.load(pjoin(motion_dir, name + ".npy"))[1:]
            if (len(motion)) < self.min_motion_length or (len(motion) >=200):
                bad_count += 1
                continue
            if np.max(np.abs((motion - self.mean) / self.std)) > 1e3: # filter outliers
                bad_count += 1
                continue
            text_data_1 = []
            flag = True

            # name = style_to_neutral[name]
            text_path = pjoin(text_dir, name + ".txt")
            assert os.path.exists(text_path)
            with cs.open(text_path) as f:
                for line in f.readlines():
                    text_dict_1 = {}
                    line_split = line.strip().split("#")
                    caption = line_split[0]
                    text_dict_1["caption"] = caption
                    text_data_1.append(text_dict_1)
                
                if flag:
                    data_dict_1[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data_1,
                        "style": style_label,
                    }
                    new_name_list_1.append(name)
                    length_list_1.append(len(motion))
                    style_list_1.append(style_label)
                    count += 1

        name_list_1, length_list_1, style_list_1 = zip(
            *sorted(zip(new_name_list_1, length_list_1, style_list_1), key=lambda x: x[1]))

        if progress_bar:
            enumerator_2 = enumerate(
                tqdm(
                    id_list_2,
                    f"Loading HumanML3D {split_subfile_2.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator_2 = enumerate(id_list_2)

        count = 0
        bad_count = 0
        new_name_list_2 = []
        length_list_2 = []
        style_list_2 = []

        for i, name in enumerator_2:
            if count > maxdata:
                break
            motion = np.load(pjoin(motion_dir, name + ".npy"))
            if (len(motion)) < self.min_motion_length or (len(motion) >=200):
                bad_count += 1
                continue
            text_data_2 = []
            flag = True

            # name = style_to_neutral[name]
            text_path = pjoin(text_dir, name + ".txt")
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
                        "style": -1,
                    }
                    new_name_list_2.append(name)
                    length_list_2.append(len(motion))
                    style_list_2.append(-1)
                    count += 1            

        name_list_2, length_list_2, style_list_2 = zip(
            *sorted(zip(new_name_list_2, length_list_2, style_list_2), key=lambda x: x[1]))

        self.length_arr_1 = np.array(length_list_1)
        self.data_dict_1 = data_dict_1
        self.name_list_1 = name_list_1
        self.style_list_1 = style_list_1

        self.length_arr_2 = np.array(length_list_2)
        self.data_dict_2 = data_dict_2
        self.name_list_2 = name_list_2
        self.style_list_2 = style_list_2

        self.nfeats = motion.shape[1]
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length

        self.pointer_1 = np.searchsorted(self.length_arr_1, length)
        print("Pointer Pointing at %d" % self.pointer_1)

        # self.pointer_2 = np.searchsorted(self.length_arr_2, length)
        # print("Pointer Pointing at %d" % self.pointer_2)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def transform(self,data):
        return (data - self.mean) / self.std

    def get_mean_std(self):
        return self.mean, self.std
    
    def __len__(self):
        return len(self.name_list_1) - self.pointer

    def __getitem__(self, item):
        idx_1 = (self.pointer_1 + item) #% len(self.name_list_1)
        data_1 = self.data_dict_1[self.name_list_1[idx_1]]
        motion_1, m_length_1, text_list_1, style_1 = data_1["motion"], data_1["length"], data_1["text"], data_1["style"]

        idx_2 = (self.pointer_1 + item + random.randint(0,len(self.name_list_2)-1)) % len(self.name_list_2)
        data_2 = self.data_dict_2[self.name_list_2[idx_2]]
        motion_2, m_length_2, text_list_2, style_2 = data_2["motion"], data_2["length"], data_2["text"], data_2["style"]
      
        # Randomly select a caption
        text_data_1 = random.choice(text_list_1)
        caption_1 = text_data_1["caption"]

        text_data_2 = random.choice(text_list_2)
        caption_2 = text_data_2["caption"]

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
        
        m_length_1 = min(196, m_length_1)
        m_length_2 = min(196, m_length_2)
        idx_1 = random.randint(0, len(motion_1) - m_length_1)
        motion_1 = motion_1[idx_1:idx_1 + m_length_1]
        "Z Normalization"
        motion_1 = (motion_1 - self.mean) / self.std


        idx_2 = random.randint(0, len(motion_2) - m_length_2)
        motion_2 = motion_2[idx_2:idx_2 + m_length_2]
        "Z Normalization"
        motion_2 = (motion_2 - self.mean) / self.std

        # # debug check nan
        # if np.any(np.isnan(motion_1)):
        #     raise ValueError("nan in motion")
        
        if m_length_1 < self.max_motion_length:
            pad_len = self.max_motion_length - m_length_1
            motion_1 = np.pad(motion_1, ((0, pad_len), (0, 0)))
        if m_length_2 < self.max_motion_length:
            pad_len = self.max_motion_length - m_length_2
            motion_2 = np.pad(motion_2, ((0, pad_len), (0, 0)))

        return (
            caption_1,
            motion_1.astype(np.float32),
            m_length_1,
            style_1,

            caption_2,
            motion_2.astype(np.float32),
            m_length_2,
            style_2,
        )
