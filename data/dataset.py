import json
import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset


class TextStyleDataset(Dataset):
    def __init__(self, config, styles):
        self.motion_dir  = config["motion_dir"]
        self.mean        = np.load(config["mean_path"])
        self.std         = np.load(config["std_path"])
        self.window_size = config["window_size"]

        # Style
        with open(config["style_json"], "r") as f:
            style_map = json.load(f)
        styles_sorted = sorted(style_map.keys())
        self.style_to_style_idx = {style: i for i, style in enumerate(styles_sorted)}
        self.style_idx_to_style = {i: style for style, i in self.style_to_style_idx.items()}
        self.style_map          = {self.style_to_style_idx[style]: style_map[style] for style in styles}

        # Content
        with open(config["content_json"], "r") as f:
            content_map = json.load(f)
        contents_sorted = sorted(content_map.keys())
        self.content_to_content_idx = {c: i for i, c in enumerate(contents_sorted)}
        self.content_idx_to_content = {i: c for c, i in self.content_to_content_idx.items()}

        # Motion ID to content index
        self.motion_to_content_idx = {}
        for content, motion_ids in content_map.items():
            content_idx = self.content_to_content_idx[content]
            for motion_id in motion_ids:
                self.motion_to_content_idx[motion_id] = content_idx

        self.content_prompts = {
            "BR": "a person is running backward",
            "BW": "a person is walking backward",
            "FR": "a person is running forward",
            "FW": "a person is walking forward",
            "ID": "a person is standing still",
            "SR": "a person is running sideways",
            "SW": "a person is walking sideways",
        }
        
        exclude_keys = ["TR1", "TR2", "TR3"]
        self.exclude_content_idcs = {
            self.content_to_content_idx[k] for k in exclude_keys
            if k in self.content_to_content_idx
        }
        
        self.items, self.lengths = [], []
        for style_idx, motion_ids in self.style_map.items():
            for motion_id in motion_ids:
                motion_path = os.path.join(config["motion_dir"], f"{motion_id}.npy")
                motion      = np.load(motion_path, mmap_mode="r")
                if motion.shape[0] < self.window_size:
                    continue

                # text_path   = os.path.join(config["text_dir"], f"{motion_id}.txt")
                # captions = []
                # with open(text_path, "r") as f:
                #     for line in f:
                #         parts = line.strip().split("#")
                #         if parts and parts[0]:
                #             captions.append(parts[0].strip())

                content_idx = self.motion_to_content_idx[motion_id]
                if content_idx is None or content_idx in self.exclude_content_idcs:
                    continue

                self.items.append({"motion_id": motion_id, "style_idx": style_idx, "content_idx": content_idx})
                self.lengths.append(motion.shape[0] - self.window_size)
        self.cumsum = np.cumsum([0] + self.lengths)

    def __len__(self):
        return int(self.cumsum[-1])

    def __getitem__(self, idx):
        if idx != 0:
            motion_idx = np.searchsorted(self.cumsum, idx) - 1
            offset = idx - self.cumsum[motion_idx] - 1
        else:
            motion_idx = 0
            offset = 0

        meta = self.items[motion_idx]
        motion_id, style_idx, content_idx = meta["motion_id"], meta["style_idx"], meta["content_idx"]

        start, end = offset, offset + self.window_size
        motion = np.load(os.path.join(self.motion_dir, f"{motion_id}.npy"), mmap_mode="r")
        window = motion[start:end]
        window = (window - self.mean) / self.std
        window = torch.tensor(window, dtype=torch.float32)

        content_id = self.content_idx_to_content[content_idx]
        caption = self.content_prompts[content_id]
        return window, caption, style_idx, content_idx


class TextStyleDataset2(Dataset):
    def __init__(self, config, styles):
        self.motion_dir  = config["motion_dir"]
        self.mean        = np.load(config["mean_path"])
        self.std         = np.load(config["std_path"])
        # self.window_size = config["window_size"]

        # Style
        with open(config["style_json"], "r") as f:
            style_map = json.load(f)
        styles_sorted = sorted(style_map.keys())
        self.style_to_style_idx = {style: i for i, style in enumerate(styles_sorted)}
        self.style_idx_to_style = {i: style for style, i in self.style_to_style_idx.items()}
        self.style_map          = {self.style_to_style_idx[style]: style_map[style] for style in styles}

        # Content
        with open(config["content_json"], "r") as f:
            content_map = json.load(f)
        contents_sorted = sorted(content_map.keys())
        self.content_to_content_idx = {c: i for i, c in enumerate(contents_sorted)}
        self.content_idx_to_content = {i: c for c, i in self.content_to_content_idx.items()}

        # Motion ID to content index
        self.motion_to_content_idx = {}
        for content, motion_ids in content_map.items():
            content_idx = self.content_to_content_idx[content]
            for motion_id in motion_ids:
                self.motion_to_content_idx[motion_id] = content_idx

        self.content_prompts = {
            "BR": "a person is running backward",
            "BW": "a person is walking backward",
            "FR": "a person is running forward",
            "FW": "a person is walking forward",
            "ID": "a person is standing still",
            "SR": "a person is running sideways",
            "SW": "a person is walking sideways",
        }
        
        exclude_keys = ["TR1", "TR2", "TR3"]
        self.exclude_content_idcs = {
            self.content_to_content_idx[k] for k in exclude_keys
            if k in self.content_to_content_idx
        }
        
        self.items = []
        for style_idx, motion_ids in self.style_map.items():
            for motion_id in motion_ids:
                motion_path = os.path.join(config["motion_dir"], f"{motion_id}.npy")
                motion      = np.load(motion_path, mmap_mode="r")
                num_frames  = motion.shape[0]
                if num_frames < config['min_frames']:
                    continue

                content_idx = self.motion_to_content_idx[motion_id]
                if content_idx is None or content_idx in self.exclude_content_idcs:
                    continue

                self.items.append({"motion_id": motion_id, "style_idx": style_idx, "content_idx": content_idx})
                self.lengths.append(motion.shape[0] - self.window_size)
        self.cumsum = np.cumsum([0] + self.lengths)

    def __len__(self):
        return int(self.cumsum[-1])

    def __getitem__(self, idx):
        if idx != 0:
            motion_idx = np.searchsorted(self.cumsum, idx) - 1
            offset = idx - self.cumsum[motion_idx] - 1
        else:
            motion_idx = 0
            offset = 0

        meta = self.items[motion_idx]
        motion_id, style_idx, content_idx = meta["motion_id"], meta["style_idx"], meta["content_idx"]

        start, end = offset, offset + self.window_size
        motion = np.load(os.path.join(self.motion_dir, f"{motion_id}.npy"), mmap_mode="r")
        window = motion[start:end]
        window = (window - self.mean) / self.std
        window = torch.tensor(window, dtype=torch.float32)

        content_id = self.content_idx_to_content[content_idx]
        caption = self.content_prompts[content_id]
        return window, caption, style_idx, content_idx


# class Text2MotionDataset(data.Dataset):
#     def __init__(self, opt, mean, std, split_file):
#         self.opt = opt
#         self.max_length = 20
#         self.pointer = 0
#         self.max_motion_length = opt.max_motion_length
#         min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

#         data_dict = {}
#         id_list = []
#         with cs.open(split_file, 'r') as f:
#             for line in f.readlines():
#                 id_list.append(line.strip())
#         # id_list = id_list[:250]

#         new_name_list = []
#         length_list = []
#         for name in tqdm(id_list):
#             try:
#                 motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
#                 if (len(motion)) < min_motion_len or (len(motion) >= 200):
#                     continue
#                 text_data = []
#                 flag = False
#                 with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
#                     for line in f.readlines():
#                         text_dict = {}
#                         line_split = line.strip().split('#')
#                         # print(line)
#                         caption = line_split[0]
#                         tokens = line_split[1].split(' ')
#                         f_tag = float(line_split[2])
#                         to_tag = float(line_split[3])
#                         f_tag = 0.0 if np.isnan(f_tag) else f_tag
#                         to_tag = 0.0 if np.isnan(to_tag) else to_tag

#                         text_dict['caption'] = caption
#                         text_dict['tokens'] = tokens
#                         if f_tag == 0.0 and to_tag == 0.0:
#                             flag = True
#                             text_data.append(text_dict)
#                         else:
#                             try:
#                                 n_motion = motion[int(f_tag*20) : int(to_tag*20)]
#                                 if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
#                                     continue
#                                 new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
#                                 while new_name in data_dict:
#                                     new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
#                                 data_dict[new_name] = {'motion': n_motion,
#                                                        'length': len(n_motion),
#                                                        'text':[text_dict]}
#                                 new_name_list.append(new_name)
#                                 length_list.append(len(n_motion))
#                             except:
#                                 print(line_split)
#                                 print(line_split[2], line_split[3], f_tag, to_tag, name)
#                                 # break

#                 if flag:
#                     data_dict[name] = {'motion': motion,
#                                        'length': len(motion),
#                                        'text': text_data}
#                     new_name_list.append(name)
#                     length_list.append(len(motion))
#             except Exception as e:
#                 # print(e)
#                 pass

#         # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
#         name_list, length_list = new_name_list, length_list

#         self.mean = mean
#         self.std = std
#         self.length_arr = np.array(length_list)
#         self.data_dict = data_dict
#         self.name_list = name_list

#     def inv_transform(self, data):
#         return data * self.std + self.mean

#     def __len__(self):
#         return len(self.data_dict) - self.pointer

#     def __getitem__(self, item):
#         idx = self.pointer + item
#         data = self.data_dict[self.name_list[idx]]
#         motion, m_length, text_list = data['motion'], data['length'], data['text']
#         # Randomly select a caption
#         text_data = random.choice(text_list)
#         caption, tokens = text_data['caption'], text_data['tokens']

#         if self.opt.unit_length < 10:
#             coin2 = np.random.choice(['single', 'single', 'double'])
#         else:
#             coin2 = 'single'

#         if coin2 == 'double':
#             m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
#         elif coin2 == 'single':
#             m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
#         idx = random.randint(0, len(motion) - m_length)
#         motion = motion[idx:idx+m_length]

#         "Z Normalization"
#         motion = (motion - self.mean) / self.std

#         if m_length < self.max_motion_length:
#             motion = np.concatenate([motion,
#                                      np.zeros((self.max_motion_length - m_length, motion.shape[1]))
#                                      ], axis=0)
#         # print(word_embeddings.shape, motion.shape)
#         # print(tokens)
#         return caption, motion, m_length

#     def reset_min_len(self, length):
#         assert length <= self.max_motion_length
#         self.pointer = np.searchsorted(self.length_arr, length)
#         print("Pointer Pointing at %d" % self.pointer)


# class StyleDataset(Dataset):
#     def __init__(self, motion_dir, mean, std, window_size, label_to_ids, style_to_label, ids_to_content):
#         # Normalize motion data for VAE
#         self.mean = mean
#         self.std = std 
#         self.window_size = window_size

#         # Style-to-label mapping
#         self.style_to_label = style_to_label
#         self.label_to_style = {i: style for style, i in self.style_to_label.items()}

#         # IDs to content
#         self.ids_to_content = ids_to_content

#         self.data = []       # List of (motion_id, motion_array)
#         self.labels = []     # List of label indices
#         self.lengths = []    # number of valid windows for each motion
#         self.ids = []        
#         self.ids_to_idx = {} 

#         for style_name, motion_ids in label_to_ids.items():
#             label_idx = self.style_to_label[style_name]
#             for motion_id in motion_ids:
#                 try:
#                     motion_path = os.path.join(motion_dir, f"{motion_id}.npy")
#                     motion = np.load(motion_path)
#                     if motion.shape[0] < window_size:
#                         continue
#                     self.data.append((motion_id, motion))
#                     self.labels.append(label_idx)
#                     self.lengths.append(motion.shape[0] - window_size)
#                     self.ids.append(motion_id)
#                     self.ids_to_idx[motion_id] = len(self.data) - 1
#                 except Exception as e:
#                     print(f"❌ Error loading {motion_id}: {e}")

#         self.cumsum = np.cumsum([0] + self.lengths)

#     def __len__(self):
#         return self.cumsum[-1]

#     def __getitem__(self, idx):
#         if idx != 0:
#             motion_idx = np.searchsorted(self.cumsum, idx) - 1
#             offset = idx - self.cumsum[motion_idx] - 1
#         else:
#             motion_idx = 0
#             offset = 0

#         motion_id, motion = self.data[motion_idx]
#         window  = motion[offset:offset + self.window_size]
#         window  = (window - self.mean) / self.std
#         label   = self.labels[motion_idx]
#         content = self.ids_to_content[motion_id]

#         return torch.tensor(window, dtype=torch.float32), label, content, motion_id