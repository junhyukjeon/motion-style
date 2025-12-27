import random
from collections import defaultdict
from torch.utils.data import BatchSampler


class TrainSampler(BatchSampler):
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.batch_size = int(config["batch_size"])
        self.samples_per_style = int(config["samples_per_style"])

        # style_idx -> list of dataset indices (indices into dataset.items)
        self.styles_to_indices = defaultdict(list)
        for idx, item in enumerate(dataset.items):
            style_idx = item["style_idx"]
            self.styles_to_indices[style_idx].append(idx)

        # keep only styles that actually have samples
        self.styles = [s for s, idxs in self.styles_to_indices.items() if len(idxs) > 0]

        if len(self.styles) == 0:
            raise ValueError("TrainSampler100Style: no styles with samples were found in the dataset.")

        self._generate_batches()

    def _generate_batches(self):
        self.batches = []
        B = self.batch_size
        sps = self.samples_per_style
        styles_per_batch = max(1, B // sps)  # how many distinct styles in a batch

        # number of full batches per epoch
        num_batches = len(self.dataset) // B

        # global pool for topping up short batches
        all_indices = [i for s in self.styles for i in self.styles_to_indices[s]]

        for _ in range(num_batches):
            # choose which styles to use in this batch
            if styles_per_batch <= len(self.styles):
                selected_styles = random.sample(self.styles, styles_per_batch)
            else:
                # if we want more styles than exist, sample with replacement
                selected_styles = random.choices(self.styles, k=styles_per_batch)

            batch = []
            for s in selected_styles:
                pool = self.styles_to_indices[s]

                # sample sps indices from this style
                if len(pool) >= sps:
                    batch.extend(random.sample(pool, sps))
                else:
                    # not enough unique motions in this style -> sample with replacement
                    batch.extend(random.choices(pool, k=sps))

            # enforce exact batch_size
            if len(batch) > B:
                batch = batch[:B]
            elif len(batch) < B:
                # top up from global pool
                batch.extend(random.choices(all_indices, k=B - len(batch)))

            self.batches.append(batch)

    def __iter__(self):
        # regenerate and shuffle batches each epoch
        self._generate_batches()
        random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# class TrainSampler(BatchSampler):
#     def __init__(self, config, dataset):
#         self.config = config
#         self.dataset = dataset
#         self.batch_size = config['batch_size']
#         self.samples_per_style = config['samples_per_style']

#         # style_idx -> list of dataset indices (one index per motion)
#         self.styles_to_motions = defaultdict(list)
#         for motion_idx, item in enumerate(dataset.items):
#             style_idx = item["style_idx"]
#             self.styles_to_motions[style_idx].append(motion_idx)

#         # styles that actually have samples
#         self.styles = [s for s, idxs in self.styles_to_motions.items() if len(idxs) > 0]

#         self.generate_batches()

#     def generate_batches(self):
#         self.batches = []
#         B   = self.batch_size
#         sps = self.samples_per_style
#         styles_per_batch = max(1, B // sps)   # how many styles per batch

#         # how many batches to make this epoch
#         num_batches = len(self.dataset) // B

#         # flat pool for filling short batches
#         all_indices = [i for s in self.styles for i in self.styles_to_motions[s]]

#         for _ in range(num_batches):
#             # choose styles for this batch (with replacement if needed)
#             if styles_per_batch <= len(self.styles):
#                 selected_styles = random.sample(self.styles, styles_per_batch)
#             else:
#                 selected_styles = random.choices(self.styles, k=styles_per_batch)

#             batch = []
#             for s in selected_styles:
#                 pool = self.styles_to_motions[s]
#                 # take sps samples from this style (with replacement if too few)
#                 if len(pool) >= sps:
#                     batch.extend(random.sample(pool, sps))
#                 else:
#                     batch.extend(random.choices(pool, k=sps))

#             # adjust to exact batch_size
#             if len(batch) > B:
#                 batch = batch[:B]
#             elif len(batch) < B:
#                 # top up from the global pool
#                 batch.extend(random.choices(all_indices, k=B - len(batch)))

#             self.batches.append(batch)

#     def __iter__(self):
#         self.generate_batches()
#         random.shuffle(self.batches)
#         return iter(self.batches)

#     def __len__(self):
#         return len(self.batches)


# class MixedSampler(BatchSampler):
#     """
#     Batch sampler that enforces style-aware batching on the 100STYLE half
#     of MixedTextStyleDataset. Works with dataset indexing contract:
#         __getitem__(item) uses idx_1 = pointer_1 + item  (100STYLE)
#     so we must emit 'item' in [0, len(name_list_1) - pointer_1).

#     Expects config:
#         - batch_size (int)
#         - samples_per_style (int)
#         - drop_last (optional, bool)
#         - seed (optional, int)
#     """
#     def __init__(self, config, dataset):
#         self.ds = dataset
#         self.B = int(config["batch_size"])
#         self.sps = int(config["samples_per_style"])
#         self.drop_last = bool(config.get("drop_last", False))
#         self.rng = random.Random(config.get("seed", 42))

#         # ---- Build 100STYLE “item” index space ----
#         # name_list_1/style_list_1 are aligned and sorted by length
#         # Valid 100STYLE indices are [pointer_1, len(name_list_1))
#         if not hasattr(dataset, "pointer_1"):
#             raise ValueError("Dataset missing pointer_1; call reset_max_len() before creating the sampler.")
#         p1 = int(dataset.pointer_1)
#         n1 = len(dataset.name_list_1)

#         if p1 >= n1:
#             raise ValueError("pointer_1 >= len(name_list_1): no 100STYLE samples available after length filtering.")

#         # Map: style_idx -> list of dataset “items” (item = global_idx - pointer_1)
#         self.styles_to_items = defaultdict(list)
#         for global_i in range(p1, n1):
#             style_idx = dataset.style_list_1[global_i]
#             item = global_i - p1
#             self.styles_to_items[style_idx].append(item)

#         # Keep only styles that actually have ≥1 sample
#         self.styles = [s for s, lst in self.styles_to_items.items() if len(lst) > 0]
#         if not self.styles:
#             raise ValueError("No 100STYLE styles available for batching.")

#         # Flat pool for top-ups (all valid “item” indices)
#         self.all_items = []
#         for s in self.styles:
#             self.all_items.extend(self.styles_to_items[s])

#         # Precompute an approximate epoch length
#         # (we use the total 100STYLE available; DataLoader will stop at len(self))
#         total = len(self.all_items)
#         self._num_batches = total // self.B if self.drop_last else math.ceil(total / self.B)

#         # Prepare first epoch
#         self._make_epoch_batches()

#     def _choose_styles_for_batch(self, styles_per_batch):
#         # Sample styles without replacement if we can; otherwise with replacement
#         if styles_per_batch <= len(self.styles):
#             return self.rng.sample(self.styles, styles_per_batch)
#         return [self.rng.choice(self.styles) for _ in range(styles_per_batch)]

#     def _take_sps(self, style_idx, k):
#         """Take k items from this style bucket; with replacement if bucket < k."""
#         pool = self.styles_to_items[style_idx]
#         if len(pool) >= k:
#             # sample without replacement
#             return self.rng.sample(pool, k)
#         # with replacement
#         return [self.rng.choice(pool) for _ in range(k)]

#     def _make_epoch_batches(self):
#         self.batches = []
#         B, sps = self.B, self.sps
#         styles_per_batch = max(1, B // max(1, sps))

#         for _ in range(self._num_batches):
#             selected_styles = self._choose_styles_for_batch(styles_per_batch)

#             batch = []
#             for s in selected_styles:
#                 batch.extend(self._take_sps(s, sps))

#             # Trim or top-up to exact batch size using the full 100STYLE pool
#             if len(batch) > B:
#                 batch = batch[:B]
#             elif len(batch) < B:
#                 batch.extend(self.rng.choices(self.all_items, k=B - len(batch)))

#             self.batches.append(batch)

#     def __iter__(self):
#         # Rebuild and reshuffle every epoch to avoid staleness
#         self._make_epoch_batches()
#         self.rng.shuffle(self.batches)
#         return iter(self.batches)

#     def __len__(self):
#         return self._num_batches


# class TrainSampler(BatchSampler):
#     def __init__(self, config, dataset):
#         self.config = config
#         self.dataset = dataset
#         self.batch_size = config['batch_size']
#         self.samples_per_style = config['samples_per_style']

#         self.styles_to_motions = defaultdict(list)
#         for motion_idx, item in enumerate(dataset.items):
#             style_idx = item["style_idx"]
#             self.styles_to_motions[style_idx].append(motion_idx)

#         self.styles_to_frames = defaultdict(list)
#         cumsum = dataset.cumsum
#         for style_idx, motion_idcs in self.styles_to_motions.items():
#             for motion_idx in motion_idcs:
#                 start = cumsum[motion_idx]
#                 end   = cumsum[motion_idx + 1]
#                 if end > start:
#                     self.styles_to_frames[style_idx].append((start, end))

#         self.styles = [style for style, ranges in self.styles_to_frames.items() if ranges]
#         self.generate_batches()

#     def generate_batches(self):
#         self.batches = []
#         num_batches = len(self.dataset) // self.batch_size

#         for _ in range(num_batches):
#             styles_per_batch = self.batch_size // self.samples_per_style
#             selected_styles = random.sample(self.styles, styles_per_batch)

#             batch = []
#             for style in selected_styles:
#                 ranges = self.styles_to_frames[style]
#                 for _ in range(self.samples_per_style):
#                     r_start, r_end = random.choice(ranges)
#                     frame_idx = random.randrange(r_start, r_end)
#                     batch.append(frame_idx)
#             self.batches.append(batch)

#     def __iter__(self):
#         self.generate_batches()
#         random.shuffle(self.batches)
#         return iter(self.batches)

#     def __len__(self):
#         return len(self.batches)


class StyleSampler(BatchSampler):
    def __init__(self, config, dataset, target_style):
        self.config       = config
        self.dataset      = dataset
        self.batch_size   = config['batch_size']
        self.target_style = target_style

        # style_idx -> list of dataset indices (one index per motion)
        self.styles_to_motions = defaultdict(list)
        for motion_idx, item in enumerate(dataset.items):
            self.styles_to_motions[item["style_idx"]].append(motion_idx)

        # pool of indices for the requested style
        self.pool = self.styles_to_motions.get(self.target_style, [])
        if not self.pool:
            raise ValueError(f"No motions found for style_idx={self.target_style}")

        self.generate_batches()

    def generate_batches(self):
        self.batches = []
        B = self.batch_size
        # keep epoch size consistent with TrainSampler
        num_batches = max(1, len(self.dataset) // B)

        for _ in range(num_batches):
            # if enough motions in this style, sample without replacement; else with replacement
            if len(self.pool) >= B:
                batch = random.sample(self.pool, B)
            else:
                batch = random.choices(self.pool, k=B)
            self.batches.append(batch)

    def __iter__(self):
        self.generate_batches()
        random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# class StyleSampler(BatchSampler):
#     def __init__(self, config, dataset, target_style):
#         self.config = config
#         self.dataset = dataset
#         self.batch_size = config['batch_size']
#         self.target_style = target_style

#         self.styles_to_motions = defaultdict(list)
#         for motion_idx, item in enumerate(dataset.items):
#             style_idx = item["style_idx"]
#             self.styles_to_motions[style_idx].append(motion_idx)

#         self.styles_to_frames = defaultdict(list)
#         cumsum = dataset.cumsum
#         for style_idx, motion_idcs in self.styles_to_motions.items():
#             for motion_idx in motion_idcs:
#                 start = cumsum[motion_idx]
#                 end   = cumsum[motion_idx + 1]
#                 if end > start:
#                     self.styles_to_frames[style_idx].append((start, end))

#         if target_style not in self.styles_to_frames:
#             raise ValueError(f"No windows found for style_idx={target_style}")
#         self.styles = [target_style]
#         self.generate_batches()

#     def generate_batches(self):
#         self.batches = []
#         num_batches = len(self.dataset) // self.batch_size

#         for _ in range(num_batches):
#             batch = []
#             ranges = self.styles_to_frames[self.target_style]
#             for _ in range(self.batch_size):
#                 r_start, r_end = random.choice(ranges)
#                 frame_idx = random.randrange(r_start, r_end)
#                 batch.append(frame_idx)
#             self.batches.append(batch)

#     def __iter__(self):
#         self.generate_batches()
#         random.shuffle(self.batches)
#         return iter(self.batches)

#     def __len__(self):
#         return len(self.batches)


class TwoStyleSampler(BatchSampler):
    """
    Samples batches using ONLY the specified two style labels.
    - Random content (no constraints).
    - Balanced per batch as much as possible.
    - Works like your other samplers (builds frame indices per label).

    Config requirements:
      {
        "type": "TwoStyleSampler",
        "batch_size": <int>,
        "samples_per_class": <int>,
        "target_style_labels": [<neutral_label_id>, <angry_label_id>]
      }

    Notes:
    - For perfectly balanced batches, set batch_size == 2 * samples_per_class.
      If batch_size > 2*samples_per_class, the two labels are repeated to fill the batch.
    - We sample frames without replacement within a batch (like your other samplers).
      Ensure each chosen label has at least `samples_per_class` frames available.
    """
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.batch_size = int(config['batch_size'])
        self.samples_per_class = int(config['samples_per_class'])

        # Which two labels to use (IDs, not names)
        target = config.get('target_style_labels', None)
        if not target or len(target) != 2:
            raise ValueError(
                "TwoStyleSampler requires config['target_style_labels'] = [label_id_a, label_id_b]"
            )
        self.target_labels = list(target)

        # Build label -> motion indices
        self.label_to_motion_indices = defaultdict(list)
        for motion_idx, label in enumerate(dataset.labels):
            self.label_to_motion_indices[label].append(motion_idx)

        # Build label -> frame indices
        self.label_to_frame_indices = defaultdict(list)
        cumsum = dataset.cumsum  # assumes cumsum[i]..cumsum[i+1] = frames for motion i
        for label, motion_indices in self.label_to_motion_indices.items():
            if label not in self.target_labels:
                continue
            for m_idx in motion_indices:
                start = cumsum[m_idx]
                end   = cumsum[m_idx + 1]
                self.label_to_frame_indices[label].extend(range(start, end))

        # Keep only the target labels that actually have enough frames
        self.labels = [
            lbl for lbl in self.target_labels
            if len(self.label_to_frame_indices[lbl]) >= self.samples_per_class
        ]
        if len(self.labels) < 2:
            raise RuntimeError(
                "TwoStyleSampler: not enough frames for one or both target labels. "
                f"Have counts: {{lbl: len(frames) for lbl, frames in self.label_to_frame_indices.items()}}"
            )

        self.batches = []
        self.generate_batches()

    def _balanced_label_list(self, k: int):
        """
        Return a list of k labels, balanced between the two target labels.
        Example: target=[A,B], k=4 -> [A,B,A,B] (shuffled)
        """
        a, b = self.labels[:2]  # we validated we have exactly two usable labels
        q, r = divmod(k, 2)
        out = [a, b] * q + random.sample([a, b], r)
        random.shuffle(out)
        return out

    def generate_batches(self):
        self.batches = []
        num_batches = len(self.dataset) // self.batch_size
        num_labels_per_batch = self.batch_size // self.samples_per_class
        if num_labels_per_batch < 1:
            raise ValueError("batch_size must be >= samples_per_class")

        for _ in range(num_batches):
            # choose which label each group of samples_per_class will come from
            selected_labels = self._balanced_label_list(num_labels_per_batch)

            batch = []
            for lbl in selected_labels:
                # sample 'samples_per_class' distinct frames from that label
                pool = self.label_to_frame_indices[lbl]
                batch.extend(random.sample(pool, self.samples_per_class))

            self.batches.append(batch)

    def __iter__(self):
        self.generate_batches()
        random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class StyleContentSampler(BatchSampler):
    def __init__(self, config, dataset, fixed_label=51):
        self.config = config
        self.dataset = dataset
        self.batch_size = config['batch_size']
        self.samples_per_class = config['samples_per_class']
        self.num_labels_per_batch = self.batch_size // self.samples_per_class
        self.fixed_label = fixed_label

        self.content_to_style_to_indices = defaultdict(lambda: defaultdict(list))
        self.valid_contents = []

        # Build content → style → frame indices
        cumsum = dataset.cumsum
        for motion_idx, (motion_id, _) in enumerate(dataset.data):
            label = dataset.labels[motion_idx]
            content = dataset.ids_to_content[motion_id]
            start = cumsum[motion_idx]
            end = cumsum[motion_idx + 1]
            self.content_to_style_to_indices[content][label].extend(range(start, end))

        # Filter for valid contents that contain the fixed_label and enough other styles
        for content, style_dict in self.content_to_style_to_indices.items():
            if self.fixed_label not in style_dict:
                continue
            valid_other_styles = [
                s for s in style_dict if s != self.fixed_label and len(style_dict[s]) >= self.samples_per_class
            ]
            if len(style_dict[self.fixed_label]) >= self.samples_per_class and len(valid_other_styles) >= (self.num_labels_per_batch - 1):
                self.valid_contents.append(content)

        self.batches = []
        self.generate_batches()

    def generate_batches(self):
        self.batches = []
        num_batches = len(self.dataset) // self.batch_size

        for _ in range(num_batches):
            content = random.choice(self.valid_contents)
            style_dict = self.content_to_style_to_indices[content]

            # Always include the fixed label
            selected_styles = [self.fixed_label]
            candidate_styles = [
                s for s in style_dict if s != self.fixed_label and len(style_dict[s]) >= self.samples_per_class
            ]
            selected_styles += random.sample(candidate_styles, self.num_labels_per_batch - 1)

            # Sample from each selected style
            batch = []
            for style in selected_styles:
                batch.extend(random.sample(style_dict[style], self.samples_per_class))

            self.batches.append(batch)

    def __iter__(self):
        self.generate_batches()
        random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class RandomStyleSampler(BatchSampler):
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.batch_size = self.config['batch_size']
        self.samples_per_class = self.config['samples_per_class']

        self.label_to_motion_indices = defaultdict(list)
        for motion_idx, label in enumerate(dataset.labels):
            self.label_to_motion_indices[label].append(motion_idx)

        self.label_to_frame_indices = defaultdict(list)
        cumsum = dataset.cumsum
        for label, motion_indices in self.label_to_motion_indices.items():
            for m_idx in motion_indices:
                start = cumsum[m_idx]
                end = cumsum[m_idx + 1]
                self.label_to_frame_indices[label].extend(range(start, end))

        self.labels = [label for label in self.label_to_frame_indices if len(self.label_to_frame_indices[label]) >= self.samples_per_class]
        self.batches = []
        self.generate_batches()

    def generate_batches(self):
        self.batches = []
        num_batches = len(self.dataset) // self.batch_size
        num_labels_per_batch = self.batch_size // self.samples_per_class

        for _ in range(num_batches):
            selected_labels = random.sample(self.labels, num_labels_per_batch)
            batch = []
            for label in selected_labels:
                batch.extend(random.sample(self.label_to_frame_indices[label], self.samples_per_class))
            self.batches.append(batch)

    def __iter__(self):
        self.generate_batches()
        random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


SAMPLER_REGISTRY = {
    "TrainSampler": TrainSampler,
    "StyleSampler": StyleSampler,
    "TwoStyleSampler": TwoStyleSampler,
    "StyleContentSampler": StyleContentSampler,
    "RandomStyleSampler": RandomStyleSampler,
    # "MixedSampler": MixedSampler,
}