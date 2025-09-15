import random
from collections import defaultdict
from torch.utils.data import BatchSampler


class TrainSampler(BatchSampler):
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.batch_size = config['batch_size']
        self.samples_per_style = config['samples_per_style']

        self.styles_to_motions = defaultdict(list)
        for motion_idx, item in enumerate(dataset.items):
            style_idx = item["style_idx"]
            self.styles_to_motions[style_idx].append(motion_idx)

        self.styles_to_frames = defaultdict(list)
        cumsum = dataset.cumsum
        for style_idx, motion_idcs in self.styles_to_motions.items():
            for motion_idx in motion_idcs:
                start = cumsum[motion_idx]
                end   = cumsum[motion_idx + 1]
                if end > start:
                    self.styles_to_frames[style_idx].append((start, end))

        self.styles = [style for style, ranges in self.styles_to_frames.items() if ranges]
        self.generate_batches()

    def generate_batches(self):
        self.batches = []
        num_batches = len(self.dataset) // self.batch_size

        for _ in range(num_batches):
            styles_per_batch = self.batch_size // self.samples_per_style
            selected_styles = random.sample(self.styles, styles_per_batch)

            batch = []
            for style in selected_styles:
                ranges = self.styles_to_frames[style]
                for _ in range(self.samples_per_style):
                    r_start, r_end = random.choice(ranges)
                    frame_idx = random.randrange(r_start, r_end)
                    batch.append(frame_idx)
            self.batches.append(batch)

    def __iter__(self):
        self.generate_batches()
        random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class StyleSampler(BatchSampler):
    def __init__(self, config, dataset, target_style):
        self.config = config
        self.dataset = dataset
        self.batch_size = config['batch_size']
        self.target_style = target_style

        self.styles_to_motions = defaultdict(list)
        for motion_idx, item in enumerate(dataset.items):
            style_idx = item["style_idx"]
            self.styles_to_motions[style_idx].append(motion_idx)

        self.styles_to_frames = defaultdict(list)
        cumsum = dataset.cumsum
        for style_idx, motion_idcs in self.styles_to_motions.items():
            for motion_idx in motion_idcs:
                start = cumsum[motion_idx]
                end   = cumsum[motion_idx + 1]
                if end > start:
                    self.styles_to_frames[style_idx].append((start, end))

        if target_style not in self.styles_to_frames:
            raise ValueError(f"No windows found for style_idx={target_style}")
        self.styles = [target_style]
        self.generate_batches()

    def generate_batches(self):
        self.batches = []
        num_batches = len(self.dataset) // self.batch_size

        for _ in range(num_batches):
            batch = []
            ranges = self.styles_to_frames[self.target_style]
            for _ in range(self.batch_size):
                r_start, r_end = random.choice(ranges)
                frame_idx = random.randrange(r_start, r_end)
                batch.append(frame_idx)
            self.batches.append(batch)

    def __iter__(self):
        self.generate_batches()
        random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# class StyleSampler(BatchSampler):
#     def __init__(self, config, dataset):
#         self.config = config
#         self.dataset = dataset
#         self.batch_size = self.config['batch_size']
#         self.samples_per_class = self.config['samples_per_class']

#         self.label_to_motion_indices = defaultdict(list)
#         for motion_idx, label in enumerate(dataset.labels):
#             self.label_to_motion_indices[label].append(motion_idx)

#         self.label_to_frame_indices = defaultdict(list)
#         cumsum = dataset.cumsum
#         for label, motion_indices in self.label_to_motion_indices.items():
#             for m_idx in motion_indices:
#                 start = cumsum[m_idx]
#                 end   = cumsum[m_idx + 1]
#                 self.label_to_frame_indices[label].extend(range(start, end))

#         self.labels = [label for label in self.label_to_frame_indices if len(self.label_to_frame_indices[label]) >= self.samples_per_class]
#         self.batches = []
#         self.generate_batches()

#     def generate_batches(self, fixed_label=51): # 51 is "Neutral"
#         self.batches = []
#         num_batches = len(self.dataset) // self.batch_size

#         for _ in range(num_batches):
#             # Step 1: Choose one fixed label if provided
#             if fixed_label is not None:
#                 selected_labels = [fixed_label]
#                 candidate_labels = [l for l in self.labels if l != fixed_label]
#             else:
#                 selected_labels = []
#                 candidate_labels = self.labels[:]

#             # Step 2: Randomly pick the remaining labels
#             num_labels_per_batch = self.batch_size // self.samples_per_class
#             selected_labels += random.sample(candidate_labels, num_labels_per_batch - len(selected_labels))

#             # Step 3: Sample frames for each label
#             batch = []
#             for label in selected_labels:
#                 batch.extend(random.sample(self.label_to_frame_indices[label], self.samples_per_class))

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
}