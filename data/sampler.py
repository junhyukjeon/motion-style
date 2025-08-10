import random
from collections import defaultdict
from torch.utils.data import BatchSampler

class StyleSampler(BatchSampler):
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

    def generate_batches(self, fixed_label=51): # 51 is "Neutral"
        self.batches = []
        num_batches = len(self.dataset) // self.batch_size

        for _ in range(num_batches):
            # Step 1: Choose one fixed label if provided
            if fixed_label is not None:
                selected_labels = [fixed_label]
                candidate_labels = [l for l in self.labels if l != fixed_label]
            else:
                selected_labels = []
                candidate_labels = self.labels[:]

            # Step 2: Randomly pick the remaining labels
            num_labels_per_batch = self.batch_size // self.samples_per_class
            selected_labels += random.sample(candidate_labels, num_labels_per_batch - len(selected_labels))

            # Step 3: Sample frames for each label
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
    "StyleSampler": StyleSampler,
    "StyleContentSampler": StyleContentSampler,
    "RandomStyleSampler": RandomStyleSampler,
}