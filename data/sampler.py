import random
from collections import defaultdict
from torch.utils.data import BatchSampler

class StyleSampler(BatchSampler):
    def __init__(self, dataset, batch_size, samples_per_class):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class

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

        self.labels = [label for label in self.label_to_frame_indices if len(self.label_to_frame_indices[label]) >= samples_per_class]
        self.batches = []
        self.generate_batches()

    def generate_batches(self):
        self.batches = []
        num_batches = len(self.dataset) // self.batch_size
        for _ in range(num_batches):
            selected_labels = random.sample(self.labels, self.batch_size // self.samples_per_class)
            batch = []
            for label in selected_labels:
                batch.extend(random.sample(self.label_to_frame_indices[label], self.samples_per_class))
            self.batches.append(batch)

    def __iter__(self):
        # Optional: shuffle cached batches
        random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

    def set_epoch(self, epoch=None):
        self.generate_batches()