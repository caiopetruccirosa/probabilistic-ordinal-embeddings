import os
import torch

from PIL import Image
from torchvision import transforms
from typing import Any, Optional, cast
from torch.utils.data import Dataset

class AdienceAgeEstimationDataset(Dataset):
    def __init__(self, data_root_dir: str, annotations: list[dict[str, Any]], transform: Optional[transforms.Compose] = None):
        self.data_root_dir = data_root_dir
        self.annotations = annotations
        self.transform = transform if transform is not None else transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.annotations[idx]

        img_filepath = os.path.join(self.data_root_dir, sample['user_id'], f"landmark_aligned_face.{sample['face_id']}.{sample['original_image']}")
        age_label    = sample['age']

        img = Image.open(img_filepath)
        if img.mode == 'L':
            img = img.convert('RGB')

        img = self.transform(img)
        img = cast(torch.Tensor, img)

        return img, age_label

    def __len__(self):
        return len(self.annotations)