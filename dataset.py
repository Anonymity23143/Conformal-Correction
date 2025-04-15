from torch.utils.data import Dataset
from typing import Any, Tuple
from PIL import Image


class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc):
        self.datasets = datasets
        self.transformFunc = transformFunc

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.datasets[index][0], self.datasets[index][1]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.transformFunc is not None:
            img = self.transformFunc(img)

        return img, target

    def __len__(self) -> int:
        return len(self.datasets)
