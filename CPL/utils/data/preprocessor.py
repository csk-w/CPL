# CCC/utils/preprocessor.py

from torch.utils.data import Dataset
from PIL import Image


class Preprocessor(Dataset):
    def __init__(self, dataset, transform=None, load_img=True):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.load_img = load_img

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        # === START MODIFICATION: Handle extended dataset format ===

        item = self.dataset[indices]
        img_path, pid, cam = item[0], item[1], item[2]

        is_gen = item[3] if len(item) > 3 else 0
        quality_score = item[4] if len(item) > 4 else 1.0
        source_path = item[5] if len(item) > 5 else img_path
        # === END MODIFICATION ===

        if self.load_img:
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

            # === START MODIFICATION: Return all metadata ===
            return img, pid, cam, is_gen, quality_score, source_path
            # === END MODIFICATION ===

        return pid, cam