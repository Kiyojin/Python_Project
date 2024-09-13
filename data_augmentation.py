import random
import numpy as np
import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from dataset import ImagesDataset


def augment_image(img_np: np.ndarray, index: int) -> (torch.Tensor, str):
    v = index % 9

    # Transformation list
    transform_dict = {1: transforms.GaussianBlur(kernel_size=5),
                      2: transforms.RandomRotation(degrees=360),
                      3: transforms.RandomVerticalFlip(),
                      4: transforms.RandomHorizontalFlip(),
                      5: transforms.ColorJitter(),
                      6: transforms.RandomAffine(degrees=180, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                      7: transforms.RandomPerspective(),
                      }

    # Before creating a tensor from `img_np`, check if it's already a tensor
    if torch.is_tensor(img_np):
        img_tensor = img_np
    else:
        img_tensor = torch.from_numpy(img_np)

    # Transformation options
    if v == 0:
        transformation_name = "Original"
    elif v >= 1 and v <= 7:
        img_tensor = transform_dict[v](img_tensor)
        transformation_name = transform_dict[v].__class__.__name__
    else:
        random_transforms = [transform_dict[i] for i in random.sample(range(1, 8), 3)]
        compose = transforms.Compose(random_transforms)
        img_tensor = compose(img_tensor)
        transformation_name = "Compose"

    return img_tensor, transformation_name


class TransformedImagesDataset(Dataset):

    def __init__(self, data_set: Dataset):
        self.data_set = data_set

    def __getitem__(self, index: int):
        # Get index of dataset
        index_dataset = index // 9
        # Get image data and transform
        img_np, classid, classname, img_path = self.data_set[index_dataset]
        transformed_img, transform_name = augment_image(img_np, index)
        return transformed_img, transform_name, index, classid, classname, img_path

    def __len__(self):
        # *7 for every transformation option
        return len(self.data_set) * 9


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    dataset = ImagesDataset("./training_data", 100, 100, int)
    transformed_ds = TransformedImagesDataset(dataset)
    fig, axes = plt.subplots(4, 4)
    for i in range(0, 16):
        trans_img, trans_name, index, classid, classname, img_path = transformed_ds.__getitem__(i)
        _i = i // 4
        _j = i % 4
        axes[_i, _j].imshow(transforms.functional.to_pil_image(trans_img))
        axes[_i, _j].set_title(f'{trans_name}\n{classname}')

    fig.tight_layout()
    plt.show()
