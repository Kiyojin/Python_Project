import torch
from torch.utils.data import DataLoader, random_split

from data_augmentation import TransformedImagesDataset
from dataset import ImagesDataset


def stacking(batch_as_list: list):
    # Unpack elements in the list
    trans_img, trans_name, index, class_id, class_name, img_path = zip(*batch_as_list)
    # Stack images along first dimension
    trans_img = torch.stack(trans_img, dim=0)
    # Reshape indices and class_ids into a [N, 1] tensor
    index = torch.tensor(index, dtype=torch.long).unsqueeze(1)
    class_id = torch.tensor(class_id, dtype=torch.long).unsqueeze(1)
    return trans_img, list(trans_name), index, class_id, list(class_name), list(img_path)

def get_dataset(validation_split=0.2):
    # Create dataset
    dataset = ImagesDataset("./training_data", 100, 100, int)
    transformed_ds = TransformedImagesDataset(dataset)

    # Determine sizes
    total_size = len(transformed_ds)
    eval_size = int(validation_split * total_size)
    train_size = total_size - eval_size

    # Split dataset
    train_ds, eval_ds = random_split(transformed_ds, [train_size, eval_size])
    return train_ds, eval_ds

if __name__ == "__main__":
    dataset = ImagesDataset("./training_data", 100, 100, int)
    transformed_ds = TransformedImagesDataset(dataset)
    dl = DataLoader(transformed_ds, batch_size=7, shuffle=False, collate_fn=stacking)
    for i, (images, trans_names, indices, classids, classnames, img_paths) in enumerate(dl):
        print(f'mini batch: {i}')
        print(f'images shape: {images.shape}')
        print(f'trans_names: {trans_names}')
        print(f'indices: {indices}')
        print(f'class ids: {classids}')
        print(f'class names: {classnames}\n')
