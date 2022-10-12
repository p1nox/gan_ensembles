import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_dataset_dataloader(batch_size, image_size, dataset_path):
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(176),
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(dataset_path, transform)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    return dataset, data_loader


def scale(x, feature_range=(-1, 1)):
    min_val, max_val = feature_range
    x = x * (max_val - min_val) + min_val
    return x
