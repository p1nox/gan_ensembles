import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class ImageFolderWithPaths(datasets.ImageFolder):
    '''Custom dataset that includes image file paths.
        Extends torchvision.datasets.ImageFolder
        Src: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d'''
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def get_dataset_dataloader(batch_size, image_size, dataset_path):
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(176),
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor()
    ])

    dataset = ImageFolderWithPaths(dataset_path, transform)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    return dataset, data_loader
