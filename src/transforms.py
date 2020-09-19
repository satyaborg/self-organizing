from torchvision import transforms, utils

def mnist_transforms(crop_size: int=20):
    """Transformations for MNIST
    Crops images: 28x28 -> 20x20
    """
    transform = transforms.Compose([
                transforms.CenterCrop(crop_size), 
                transforms.ToTensor() 
            ])
    return dict(train=transform, test=transform)

def cifar_transforms(augment: bool=True, channels: int=1, num_output_channels: int=1):
    """Transformations for CIFAR10
    Comments by KP:
    Convert CIFAR's RGB image to Grayscale, as color unimportant for us.
    For 20x20 locs=hidden units in Autoenc's, put padding of 4 on each side.
    For 16x16 locs=hidden units in Autoenc's, no padding.
    
    For CIFAR-10 (with normalization + standard data augmentation):
    Standard augmentation techniques as per the following papers are applied:
    (McDonnell, ICLR 2018; Huang etc ICLR 2017; Lin et al., 2013; Romero et al., 2014; 
    Lee et al., 2015; Springenberg et al., 2014; Srivastava et al., 2015; 
    Huang et al., 2016b; Larsson et al.,2016), in which the images are 
    zero-padded with 4 pixels on each side, randomly cropped to produce
    32×32 images, and horizontally mirrored with probability 0.5.

    SBG: Try 3 channels, no grayscale
    """
    if channels > 1:
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
    else:
        transform = transforms.Compose([
                        transforms.Grayscale(num_output_channels=num_output_channels),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ])
    # image augmentations
    transform_aug = transforms.Compose([
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomCrop(32, padding=4),
                        transform
                    ])
    return dict(train=transform_aug if augment else transform, test=transform)


