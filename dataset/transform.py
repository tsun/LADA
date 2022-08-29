from torchvision import transforms
from PIL import Image
from .randaugment import RandAugment
from torchvision.transforms import (Resize, Compose, ToTensor, Normalize, CenterCrop, RandomCrop,
                                    RandomResizedCrop, RandomHorizontalFlip)

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


def build_transforms(cfg, domain='source'):
    if cfg.DATASET.NAME in ["mnist", "svhn"]:
        transforms = None
    else:
        # train
        choices = cfg.DATASET.SOURCE_TRANSFORMS if domain=='source' else cfg.DATASET.TARGET_TRANSFORMS
        train_transform = build_transform(choices)
        # query
        choices = cfg.DATASET.QUERY_TRANSFORMS
        query_transform = build_transform(choices)
        # test
        choices = cfg.DATASET.TEST_TRANSFORMS
        test_transform = build_transform(choices)

        transforms = {'train':train_transform, 'query':query_transform, 'test':test_transform}

    return transforms

def build_transform(choices):
    transform = []
    if 'Resize' in choices:
        transform += [Resize((256, 256))]  # make sure resize to equal length and width

    if 'ResizeImage' in choices:
        transform += [ResizeImage(256)]

    if 'RandomHorizontalFlip' in choices:
        transform += [RandomHorizontalFlip(p=0.5)]

    if 'RandomCrop' in choices:
        transform += [RandomCrop(224)]

    if 'RandomResizedCrop' in choices:
        transform += [RandomResizedCrop(224)]

    if 'CenterCrop' in choices:
        transform += [CenterCrop(224)]


    transform += [ToTensor()]

    if 'Normalize' in choices:
        normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    transform += [normalize]

    return Compose(transform)


rand_transform = transforms.Compose([
    RandAugment(1, 2.0),
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
