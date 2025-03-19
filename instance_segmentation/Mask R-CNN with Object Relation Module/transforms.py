import random
from torchvision.transforms import functional as F
import torchvision
import math
import torch


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        return image, target

class RandomRotation(torchvision.transforms.RandomRotation):
    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
        super().__init__(degrees, resample=resample, expand=expand, center=center, fill=fill)

    def __call__(self, img, target):
        angle = self.get_params(self.degrees)

        # Rotate the image
        img = F.rotate(img[None], angle, self.resample, self.expand, self.center, self.fill)[0]

        # Rotate the masks
        masks = target["masks"]
        masks = F.rotate(masks[None], angle, self.resample, self.expand, self.center, self.fill)[0]
        target["masks"] = masks

        # Rotate the bounding boxes
        rot_bbox = []
        for mask in masks:
            xmin = torch.min(torch.where(mask)[1])
            ymin = torch.min(torch.where(mask)[0])
            xmax = torch.max(torch.where(mask)[1])
            ymax = torch.max(torch.where(mask)[0])
            mask_bbox = [xmin, ymin, xmax, ymax]
            rot_bbox.append(mask_bbox)
            
        target["boxes"] = torch.as_tensor(rot_bbox)

        return img, target
