import os
import json
import torch
from PIL import Image
import torch.utils.data as data
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

def convert_coco_poly_mask(segmentations, height, width):
    """[nums_ins, H, W] 每个通道表示一个实例的掩膜"""
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]  # H,W,1

        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)  # H,W
        masks.append(mask)

    
    if masks:  # [H, W]*n_polys list -> [n_polys, H, W] Tensor
        masks = torch.stack(masks, dim=0)
    else:
        # 如果mask为空，则说明没有目标，直接返回[]。实际上没有目标的图片已经被去除
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

class DatasetsCOCO(data.Dataset):
    def __init__(self, root, dataset="train", transforms=None):
        super(DatasetsCOCO, self).__init__()
        assert dataset in ["train", "val"], 'dataset must be in ["train", "val"]'
        assert os.path.exists(root), "file '{}' does not exist.".format(root)
        anno_file = f"anno_{dataset}.json"  # COCO标注文件
        img_file = dataset  # 图片文件夹 train or test
        self.img_path = os.path.join(root, img_file)  # root/train
        self.anno_path = os.path.join(root, "annotations", anno_file)  # root/annotations/anno_train.json
        self.mode = dataset  # train_set or test_set
        self.transforms = transforms
        self.coco = COCO(self.anno_path)

        #  "categories": [{"supercategory": null, "id": 0, "name": "_background_"}, {"supercategory": null, "id": 1, "name": "PN"}]
        # {0: '_background_', 1: 'PN'}
        data_classes = dict([(v["id"], v["name"]) for k, v in self.coco.cats.items()]) # 
        max_index = max(data_classes.keys())
        coco_classes = {}
        for k in range(1, max_index + 1):  # 不包括'_background_'
            coco_classes[k] = data_classes[k]

        if dataset == "train":
            json_str = json.dumps(coco_classes, indent=4)
            with open("coco_indices.json", "w") as f:
                f.write(json_str)
        
        self.coco_classes = coco_classes

        self.ids = list(sorted(self.coco.imgs.keys()))  # 图片索引

    def parse_targets(self,
                      img_id: int,
                      coco_targets: list,
                      w: int = None,
                      h: int = None):
        assert w > 0
        assert h > 0

        # 只筛选出单个对象的情况  不用密集目标进行训练？
        # coco_targets: 一张图片内的目标 list[dict]
        anno = [obj for obj in coco_targets if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]  # [num_coco_targets, 4]

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # [xmin, ymin, w, h] -> [xmin, ymin, xmax, ymax]
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]  # [num_coco_targets]
        classes = torch.tensor(classes, dtype=torch.int64)

        area = torch.tensor([obj["area"] for obj in anno])  # [num_coco_targets]
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])  # [num_coco_targets]

        segmentations = [obj["segmentation"] for obj in anno]  # [[[]],[[]],...]
        masks = convert_coco_poly_mask(segmentations, h, w)

        # 筛选出合法的目标，即x_max>x_min且y_max>y_min
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]  # [num_coco_targets, 4]
        classes = classes[keep]  # [num_coco_targets]
        masks = masks[keep]  # [n_polys, H, W] num_coco_targets=n_polys
        area = area[keep]
        iscrowd = iscrowd[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])

        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # [{'id','image_id','category_id','segmentation','area','bbox','iscrowd'},{...}] 相同image_id
        coco_target = coco.loadAnns(ann_ids)  # json下半部分
        path = coco.loadImgs(img_id)[0]['file_name']  # json上半部分
        # path = path.replace("\\", "/") 
        img = Image.open(os.path.join(self.img_path, path)).convert('RGB')

        w, h = img.size
        target = self.parse_targets(img_id, coco_target, w, h)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def get_height_and_width(self, index):
        coco = self.coco
        img_id = self.ids[index]

        img_info = coco.loadImgs(img_id)[0]
        w = img_info["width"]
        h = img_info["height"]
        return h, w

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

if __name__ == '__main__':
    train = DatasetsCOCO(".\PN_instance_COCO_datasets", dataset="train")
    train.__getitem__(0)