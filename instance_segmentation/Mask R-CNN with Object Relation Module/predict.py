import os
import time
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from model import AnchorsGenerator
from model import MaskRCNN
from draw_box_utils import draw_objs


def create_model(num_classes, box_thresh=0.5):
    import torchvision
    from torchvision.models.feature_extraction import create_feature_extractor

    # vgg16
    backbone = torchvision.models.vgg16_bn(pretrained=False)
    # print(backbone)
    backbone = create_feature_extractor(backbone, return_nodes={"features.42": "0"})
    # out = backbone(torch.rand(1, 3, 224, 224))
    # print(out["0"].shape)
    backbone.out_channels = 512

    # resnet50 backbone
    # backbone = torchvision.models.resnet50(pretrained=True)
    # # print(backbone)
    # backbone = create_feature_extractor(backbone, return_nodes={"layer3": "0"})
    # # out = backbone(torch.rand(1, 3, 224, 224))
    # # print(out["0"].shape)
    # backbone.out_channels = 1024

    # EfficientNetB0
    # backbone = torchvision.models.efficientnet_b0(pretrained=True)
    # # print(backbone)
    # backbone = create_feature_extractor(backbone, return_nodes={"features.5": "0"})
    # # out = backbone(torch.rand(1, 3, 224, 224))
    # # print(out["0"].shape)
    # backbone.out_channels = 112

    anchor_generator = AnchorsGenerator(sizes=((64, 96, 128),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行RoIAlign pooling
                                                    output_size=[7, 7],  # RoIAlign pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率
    mask_roi_pool = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"], output_size=14, sampling_ratio=2)

    model = MaskRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler,
                       mask_roi_pool=mask_roi_pool,
                       rpn_score_thresh=box_thresh,
                       rpn_nms_thresh=0.7,
                       box_score_thresh=box_thresh,
                       box_nms_thresh=0.5,
                       )

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    num_classes = 1  # 不包含背景
    box_thresh = 0.5
    weights_path = "./save_weights/model_29.pth"
    img_path = "./test.jpg"
    label_json_path = './coco_indices.json'

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    # load train weights
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    # load image
    assert os.path.exists(img_path), f"{img_path} does not exits."
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        predict_mask = predictions["masks"].to("cpu").numpy()
        predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")
            return

        plot_img = draw_objs(original_img,
                             boxes=predict_boxes,
                             classes=predict_classes,
                             scores=predict_scores,
                             masks=predict_mask,
                             category_index=category_index,
                             line_thickness=1,
                             font='arial.ttf',
                             font_size=2)
        plt.imshow(plot_img)
        plt.show()
        # 保存预测的图片结果
        plot_img.save("test_result.jpg")


if __name__ == '__main__':
    main()

