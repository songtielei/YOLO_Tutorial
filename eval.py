import argparse
import os

from copy import deepcopy
import torch

from evaluator.voc_evaluator import VOCAPIEvaluator
from evaluator.coco_evaluator import COCOAPIEvaluator
from evaluator.ourdataset_evaluator import OurDatasetEvaluator

# load transform
from dataset.build import build_transform

# load some utils
from utils.misc import load_weight
from utils.misc import compute_flops

from config import build_dataset_config, build_model_config, build_trans_config
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-Tutorial')
    # basic
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='the max size of input image')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')

    # model
    parser.add_argument('-m', '--model', default='yolov1', type=str,
                        help='build yolo')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-ct', '--conf_thresh', default=0.005, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.6, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=1000, type=int,
                        help='topk candidates for testing')
    parser.add_argument("--no_decode", action="store_true", default=False,
                        help="not decode in inference or yes")
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse Conv & BN')

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc.')
    parser.add_argument('--mosaic', default=None, type=float,
                        help='mosaic augmentation.')
    parser.add_argument('--mixup', default=None, type=float,
                        help='mixup augmentation.')
    parser.add_argument('--load_cache', action='store_true', default=False,
                        help='load data into memory.')

    # TTA
    parser.add_argument('-tta', '--test_aug', action='store_true', default=False,
                        help='use test augmentation.')

    return parser.parse_args()


def voc_test(model, data_dir, device, transform):
    evaluator = VOCAPIEvaluator(data_dir=data_dir,
                                device=device,
                                transform=transform,
                                display=True)

    # VOC evaluation
    evaluator.evaluate(model)

def coco_test(model, data_dir, device, transform, test=False):
    if test:
        # test-dev
        print('test on test-dev 2017')
        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        device=device,
                        testset=True,
                        transform=transform)

    else:
        # eval
        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        device=device,
                        testset=False,
                        transform=transform)

    # COCO evaluation
    evaluator.evaluate(model)

def our_test(model, data_dir, device, transform):
    evaluator = OurDatasetEvaluator(
        data_dir=data_dir,
        device=device,
        image_set='val',
        transform=transform)

    # WiderFace evaluation
    evaluator.evaluate(model)


if __name__ == '__main__':
    args = parse_args()
    # 如果args.cuda为True，则使用GPU来推理，否则使用CPU来训练（可接受）
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 构建测试所用到的 Dataset & Model & Transform相关的config变量
    data_cfg = build_dataset_config(args)
    model_cfg = build_model_config(args)
    trans_cfg = build_trans_config(model_cfg['trans_type'])
    
    data_dir = os.path.join(args.root, data_cfg['data_name'])
    num_classes = data_cfg['num_classes']

    # 构建YOLO模型
    model = build_model(args, model_cfg, device, num_classes, False)

    # 加载已训练好的模型权重
    model = load_weight(model, args.weight, args.fuse_conv_bn)
    model.to(device).eval()

    # 计算模型的参数量和FLOPs
    model_copy = deepcopy(model)
    model_copy.trainable = False
    model_copy.eval()
    compute_flops(
        model=model_copy,
        img_size=args.img_size, 
        device=device)
    del model_copy

    # 构建测试所用到的数据预处理Transform类
    val_transform, trans_cfg = build_transform(args, trans_cfg, model_cfg['max_stride'], is_train=False)

    # 开始在指定的数据集上去测试我们的代码
    # 对于使用VOC数据集训练出来的模型，就使用VOC测试集来做测试
    # 对于使用COCO数据集训练出来的模型，就使用COCO验证机来做测试
    with torch.no_grad():
        if args.dataset == 'voc':
            voc_test(model, data_dir, device, val_transform)
        elif args.dataset == 'coco-val':
            coco_test(model, data_dir, device, val_transform, test=False)
        elif args.dataset == 'coco-test':
            coco_test(model, data_dir, device, val_transform, test=True)
        elif args.dataset == 'ourdataset':
            our_test(model, data_dir, device, val_transform)
