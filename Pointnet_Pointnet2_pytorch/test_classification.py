from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import json
import torch
import logging
from tqdm import tqdm
import sys
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser("Testing")
    parser.add_argument(
        "--use_cpu", action="store_true", default=False, help="use cpu mode"
    )
    parser.add_argument("--gpu", type=str, default="0", help="specify gpu device")
    parser.add_argument(
        "--batch_size", type=int, default=24, help="batch size in training"
    )
    parser.add_argument(
        "--num_category",
        default=40,
        type=int,
        choices=[10, 40],
        help="training on ModelNet10/40",
    )
    parser.add_argument("--num_point", type=int, default=1024, help="Point Number")
    parser.add_argument("--log_dir", type=str, required=True, help="Experiment root")
    parser.add_argument(
        "--use_normals", action="store_true", default=False, help="use normals"
    )
    parser.add_argument(
        "--use_uniform_sample",
        action="store_true",
        default=False,
        help="use uniform sampiling",
    )
    parser.add_argument(
        "--num_votes",
        type=int,
        default=3,
        help="Aggregate classification scores with voting",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="The path of completed point clouds",
    )
    return parser.parse_args()


def test(model, loader, num_class=40, vote_num=1):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))

    for j, (points, target, fn) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = (
                pred_choice[target == cat]
                .eq(target[target == cat].long().data)
                .cpu()
                .sum()
            )
            class_acc[cat, 0] += classacc.item() / float(
                points[target == cat].size()[0]
            )
            class_acc[cat, 1] += 1
        result = pred_choice.eq(target.long().data).cpu()
        correct = result.sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    cared_class_acc = []
    cared_class_ckeck = []
    for iidx, cared_obj in enumerate(class_acc):
        if cared_obj[1] == 0:
            continue
        cared_class_acc.append(cared_obj[0] / cared_obj[1])
        cared_class_ckeck.append([cared_obj[0] / cared_obj[1], iidx])

    print(cared_class_ckeck)
    cared_class_acc = sum(cared_class_acc) / len(cared_class_acc)
    instance_acc = np.mean(mean_correct)
    return instance_acc, cared_class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    experiment_dir = "log/classification/" + args.log_dir

    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler("%s/eval.txt" % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string("PARAMETER ...")
    log_string(args)

    log_string("Load dataset ...")
    data_path = args.data_path

    test_dataset = ModelNetDataLoader(
        root=data_path, args=args, split="test", process_data=False
    )
    testDataLoader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10
    )

    num_class = args.num_category
    model_name = os.listdir(experiment_dir + "/logs")[0].split(".")[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + "/checkpoints/best_model.pth")
    classifier.load_state_dict(checkpoint["model_state_dict"])
    times = 5
    InsAcc = []
    ClsAcc = []
    for time in range(times):
        with torch.no_grad():
            instance_acc, class_acc = test(
                classifier.eval(),
                testDataLoader,
                vote_num=args.num_votes,
                num_class=num_class,
            )
            log_string(
                "Test Instance Accuracy: %f, Class Accuracy: %f"
                % (instance_acc, class_acc)
            )
            InsAcc.append(instance_acc)
            ClsAcc.append(class_acc)

    log_string(
        "Final Test Instance Accuracy: %f, Final Class Accuracy: %f"
        % (np.mean(InsAcc), np.mean(ClsAcc))
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
