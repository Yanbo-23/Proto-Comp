import torch
import torch.nn as nn
import os
import json
from torch import nn, optim
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
import numpy as np
import random
from models.point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from models.point_e.diffusion.sampler import PointCloudSampler
from models.point_e.models.download import load_checkpoint
from models.point_e.models.configs import MODEL_CONFIGS, model_from_config
from models.point_e.util.common import get_linear_scheduler
import models.point_e.util.builder as builder
from tools.data_transforms import PointcloudRotate
import open3d as o3d
from tqdm.auto import tqdm
from extensions.chamfer_dist import ChamferDistanceL1_PM

os.environ["OMP_NUM_THREADS"] = "4"


def savePC(partial, samples, sample_gt, prompt, path):

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(samples[0].detach().cpu().numpy())
    point_cloud_gt = o3d.geometry.PointCloud()

    point_cloud_gt.points = o3d.utility.Vector3dVector(
        sample_gt[0].detach().cpu().numpy()
    )
    point_cloud_partial = o3d.geometry.PointCloud()
    point_cloud_partial.points = o3d.utility.Vector3dVector(
        partial[0].detach().cpu().numpy()
    )
    if not os.path.exists("vis_" + path):
        os.mkdir("vis_" + path)
    o3d.io.write_point_cloud(os.path.join("vis_" + path, "templete-2.pcd"), point_cloud)
    o3d.io.write_point_cloud(
        os.path.join("vis_" + path, "templete-0.pcd"), point_cloud_gt
    )
    o3d.io.write_point_cloud(
        os.path.join("vis_" + path, "templete-1.pcd"), point_cloud_partial
    )
    np.savetxt(
        os.path.join("vis_" + path, "pred.txt"),
        samples[0].detach().cpu().numpy(),
        delimiter=";",
    )
    np.savetxt(
        os.path.join("vis_" + path, "part.txt"),
        partial[0].detach().cpu().numpy(),
        delimiter=";",
    )
    np.savetxt(
        os.path.join("vis_" + path, "gt.txt"),
        sample_gt[0].detach().cpu().numpy(),
        delimiter=";",
    )
    with open(os.path.join("vis_" + path, "prompt.txt"), "w") as file:
        file.write(prompt[0])

    return


def run_net(args, config, train_writer=None, val_writer=None):

    with open(config.dataset.promptdict_path) as json_file:
        prompt_dict = json.load(json_file)

    logger = get_logger(args.log_name)
    rotate_transform = PointcloudRotate()
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(
        args, config.dataset.train
    ), builder.dataset_builder(args, config.dataset.val)
    device = torch.device(f"cuda:{args.local_rank}")
    start_epoch = 0
    best_metrics = None
    metrics = None
    opt_config = config.opt_config
    sched_config = config.sched_config

    base_model = model_from_config(
        MODEL_CONFIGS[config.model.backbone_model], config.model, device
    )

    base_diffusion = diffusion_from_config(
        DIFFUSION_CONFIGS[config.model.backbone_model]
    )

    base_model.load_state_dict(load_checkpoint(args.init_weights, device), strict=False)

    if args.use_gpu:
        base_model.to(args.local_rank)

    if args.resume:
        start_epoch, best_metrics = builder.resume_model(
            base_model, args, logger=logger
        )
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)

    for name, param in base_model.named_parameters():
        if "control" not in name:
            param.requires_grad = False
            print(["FREZE!", name])
        else:
            print(["Traning!", name])

    print_log("Trainable_parameters:", logger=logger)
    print_log("=" * 25, logger=logger)

    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log("=" * 25, logger=logger)

    print_log("Untrainable_parameters:", logger=logger)
    print_log("=" * 25, logger=logger)
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log("=" * 25, logger=logger)

    if args.distributed:

        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log("Using Synchronized BatchNorm ...", logger=logger)
        base_model = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[args.local_rank % torch.cuda.device_count()],
            find_unused_parameters=True,
        )
        print_log("Using Distributed Data parallel ...", logger=logger)
    else:
        print_log("Using Data parallel ...", logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    if opt_config["type"] == "adamw":
        opt = optim.AdamW(
            base_model.parameters(),
            lr=opt_config["lr"],
            betas=tuple(opt_config["betas"]),
            eps=opt_config["eps"],
            weight_decay=opt_config["weight_decay"],
        )
    elif opt_config["type"] == "sgd":
        opt = optim.SGD(
            base_model.parameters(),
            lr=opt_config["lr"],
            momentum=opt_config.get("momentum", 0.0),
            nesterov=opt_config.get("nesterov", False),
            weight_decay=opt_config.get("weight_decay", 0.0),
        )

    if sched_config["type"] == "linear":
        scheduler = get_linear_scheduler(
            opt,
            start_epoch=0,
            end_epoch=config.max_epoch,
            start_lr=opt_config["lr"],
            end_lr=sched_config["min_lr"],
        )
    else:
        assert False, "sched type not support"
    sampler = PointCloudSampler(
        device=device,
        models=[base_model],
        diffusions=[base_diffusion],
        num_points=[1024],
        aux_channels=["R", "G", "B"],
        guidance_scale=[3.0],
        karras_steps=[64],
        sigma_min=[1e-3],
        sigma_max=[120],
        s_churn=[3],
        use_karras=[True],
        model_kwargs_key_filter=["texts"],
    )
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(["Loss"])
        num_iter = 0

        base_model.train()
        n_batches = len(train_dataloader)
        times = 0

        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            times += 1
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            if (
                "Merged" in dataset_name
                or "PCN" in dataset_name
                or dataset_name == "Completion3D"
                or "ProjectShapeNet" in dataset_name
            ):
                partial = data[0].cuda()
                gt = data[1].cuda()
                if "PCN" in dataset_name:
                    if config.dataset.train._base_.CARS:
                        if idx == 0:
                            print_log("padding while KITTI training", logger=logger)
                        partial = misc.random_dropping(partial, epoch)
                gt = gt[:, :, [0, 2, 1]]
                partial = partial[:, :, [0, 2, 1]]

            elif "ShapeNet" in dataset_name:
                gt = data.cuda()

                partial, _ = misc.seprate_point_cloud(
                    gt,
                    npoints,
                    [int(npoints * 1 / 4), int(npoints * 3 / 4)],
                    fixed_points=None,
                )
                partial = partial.cuda()
                gt = gt[:, :, [0, 2, 1]]
                partial = partial[:, :, [0, 2, 1]]
            else:
                raise NotImplementedError(f"Train phase do not support {dataset_name}")

            num_iter += 1

            rotation_angle = np.random.randint(1, 5, size=gt.size()[0])
            angle_text = [config.angle_dict[str(i)] for i in rotation_angle]
            prompt = []
            for idx__, ids in enumerate(taxonomy_ids):
                if torch.rand(1).item() > 0.5:
                    prompt.append(
                        angle_text[idx__] + prompt_dict[ids] + angle_text[idx__]
                    )
                else:
                    prompt.append("")

            rotation_angle = rotation_angle / 4 * 2 * np.pi

            gt = rotate_transform(gt.clone(), rotation_angle)
            sample_gt = misc.fps(gt, 1024)
            zeros_tensor = torch.zeros_like(sample_gt)
            sample_gt_in = torch.cat((sample_gt, zeros_tensor), dim=2)

            partial = rotate_transform(partial.clone(), rotation_angle)
            _losses = sampler.loss_texts(partial, sample_gt_in, prompt, gt.shape[0])

            _loss = _losses
            _loss.backward()
            if args.distributed:
                _losses = dist_utils.reduce_tensor(_losses, args)
                losses.update([_losses.item()])
            else:
                losses.update([_losses.item()])

            if num_iter == config.step_per_update:
                torch.nn.utils.clip_grad_norm_(
                    base_model.parameters(),
                    getattr(config, "grad_norm_clip", 10),
                    norm_type=2,
                )
                num_iter = 0
                opt.step()
                base_model.zero_grad()

            if args.distributed:
                torch.cuda.synchronize()
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if (idx + 1) % 100 == 0:
                print_log(
                    "[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f"
                    % (
                        epoch,
                        config.max_epoch,
                        idx + 1,
                        n_batches,
                        batch_time.val(),
                        data_time.val(),
                        ["%.4f" % l for l in losses.val()],
                        opt.param_groups[0]["lr"],
                    ),
                    logger=logger,
                )

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        epoch_end_time = time.time()

        print_log(
            "[Training] EPOCH: %d EpochTime = %.3f (s)"
            % (epoch, epoch_end_time - epoch_start_time),
            logger=logger,
        )

        if (epoch + 1) % config.val_fequency == 0:

            metrics = validate(
                base_model,
                test_dataloader,
                epoch,
                val_writer,
                args,
                config,
                sampler,
                prompt_dict,
                rotate_transform,
                logger=logger,
            )

            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(
                    base_model,
                    opt,
                    epoch,
                    metrics,
                    best_metrics,
                    "ckpt-best",
                    args,
                    logger=logger,
                )
            builder.save_checkpoint(
                base_model,
                opt,
                epoch,
                metrics,
                best_metrics,
                "ckpt-last",
                args,
                logger=logger,
            )
        if (config.max_epoch - epoch) < 2:
            builder.save_checkpoint(
                base_model,
                opt,
                epoch,
                metrics,
                best_metrics,
                f"ckpt-epoch-{epoch:03d}",
                args,
                logger=logger,
            )
    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()


def validate(
    base_model,
    test_dataloader,
    epoch,
    val_writer,
    args,
    config,
    sampler,
    prompt_dict,
    rotate_transform,
    logger=None,
):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    base_model.eval()

    test_losses = AverageMeter(["testLoss"])
    test_metrics = AverageMeter(["CD1"])
    category_metrics = dict()
    n_samples = len(test_dataloader)

    all_numbers = list(range(1, len(test_dataloader)))

    counter = 0
    random_numbers = sorted(random.sample(all_numbers, 30))

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            if (idx + 1) in random_numbers:
                taxonomy_id = (
                    taxonomy_ids[0]
                    if isinstance(taxonomy_ids[0], str)
                    else taxonomy_ids[0].item()
                )
                model_id = model_ids[0]

                npoints = config.dataset.val._base_.N_POINTS
                dataset_name = config.dataset.val._base_.NAME
                if "PCN" in dataset_name:
                    partial = data[0].cuda()
                    gt = data[1].cuda()
                    gt = gt[:, :, [0, 2, 1]]
                    partial = partial[:, :, [0, 2, 1]]

                elif "ShapeNet" in dataset_name:
                    gt = data.cuda()
                    partial, _ = misc.seprate_point_cloud(
                        gt,
                        npoints,
                        [int(npoints * 1 / 4), int(npoints * 3 / 4)],
                        fixed_points=None,
                    )
                    partial = partial.cuda()
                    gt = gt[:, :, [0, 2, 1]]
                    partial = partial[:, :, [0, 2, 1]]
                else:
                    raise NotImplementedError(
                        f"Train phase do not support {dataset_name}"
                    )

                rotation_angle = np.random.randint(1, 5, size=gt.size()[0])

                angle_text = [config.angle_dict[str(i)] for i in rotation_angle]
                prompt = []
                for idx__, ids in enumerate(taxonomy_ids):
                    if torch.rand(1).item() > 0.5:
                        prompt.append(
                            angle_text[idx__] + prompt_dict[ids] + angle_text[idx__]
                        )
                    else:
                        prompt.append("")
                rotation_angle = rotation_angle / 4 * 2 * np.pi

                gt = rotate_transform(gt.clone(), rotation_angle)
                partial = rotate_transform(partial.clone(), rotation_angle)

                sample_gt = misc.fps(gt, 1024)

                for x in tqdm(
                    sampler.sample_batch_progressive(
                        partial=partial,
                        batch_size=len(taxonomy_ids),
                        model_kwargs=dict(texts=prompt),
                    )
                ):
                    samples = x
                    samples = samples[:, :3, :].transpose(1, 2)

                zeros_tensor = torch.zeros_like(sample_gt)
                sample_gt_in = torch.cat((sample_gt, zeros_tensor), dim=2)
                _losses = sampler.loss_texts(partial, sample_gt_in, prompt, gt.shape[0])

                if args.distributed:
                    _losses = dist_utils.reduce_tensor(_losses, args)
                    test_losses.update([_losses.item() * 1000])
                else:
                    test_losses.update([_losses.item() * 1000])

                _metrics = Metrics.get(samples, gt)
                if args.distributed:
                    _metrics = [
                        dist_utils.reduce_tensor(_metric, args).item()
                        for _metric in _metrics
                    ]
                else:
                    _metrics = [_metric.item() for _metric in _metrics]

                for _taxonomy_id in taxonomy_ids:
                    if _taxonomy_id not in category_metrics:
                        category_metrics[_taxonomy_id] = AverageMeter(["CD1"])
                    category_metrics[_taxonomy_id].update(_metrics)
                if (counter + 1) % 15 == 0:
                    print_log(
                        "Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s"
                        % (
                            idx + 1,
                            n_samples,
                            taxonomy_id,
                            model_id,
                            ["%.4f" % l for l in test_losses.val()],
                            ["%.4f" % m for m in _metrics],
                        ),
                        logger=logger,
                    )
                counter += 1
        for _, v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log(
            "[Validation] EPOCH: %d  Metrics = %s"
            % (epoch, ["%.4f" % m for m in test_metrics.avg()]),
            logger=logger,
        )

        if args.distributed:
            torch.cuda.synchronize()

    shapenet_dict = json.load(open("./data/shapenet_synset_dict.json", "r"))
    print_log(
        "============================ TEST RESULTS ============================",
        logger=logger,
    )
    msg = ""
    msg += "Taxonomy\t"
    msg += "Sample\t"
    for metric in test_metrics.items:
        msg += metric + "\t"
    msg += "ModelName\t"
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ""
        msg += taxonomy_id + "\t"
        msg += str(category_metrics[taxonomy_id].count(0)) + "\t"
        for value in category_metrics[taxonomy_id].avg():
            msg += "%.3f \t" % value
        msg += shapenet_dict[taxonomy_id] + "\t"
        print_log(msg, logger=logger)

    msg = ""
    msg += "Overall\t\t"
    for value in test_metrics.avg():
        msg += "%.3f \t" % value

    print_log(msg, logger=logger)

    return Metrics(config.consider_metric, test_metrics.avg())


def test_net(args, config):

    logger = get_logger(args.log_name)
    print_log("Tester start ... ", logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = model_from_config(
        MODEL_CONFIGS["base40M-textvec"], config.model, device=args.local_rank
    )

    builder.load_model(base_model, args.ckpts, logger=logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    if args.distributed:
        raise NotImplementedError()

    ChamferDisL1_PM = ChamferDistanceL1_PM()

    device = args.local_rank
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["base40M-textvec"])
    sampler = PointCloudSampler(
        device=device,
        models=[base_model],
        diffusions=[base_diffusion],
        num_points=[1024],
        aux_channels=["R", "G", "B"],
        guidance_scale=[3.0],
        karras_steps=[64],
        sigma_min=[1e-3],
        sigma_max=[120],
        s_churn=[3],
        use_karras=[True],
        model_kwargs_key_filter=["texts"],
    )

    test(
        base_model,
        test_dataloader,
        ChamferDisL1_PM,
        args,
        config,
        sampler,
        logger=logger,
    )


def test(
    base_model,
    test_dataloader,
    ChamferDisL1_PM,
    args,
    config,
    sampler,
    logger=None,
):

    base_model.eval()
    cd1 = 0
    with torch.no_grad():
        for insta_idx, (category, partail_scannet, path) in enumerate(test_dataloader):

            category = category[0]
            path = path[0]

            dataset_name = config.dataset.test._base_.NAME

            if "RealSensor" in dataset_name:

                prompt_gt = [0]
                prompt = [0]

                partail_scannet = partail_scannet.to(torch.float32).cuda()

                partial = partail_scannet

                prompt_gt[0] = "A " + category + "."

                prompt[0] = "A " + category + "."

                path_tosave = path.replace(
                    os.path.basename(os.path.dirname(os.path.dirname(path))),
                    args.save_path,
                )

                partial = misc.fps(partial, 2048)

                if os.path.exists(path_tosave):
                    print("Already exists", path_tosave)
                    point_cloud = o3d.io.read_point_cloud(path_tosave)

                    points = point_cloud.points

                    result = torch.tensor(points, dtype=torch.float32).to(
                        partial.device
                    )
                    result = torch.unsqueeze(result, dim=0)

                    cd1 += ChamferDisL1_PM(partial, result)
                    print("CD_L1: ", cd1 / (insta_idx + 1))
                    print([category, insta_idx, len(test_dataloader)])
                    continue

                for x in tqdm(
                    sampler.sample_batch_progressive(
                        partial=partial,
                        batch_size=1,
                        model_kwargs=dict(texts=prompt),
                    )
                ):
                    samples = x
                    samples = samples[:, :3, :].transpose(1, 2)
                print("PRED: ", prompt, "GT: ", prompt_gt)
                cd1 += ChamferDisL1_PM(partial, samples)

                os.makedirs(os.path.dirname(path_tosave), exist_ok=True)

                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(
                    samples[0].detach().cpu().numpy()
                )
                o3d.io.write_point_cloud(path_tosave, point_cloud)

                print([category, insta_idx, len(test_dataloader)])
                print("CD_L1: ", cd1 / (insta_idx + 1))

            else:
                raise NotImplementedError(f"Test phase do not support {dataset_name}")

    print_log(
        "============================ TEST DONE! ============================",
        logger=logger,
    )
    print_log(f"CD_L1: {cd1*1000 / len(test_dataloader):.1f}", logger)
    print_log(
        f"The completed point clouds have been save in {args.save_path}: , cd1 / len(test_dataloader)",
        logger,
    )

    return
