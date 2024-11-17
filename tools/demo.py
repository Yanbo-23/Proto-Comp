import torch
import os
from utils.logger import *
import numpy as np
import open3d as o3d
from models.point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from models.point_e.diffusion.sampler import PointCloudSampler
from models.point_e.models.configs import MODEL_CONFIGS, model_from_config
import models.point_e.util.builder as builder
from tqdm.auto import tqdm
from extensions.chamfer_dist import ChamferDistanceL1_PM
from utils import misc
from datasets.io import IO

os.environ["OMP_NUM_THREADS"] = "4"


def pc_norm(pc):
    """pc: NxC, return NxC"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1))) * 2
    pc = pc / m
    return pc


def savePC(partial, samples, save_path):

    np.savetxt(
        os.path.join(save_path, "demo.txt"),
        samples[0].detach().cpu().numpy(),
        delimiter=";",
    )
    np.savetxt(
        os.path.join(save_path, "input.txt"),
        partial[0].detach().cpu().numpy(),
        delimiter=";",
    )


def demo_net(args, config):
    assert args.pc_path is not None
    assert args.save_path is not None
    assert args.ckpts is not None
    assert args.config is not None
    assert args.prompt is not None

    logger = get_logger(args.log_name)
    print_log("Tester start ... ", logger=logger)
    base_model = model_from_config(
        MODEL_CONFIGS["base40M-textvec"], config.model, device=args.local_rank
    )

    builder.load_model(base_model, args.ckpts, logger=logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    if args.distributed:
        raise NotImplementedError()

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

    demo(
        base_model,
        args,
        sampler,
        logger=logger,
    )


def demo(
    base_model,
    args,
    sampler,
    logger=None,
):

    base_model.eval()
    pc_path = args.pc_path
    data = IO.get(pc_path).astype(np.float32)
    data = pc_norm(data)
    data = torch.from_numpy(data).float()

    prompt = [0]

    partial = data.to(torch.float32).cuda().unsqueeze(0)

    prompt[0] = args.prompt

    path_tosave = args.save_path

    partial = misc.fps(partial, 2048)

    for x in tqdm(
        sampler.sample_batch_progressive(
            partial=partial,
            batch_size=1,
            model_kwargs=dict(texts=prompt),
        )
    ):
        samples = x
        samples = samples[:, :3, :].transpose(1, 2)
    print("PROMPT: ", prompt)

    os.makedirs(path_tosave, exist_ok=True)

    savePC(partial, samples, path_tosave)
