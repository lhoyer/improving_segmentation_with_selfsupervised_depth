import argparse
import json
import math
import os
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter  # causes omp infos in contrast to tensorboardX
from tqdm import tqdm

from configs.machine_config import MachineConfig
from evaluation.metrics import runningScore
from experiments import decoder_variant
from loader.cityscapes_loader import Cityscapes
from loss.loss import pixel_wise_entropy, berhu
from models import get_model
from train import expand_cfg_vars, Trainer
from utils.utils import get_logger, np_local_seed


def label_selection_main(cfg):
    MachineConfig(cfg["machine"])
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

    if "name" not in cfg:
        cfg["name"] = "test" + run_id
    cfg['training']['log_path'] += cfg["name"]
    name = cfg['name']
    print('Start', name)

    expand_cfg_vars(cfg)

    log_dir = cfg['training']['log_path']

    print("RUNDIR: {}".format(log_dir))
    os.makedirs(log_dir)
    with open(log_dir + "/cfg.yml", 'w') as fp:
        yaml.dump(cfg, fp)

    remove_models = cfg["label_selection"].get("remove_models", True)
    train_from_scratch = cfg["label_selection"]["train_from_scratch"]
    last_from_scratch = cfg["label_selection"].get("last_from_scratch", train_from_scratch)
    label_steps = cfg["label_selection"]["label_steps"]
    train_iters_per_step = [int(v) for v in cfg["label_selection"]["train_iters"]]
    selection_tasks = cfg["label_selection"]["selection_tasks"]
    choice = cfg["label_selection"]["choice"]
    initial_samples = cfg["label_selection"]["initial_samples"]
    assert choice in ["score", "ifp", "random"]
    assert initial_samples in ["random", "ifp"]
    assert selection_tasks in ["depth", "seg", "seg+depth"]
    if train_from_scratch:
        train_iters_until_step = train_iters_per_step
    else:
        train_iters_until_step = [sum(train_iters_per_step[0:i + 1]) for i in range(len(train_iters_per_step))]
    if choice in ["ifp", "random"]:
        assert last_from_scratch or train_from_scratch
        if sum([v for k, v in cfg["label_selection"].items() if "lambda" in k]) == 0:
            label_steps = [label_steps[-1]]
            train_iters_per_step = [train_iters_per_step[-1]]
            train_iters_until_step = [train_iters_until_step[-1]]
    resume_step, resume_file = cfg["label_selection"].get("resume", (-1, ""))
    print(train_iters_until_step)
    assert len(label_steps) == len(train_iters_per_step)
    if "max_iter" in cfg['training']['lr_schedule']:
        assert cfg['training']['lr_schedule']['max_iter'] == train_iters_until_step[-1]
        assert train_from_scratch

    model_file = None
    labeled_samples = []
    if resume_file != "":
        model_file = resume_file
        with open(f"{os.path.dirname(resume_file)}_subset.json", 'r') as fp:
            labeled_samples = json.load(fp)
        print(f"LABEL_SELECTION: Resume at step {resume_step} from {resume_file} with samples {labeled_samples}")
    for i, (n_new_subset, train_iters) in enumerate(zip(label_steps, train_iters_until_step)):
        is_last_step = (i == len(label_steps) - 1)
        if i < resume_step:
            continue
        if i == 0:
            labeled_samples = choose_initial_samples(cfg, n_new_subset, mode=initial_samples)
        else:
            print(f"LABEL_SELECTION: Evaluate model {model_file}")
            labeled_samples = choose_new_samples(cfg, model_file, labeled_samples, n_new_subset, choice)

        if train_iters == 0:
            continue
        print(f"LABEL_SELECTION: Train on {len(labeled_samples)} samples: {labeled_samples}")
        current_cfg = deepcopy(cfg)
        old_model_file = model_file
        model_file_to_continue = old_model_file
        if not is_last_step:
            current_cfg['training']['val_interval'] = {"0": 4000}
        if train_from_scratch or (is_last_step and last_from_scratch):
            model_file_to_continue = None
        if selection_tasks == "depth" and not is_last_step:
            current_cfg['training']['segmentation_lambda'] = 0
        if selection_tasks == "seg" and not is_last_step:
            current_cfg['training']['pseudo_depth_lambda'] = 0
            current_cfg['training']['monodepth_lambda'] = 0
        if is_last_step and cfg["label_selection"].get("last_segmentation_only", False):
            current_cfg['training']['pseudo_depth_lambda'] = 0
            current_cfg['training']['monodepth_lambda'] = 0
        if is_last_step and cfg["label_selection"].get("last_depth_only", False):
            current_cfg['training']['segmentation_lambda'] = 0
        model_file = train_on_subset(current_cfg, labeled_samples, train_iters, model_file_to_continue,
                                     tensorboard_in_subdir=train_from_scratch or last_from_scratch)
        # If tensorboard_in_subdir is wrong: # find . -type f -wholename "*nlabels*/*.metrics" -execdir mv -t ../ {} +
        if remove_models and old_model_file is not None and old_model_file != resume_file:
            os.remove(old_model_file)
    if remove_models and model_file is not None and model_file != resume_file:
        os.remove(model_file)


def train_on_subset(base_cfg, labeled_samples, train_iters, model_file=None, tensorboard_in_subdir=True):
    base_log_dir = base_cfg["training"]["log_path"]
    cfg = deepcopy(base_cfg)

    cfg['data']['restrict_to_subset'] = {
        "mode": "fixed",
        "n_subset": len(labeled_samples),
        "subset": labeled_samples,
    }
    cfg['training']['train_iters'] = train_iters
    if 'max_iter' in cfg['training']['lr_schedule']:
        cfg['training']['lr_schedule']['max_iter'] = train_iters

    if model_file is not None:
        cfg["training"]["resume"] = model_file

    experiment_name = f"nlabels{len(labeled_samples)}"
    with open(os.path.join(base_log_dir, f"{experiment_name}_subset.json"), 'w') as fp:
        json.dump(labeled_samples, fp)
    trainer = build_trainer(cfg, experiment_name, tensorboard_in_subdir)
    last_step = trainer.train()
    model_file = trainer.save_resume(last_step)

    return model_file


is_first_trainer = True


def build_trainer(cfg, experiment_name, tensorboard_in_subdir=True):
    global is_first_trainer
    cfg = deepcopy(cfg)
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + cfg["general"]["tag"]
    run_id = experiment_name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_log_dir = cfg['training']['log_path']
    log_dir = f"{base_log_dir}/{experiment_name}/"
    os.makedirs(log_dir, exist_ok=True)

    cfg["name"] = name
    cfg["training"]["log_path"] = log_dir
    cfg['training']['disable_depth_estimator'] = not is_first_trainer or cfg['training'].get('disable_depth_estimator', False)

    if tensorboard_in_subdir:
        writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'{experiment_name}.metrics')
        img_writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'{experiment_name}.tensorboardimgs')
    else:
        writer = SummaryWriter(log_dir=base_log_dir, filename_suffix=f'{experiment_name}.metrics')
        img_writer = SummaryWriter(log_dir=base_log_dir, filename_suffix=f'{experiment_name}.tensorboardimgs')
    logger = get_logger(log_dir)
    with open(log_dir + "/cfg.yml", 'w') as fp:
        yaml.dump(cfg, fp)

    is_first_trainer = False

    return Trainer(cfg, writer, img_writer, logger, os.path.join(name, str(run_id)))


def choose_initial_samples(cfg, n, mode):
    if mode == "random":
        with np_local_seed(cfg["seed"]):
            p = np.random.permutation(get_n_total(cfg))
        return p[:n].tolist()
    elif mode == "ifp":
        with np_local_seed(cfg["seed"]):
            p = np.random.permutation(get_n_total(cfg))
        seed_sample = p[:1].tolist()
        ifp_cfg = deepcopy(cfg)
        ifp_cfg["label_selection"].update({
            "preselection_multiplier": None,
            "bias_weight": 0,
        })
        labeled_samples = choose_new_samples(ifp_cfg, None, seed_sample, n, "ifp")
        return labeled_samples
    else:
        raise NotImplementedError(mode)


def choose_samples_from_scores(scores, n_to_add):
    if isinstance(scores[0]['label_criterion'], list):
        n_criteria = len(scores[0]['label_criterion'])
        n_per_criterion = n_to_add // n_criteria
        chosen_samples, chosen_scores = [], []
        assert n_criteria * n_per_criterion == n_to_add
        for i in range(n_criteria):
            sorted_scores = sorted(scores, key=lambda k: k['label_criterion'][i], reverse=True)
            for s in sorted_scores:
                if s["idx"].item() not in chosen_samples:
                    s["used_label_criterion"] = f"C{i}_{s['label_criterion'][i]:.4f}"
                    s["depth_error"] = s["depth_error"][i]
                    if "depth_error_map" in s:
                        s["depth_error_map"] = s["depth_error_map"][i]
                    chosen_samples.append(s["idx"].item())
                    chosen_scores.append(s)
                if len(chosen_samples) >= (i + 1) * n_per_criterion:
                    break
    else:
        scores = sorted(scores, key=lambda k: k['label_criterion'], reverse=True)
        chosen_scores = scores[:n_to_add]
        for i in range(len(chosen_scores)):
            chosen_scores[i]['used_label_criterion'] = f"{chosen_scores[i]['label_criterion']:.4f}"
        chosen_samples = [s["idx"].item() for s in chosen_scores]

    return chosen_samples, chosen_scores


def choose_samples_from_ifp(initial_samples, scores, feature_distances, n_to_add, preselection_multiplier):
    assert len(scores[0]['label_criterion']) == 1
    preselected_samples = None
    if preselection_multiplier is not None:
        assert preselection_multiplier > 0
        presorted_scores = sorted(scores, key=lambda k: k['label_criterion'][0], reverse=True)
        preselected_samples = [s["idx"].item() for s in presorted_scores[:int(preselection_multiplier * n_to_add)]]
        print("LABEL_SELECTION: Preselected samples:", preselected_samples)
    idxs, ifp_distances = iterative_farthest_point(initial_samples, feature_distances, n_to_add,
                                                   preselected_samples)
    chosen_samples, chosen_scores = [], []
    for i, dist in zip(idxs, ifp_distances):
        if preselection_multiplier is not None:
            assert i in preselected_samples
        for s in scores:
            if s["idx"] == i:
                s.update({
                    "label_criterion": dist,
                    "used_label_criterion": f"{dist:.4f}",
                    "iterative_farthest_distance": dist,
                    "depth_error": s["depth_error"][0],
                })
                if "depth_error_map" in s:
                    s["depth_error_map"] = s["depth_error_map"][0]
                chosen_samples.append(i)
                chosen_scores.append(s)
    assert len(chosen_scores) == n_to_add

    return chosen_samples, chosen_scores


def choose_new_samples(cfg, model_file, current_samples, n_new_subset, choice, debug=True):
    n_to_add = n_new_subset - len(current_samples)
    assert n_to_add > 0
    n_random_choice_all = cfg["label_selection"].get("n_random_choice_all", get_n_total(cfg))
    preselection_multiplier = cfg["label_selection"]["preselection_multiplier"]
    with np_local_seed(cfg["seed"]):
        all_samples = np.random.permutation(np.arange(get_n_total(cfg)))[:n_random_choice_all].tolist()
    unlabeled_samples = [v for v in all_samples if v not in current_samples]
    # print(f"Unlabeled samples {sorted(unlabeled_samples)}")
    if choice in ["ifp"]:
        scores, feat_distances = acquire_scores(cfg, unlabeled_samples, all_samples, model_file,
                                                 depth_ifp_w=cfg["label_selection"]["depth_ifp_weight"])
        if preselection_multiplier is not None:
            assert sum([v for k, v in cfg["label_selection"].items() if "lambda" in k]) != 0
        chosen_samples, chosen_scores = choose_samples_from_ifp(current_samples, scores, feat_distances, n_to_add,
                                                                preselection_multiplier)
    elif choice == "score":
        # If all entropy lambdas are zero, we assume random mode
        if not isinstance(cfg['label_selection']['entropy_lambda'], list) and \
                sum([v for k, v in cfg["label_selection"].items() if "lambda" in k]) == 0:
            chosen_samples = unlabeled_samples[:n_to_add]
        else:
            scores, _ = acquire_scores(cfg, unlabeled_samples, all_samples, model_file)
            chosen_samples, chosen_scores = choose_samples_from_scores(scores, n_to_add)
    else:
        raise NotImplementedError(choice)
    print(f"Old samples {sorted(current_samples)}")
    print(f"New samples {sorted(chosen_samples)}")
    new_subset = deepcopy(current_samples)
    new_subset.extend(chosen_samples)
    assert len(new_subset) == n_new_subset
    assert len(new_subset) == len(set(new_subset)), f"Subset contains duplicates: {sorted(new_subset)}"

    if debug:
        logs, _ = acquire_scores(cfg, chosen_samples, all_samples, model_file, verbose=True)
        if choice in ["ifp"]:
            _, logs = choose_samples_from_ifp(current_samples, logs, feat_distances, n_to_add, preselection_multiplier)
        else:
            _, logs = choose_samples_from_scores(logs, n_to_add)
        rows, cols = 3, 3
        log_dir = os.path.join(os.path.join(cfg["training"]["log_path"], f"new_labels_{n_new_subset}"))
        os.makedirs(log_dir)
        for i, log in enumerate(logs):
            fig, axs = plt.subplots(rows, cols, sharex='col', sharey='row',
                                    gridspec_kw={'hspace': 0, 'wspace': 0},
                                    figsize=(3 * cols * 2, 3 * rows))
            axs[0][0].imshow(log["image"].permute(1, 2, 0).cpu().numpy())
            axs[0][1].imshow(log["disparity"][0].cpu().numpy(), cmap="plasma_r")
            axs[0][2].imshow(log["teacher_depth"][0].cpu().numpy(), cmap="plasma_r")
            axs[1][2].imshow(log["depth_error_map"].cpu().numpy(), cmap="plasma")
            axs[1][0].imshow(Cityscapes.decode_segmap_tocolor(log["segmentation_pred"]))
            axs[1][1].imshow(Cityscapes.decode_segmap_tocolor(log["segmentation_gt"]))
            axs[2][0].imshow(log["segmentation_entropy"].cpu().numpy(), cmap="viridis")
            # axs[2][1].imshow(log["reprojection_error_map"].cpu().numpy(), cmap="plasma")
            for ax in axs.flat:
                ax.axis("off")
            plt.savefig(os.path.join(log_dir, f"new_label_{i}_{log['used_label_criterion']}.jpg"))
            plt.close()

    return new_subset


def build_depth_trainer_model(cfg):
    cfg = deepcopy(cfg)
    cfg["model"]["arch"] = "joint_segmentation_depth"
    cfg["model"].update(cfg["monodepth_options"])
    cfg["model"]["segmentation_name"] = None
    cfg["model"]["disable_monodepth"] = False
    cfg["model"]["disable_pose"] = True
    if cfg["data"].get("depth_teacher", None) is not None:
        cfg['model']['backbone_name'] = "resnet101"
        cfg, load_backbone = decoder_variant(cfg, 6, (512, 512))
        cfg['model']['depth_pretraining'] = cfg["data"]["depth_teacher"]
        cfg['model']['backbone_pretraining'] = cfg["data"]["depth_teacher"]

    assert cfg['model']['depth_pretraining'] == cfg['model']['backbone_pretraining']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(cfg["model"], Cityscapes.n_classes).to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model

def dilate(input, kernel_size, padding):
    pad_int = int(padding)
    assert pad_int == padding
    input = input.unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=input.device)
    return torch.clamp(torch.nn.functional.conv2d(input, kernel, padding=(pad_int, pad_int)), 0, 1).squeeze(0).squeeze(
        0)

def acquire_scores(base_cfg, samples_to_score, all_samples, model_file, depth_ifp_w=0, verbose=False):
    calc_depth_distances = depth_ifp_w > 0
    depth_lambda = base_cfg["label_selection"]["depth_lambda"]
    entropy_lambda = base_cfg["label_selection"]["entropy_lambda"]
    dist_bias_weight = base_cfg["label_selection"]["bias_weight"]
    ifp_args = base_cfg["label_selection"]["ifp_args"]
    if not verbose:
        if isinstance(depth_lambda, list):
            for dl, el in zip(depth_lambda, entropy_lambda):
                assert dl + el > 0
        else:
            assert depth_lambda + entropy_lambda > 0 or calc_depth_distances

    if calc_depth_distances and ifp_args["m"] in ["aspp", "u4", "u3", "bn"]:
        depth_teacher = build_depth_trainer_model(base_cfg)

    cfg = deepcopy(base_cfg)
    cfg['data']['augmentations'] = {}
    cfg['monodepth_options'].pop('crop_h')
    cfg['monodepth_options'].pop('crop_w')
    cfg['training']['batch_size'] = 1
    cfg['data']['shuffle_trainset'] = False
    restrict_subset = all_samples if calc_depth_distances else samples_to_score
    cfg['data']['restrict_to_subset'] = {
        "mode": "fixed",
        "n_subset": len(restrict_subset),
        "subset": restrict_subset,
    }
    cfg["training"]["resume"] = model_file

    trainer = build_trainer(cfg, "label_selection_scoring")
    if cfg["training"]["resume"] is not None:
        trainer.load_resume(strict=True, load_model_only=True)
    else:
        print("LABEL_SELECTION: Warning - Evaluated model is None. This might happen when using ifp.")

    scores = []
    all_depth_features = []
    dist_i_to_img_idx = {}
    img_idx_to_dist_i = {}
    dist_bias = []
    trainer.model.eval()
    with torch.no_grad():
        depth_loss_mask = None
        for inputs in tqdm(trainer.train_data_loader):
            for k, v in inputs.items():
                cuda_tensor_names = [("color_aug", 0, 0), "pseudo_depth"]
                if verbose:
                    cuda_tensor_names.extend(["lbl", ("color", 0, 0)])
                if torch.is_tensor(v) and k in cuda_tensor_names:
                    inputs[k] = v.to(trainer.device, non_blocking=True)

            if calc_depth_distances:
                if ifp_args["pool"] == "avg":
                    pool_fn = torch.nn.functional.adaptive_avg_pool2d
                elif ifp_args["pool"] == "max":
                    pool_fn = torch.nn.functional.adaptive_max_pool2d
                else:
                    raise NotImplementedError(ifp_args["pool"])
                if ifp_args["m"] in ["aspp", "u3", "u4", "bn"]:
                    teacher_outputs = depth_teacher(inputs)
                    if ifp_args["m"] == "u3":
                        depth_features = teacher_outputs[("upconv", 3)]
                    elif ifp_args["m"] == "u4":
                        depth_features = teacher_outputs[("upconv", 4)]
                    elif ifp_args["m"] == "bn":
                        depth_features = teacher_outputs["bottleneck"]
                    else:
                        raise NotImplementedError(ifp_args["m"])
                    depth_features = pool_fn(depth_features, (ifp_args["h"], 2 * ifp_args["h"]))
                elif ifp_args["m"] == "logdepth":
                    depth_features = inputs["pseudo_depth"][0]
                    depth_features = torch.log(torch.clamp(1 / depth_features, 0.1, 80))
                    depth_features = pool_fn(depth_features, (ifp_args["h"], 2 * ifp_args["h"]))
                    depth_features.unsqueeze_(0)
                elif ifp_args["m"] == "depth":
                    depth_features = inputs["pseudo_depth"][0]
                    depth_features = torch.clamp(1 / depth_features, 0.1, 80)
                    depth_features = pool_fn(depth_features, (ifp_args["h"], 2 * ifp_args["h"]))
                    depth_features.unsqueeze_(0)
                else:
                    raise NotImplementedError(ifp_args["m"])
                assert depth_features.shape[0] == 1
                dist_i_to_img_idx[len(all_depth_features)] = inputs["idx"].item()
                img_idx_to_dist_i[inputs["idx"].item()] = len(all_depth_features)
                all_depth_features.append(depth_features.detach())
                if not verbose and dist_bias_weight == 0:
                    scores.append({
                        "idx": inputs["idx"],
                        "label_criterion": [0],
                        "depth_error": [0],
                        "entropy_mean": 0,
                    })
                    continue

            with autocast(enabled=trainer.cfg["training"]["amp"]):
                outputs = trainer.model(inputs)

            if inputs["idx"] not in samples_to_score:
                dist_bias.append(0)
                continue

            entropy_imgs = pixel_wise_entropy(outputs["semantics"])
            disp_pred = outputs["disp", 0][0][0]
            disp_pseudo = inputs["pseudo_depth"][0][0]

            depth_error_maps = []
            depth_errors = []
            depth_error_types = cfg["label_selection"].get("depth_error_types", "abs")
            if not isinstance(depth_error_types, list):
                depth_error_types = [depth_error_types]
            for depth_error_type in depth_error_types:
                if depth_error_type == "abs":
                    depth_error_map = torch.abs(disp_pred - disp_pseudo)
                elif depth_error_type == "abs_inv_log":
                    depth_pred = torch.log(torch.clamp(1 / disp_pred, 0.1, 80))
                    depth_pseudo = torch.log(torch.clamp(1 / disp_pseudo, 0.1, 80))
                    depth_error_map = torch.abs(depth_pseudo - depth_pred)
                elif depth_error_type == "abs_inv":
                    depth_pred = torch.clamp(1 / disp_pred, 0.1, 80)
                    depth_pseudo = torch.clamp(1 / disp_pseudo, 0.1, 80)
                    depth_error_map = torch.abs(depth_pseudo - depth_pred)
                elif depth_error_type == "sq":
                    depth_error_map = (disp_pred - disp_pseudo) ** 2
                elif depth_error_type == "abs_rel":
                    depth_error_map = torch.abs(disp_pred - disp_pseudo) / (disp_pseudo + 1e-1)
                elif depth_error_type == "sq_rel":
                    depth_error_map = ((disp_pred - disp_pseudo) ** 2) / (disp_pseudo + 1e-1)
                elif depth_error_type == "abs_log":
                    depth_error_map = torch.abs(torch.log(1 + disp_pred) - torch.log(1 + disp_pseudo))
                else:
                    raise NotImplementedError(depth_error_type)

                # Mask out cars moving in front with very small disparity
                mask = dilate((disp_pseudo < 0.07).float(), 7, 3)
                depth_error_map *= (1 - mask)
                # Mask out own car
                depth_error_map[int(0.87 * depth_error_map.shape[0]):, :] = 0
                depth_error = torch.mean(depth_error_map)
                depth_error_maps.append(depth_error_map.detach())
                depth_errors.append(depth_error.detach())
            entropy_mean = torch.mean(entropy_imgs[0])

            assert not (isinstance(depth_lambda, list) and len(depth_error_types) > 1)
            if isinstance(depth_lambda, list):
                label_criterion = []
                for dl, el in zip(depth_lambda, entropy_lambda):
                    label_criterion.append((dl * depth_error + el * entropy_mean).detach())
                    depth_error_maps.append(depth_error_map)
                    depth_errors.append(depth_error)
            elif isinstance(depth_error_types, list):
                label_criterion = []
                for depth_error in depth_errors:
                    label_criterion.append((depth_lambda * depth_error + entropy_lambda * entropy_mean).detach())
            else:
                label_criterion = (depth_lambda * depth_error + entropy_lambda * entropy_mean).detach()
            if dist_bias_weight > 0:
                assert len(label_criterion) == 1
                dist_bias.append(dist_bias_weight * label_criterion[0])

            scores.append({
                "idx": inputs["idx"],
                "label_criterion": label_criterion,
                "depth_error": depth_errors,
                "entropy_mean": entropy_mean.detach(),
            })

            if verbose:
                segmentation_loss = trainer.loss_fn(
                    input=outputs["semantics"], target=inputs["lbl"],
                    pixel_weights=None
                )

                preds = outputs["semantics"].data.max(1)[1].cpu().numpy()
                gts = inputs["lbl"].data.cpu().numpy()

                for k, v in outputs.items():
                    if "depth" in k or "cam_T_cam" in k:
                        outputs[k] = v.to(torch.float32)
                # trainer.monodepth_loss_calculator_train.generate_images_pred(all_inputs, outputs)
                # mono_losses = trainer.monodepth_loss_calculator_train.compute_losses(all_inputs, outputs)
                # mono_loss = mono_losses["loss"]
                mono_loss = torch.tensor([0])
                mono_outputs = trainer.model.predict_test_disp(inputs)
                trainer.monodepth_loss_calculator_val.generate_depth_test_pred(mono_outputs)

                # Crop away bottom of image with own car
                if depth_loss_mask is None:
                    depth_loss_mask = torch.ones(outputs["disp", 0].shape, device=trainer.device)
                    depth_loss_mask[:, :, int(outputs["disp", 0].shape[2] * 0.9):, :] = 0
                pseudo_depth_loss = berhu(outputs["disp", 0], inputs["pseudo_depth"], depth_loss_mask)

                running_metrics_val = runningScore(trainer.n_classes)
                running_metrics_val.update(gts, preds)
                score, class_iou = running_metrics_val.get_scores()

                scores[-1].update({
                    "image": inputs["color_aug", 0, 0][0].detach().cpu(),
                    "segmentation_entropy": entropy_imgs[0].detach().cpu(),
                    "disparity": torch.log(torch.clamp(1 / outputs["disp", 0][0], 0.1, 80)).detach().cpu(),
                    "teacher_depth": torch.log(torch.clamp(1 / inputs["pseudo_depth"][0], 0.1, 80)).detach().cpu(),
                    "depth_error_map": depth_error_maps,
                    "mIoU": score["Mean IoU : \t"],
                    "fwAcc": score["FreqW Acc : \t"],
                    "mAcc": score["Mean Acc : \t"],
                    "tAcc": score["Overall Acc: \t"],
                    "cIoU": class_iou,
                    "segmentation_loss": segmentation_loss.item(),
                    "mono_loss": mono_loss.item(),
                    "pseudo_depth_loss": pseudo_depth_loss.item(),
                    "segmentation_pred": preds[0],
                    "segmentation_gt": gts[0],
                    # "reprojection_error_map": outputs["to_optimise/0"][0].detach().cpu(),
                })

        depth_feature_distances = 0
        if calc_depth_distances:
            depth_feature_distances = _calc_feature_distance(all_depth_features, dist_bias, dist_bias_weight,
                                                             p=ifp_args["p"],
                                                             normalize_features=ifp_args.get("norm", False),
                                                             patch_wise=ifp_args.get("pw", False))
        feature_distances = depth_ifp_w * depth_feature_distances

    return scores, {'distances': feature_distances, 'dist_i_to_img_idx': dist_i_to_img_idx,
                    'img_idx_to_dist_i': img_idx_to_dist_i}


def _calc_feature_distance(features, bias, bias_weight, p, normalize_features, patch_wise):
    assert isinstance(features, list)
    assert features[0].shape[0] == 1
    features = torch.cat(features)
    N, C, H, W = features.shape
    if normalize_features:
        std, mean = torch.std_mean(features, dim=[0, 2, 3], keepdim=True)
        features = (features - mean) / std
    if patch_wise:
        print(features.shape)
        features = features.permute(0, 2, 3, 1)
        assert features.shape == (N, H, W, C)
        print(features.shape)
        features = features.flatten(end_dim=-2)
        assert features.shape == (N * H * W, C)
        print(features.shape)
        CHUNK_SIZE = 200
        N_CHUNKS = int(math.ceil(N / CHUNK_SIZE))
        feature_distances = []
        for j in range(N_CHUNKS):
            lower_i = j * CHUNK_SIZE * H * W
            upper_i = min((j + 1) * CHUNK_SIZE * H * W, N * H * W)
            current_chunk_size = int((upper_i - lower_i) / H / W)
            print(f"Chunk {j} from {lower_i} to {upper_i}")
            chunk = features[lower_i: upper_i]
            chunk_distances = torch.cdist(chunk, features, p=p)
            assert chunk_distances.shape == (current_chunk_size * H * W, N * H * W)
            print(chunk_distances.shape)
            chunk_distances = chunk_distances.reshape(current_chunk_size, H * W, N, H * W)
            chunk_distances = chunk_distances.permute(0, 2, 1, 3)
            assert chunk_distances.shape == (current_chunk_size, N, H * W, H * W)
            print(chunk_distances.shape)
            # chunk_distances = chunk_distances.reshape(current_chunk_size, N, H * W * H * W)
            chunk_distances = torch.min(chunk_distances, dim=-1).values
            chunk_distances = torch.mean(chunk_distances, dim=-1)
            print(chunk_distances.shape)
            feature_distances.append(chunk_distances)
        feature_distances = torch.cat(feature_distances)
        print(feature_distances.shape)
    else:
        features = features.flatten(start_dim=1)
        feature_distances = torch.cdist(features, features, p=p)
    if bias_weight > 0:
        assert len(bias) == feature_distances.shape[0]
        original_dist = feature_distances.clone()
        feature_distances += torch.tensor(bias, device=feature_distances.device)
        assert feature_distances[1, 3] == original_dist[1, 3] + bias[3]
        assert feature_distances[3, 1] == original_dist[3, 1] + bias[1]
    # For some reason the diagonal is not always exactly zero
    feature_distances.fill_diagonal_(0)
    return feature_distances


def iterative_farthest_point(current_samples, feature_distances, n_new, preselected_samples=None):
    dist = deepcopy(feature_distances["distances"])
    dist_i_to_img_idx = feature_distances["dist_i_to_img_idx"]
    img_idx_to_dist_i = feature_distances["img_idx_to_dist_i"]
    current_samples = [img_idx_to_dist_i[s] for s in current_samples]
    if preselected_samples is not None:
        preselected_samples = [img_idx_to_dist_i[s] for s in preselected_samples]
        ignored_samples = [i for i in range(dist.shape[0]) if i not in preselected_samples]
        dist[:, ignored_samples] = 0
    new_samples, distances = [], []
    for i in range(n_new):
        distances_to_current = dist[current_samples, :]
        min_distance_to_current = torch.min(distances_to_current, dim=0)
        farthest_sample = torch.max(min_distance_to_current.values, dim=0)
        new_sample = farthest_sample.indices.item()
        if new_sample in current_samples:
            break
        current_samples.append(new_sample)
        new_samples.append(new_sample)
        distances.append(farthest_sample.values)
    new_samples = [dist_i_to_img_idx[s] for s in new_samples]
    return new_samples, distances


def get_n_total(cfg):
    if cfg["data"]["dataset"] == "cityscapes":
        return 2975
    elif cfg["data"]["dataset"] == "camvid":
        return 367
    elif cfg["data"]["dataset"] == "mapillary":
        return 18000
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/cityscapes_joint.yml",
        help="Configuration file to use",
    )
    args = parser.parse_args()
    with open(args.config) as file_pointer:
        config = yaml.safe_load(file_pointer)
    label_selection_main(config)
