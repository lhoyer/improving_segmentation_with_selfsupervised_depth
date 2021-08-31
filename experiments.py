import itertools
from copy import deepcopy

from loader.preselected_labels import preselected_labels


def decoder_variant(cfg, dec, crop):
    cfg['model']['replace_stride_with_dilation'] = [False, False, True]
    if dec in [5, 6]:
        cfg['model']['depth_args'] = {
            'intermediate_aspp': True,
            'aspp_rates': [6, 12, 18],
            'num_ch_dec': [64, 128, 128, 256, 256],
            'max_scale_size': crop
        }
        load_backbone = (dec in [6])
    elif dec == 9:
        cfg['model']['depth_args'] = {
            'intermediate_aspp': True,
            'aspp_rates': [6, 12, 18],
            'num_ch_dec': [64, 64, 128, 128, 256],
            'batch_norm': True,
            'max_scale_size': crop
        }
        load_backbone = False
    else:
        raise NotImplementedError

    return cfg, load_backbone

def setup_optimizer(cfg, opt, lr, blr, plr, slr, gclip):
    cfg["training"]["optimizer"] = {
        "name": opt,
        "lr": lr,
        "backbone_lr": blr,
    }
    if plr is not None:
        cfg["training"]["optimizer"]["pose_lr"] = plr
    if slr is not None:
        cfg["training"]["optimizer"]["segmentation_lr"] = slr
    if opt == "sgd":
        cfg["training"]["optimizer"].update({
            "momentum": 0.9,
            "weight_decay": 0.0005
        })
    cfg["training"]["clip_grad_norm"] = gclip
    return cfg


def lr_schedule(cfg, lr_sch, max_iter, step=30e3):
    if lr_sch == "step":
        cfg["training"]["lr_schedule"] = {
            "name": "step_lr", "step_size": int(50e3), "gamma": 0.1
        }
    elif lr_sch == "step2":
        cfg["training"]["lr_schedule"] = {
            "name": "multi_step", "milestones": [int(30e3), int(40e3), int(50e3)], "gamma": 0.5
        }
    elif lr_sch == "step30":
        cfg["training"]["lr_schedule"] = {
            "name": "step_lr", "step_size": int(30e3), "gamma": 0.1
        }
    elif lr_sch == "stepx":
        cfg["training"]["lr_schedule"] = {
            "name": "step_lr", "step_size": int(step), "gamma": 0.1
        }
    elif lr_sch == "poly":
        cfg['training']['lr_schedule'] = {
            'name': 'poly_lr_2', 'power': 0.9, 'max_iter': max_iter
        }
    else:
        raise NotImplementedError

    return cfg

def setup_dataset(cfg, dataset, crop, lr_sch, train_iters=None, step=None, enable_final_val_interval=True):
    if train_iters is None:
        train_iters = {"cityscapes": int(40e3), "mapillary": int(40e3), "camvid": int(20e3), "gtaseg": int(40e3)}[dataset]
    if step is None:
        step = {"cityscapes": int(30e3), "mapillary": int(30e3), "camvid": int(15e3), "gtaseg": int(30e3)}[dataset]
    final_val_interval = {"cityscapes": 500, "mapillary": 1000, "camvid": 500, "gtaseg": 500}[dataset]

    w, h = {"cityscapes": (1024, 512), "mapillary": (704, 512), "camvid": (672, 512), "gtaseg": (1024, 512)}[dataset]
    cfg['data'].update({
        'dataset': dataset,
        'path': {"cityscapes": "MachineConfig.CITYSCAPES_DIR",
                 "camvid": "MachineConfig.CAMVID_DIR",
                 "mapillary": "MachineConfig.MAPILLARY_DIR",
                 "gtaseg": "MachineConfig.GTASEG_DIR"}[dataset],
        'val_split': {"cityscapes": "val", "mapillary": "validation", "camvid": "test", "gtaseg": "val"}[dataset],
    })
    cfg['monodepth_options']['height'] = h
    cfg['monodepth_options']['width'] = w
    cfg['monodepth_options']['crop_h'] = crop[0]
    cfg['monodepth_options']['crop_w'] = crop[1]
    cfg['training']['train_iters'] = train_iters
    cfg = lr_schedule(cfg, lr_sch, train_iters, step=step)
    if enable_final_val_interval:
        cfg['training']['val_interval'][str(int(step))] = final_val_interval

    return cfg

def setup_source_data(cfg, source_dataset):
    assert "source_data" in cfg, "Have you used configs/ssda.yml?"
    cfg['source_data'].update({
        "dataset": {"gta": "gtaseg", "syn": "synthiaseg"}[source_dataset],
        "path": {
            "gta": "MachineConfig.GTASEG_DIR", "syn": "MachineConfig.SYNTHIA_DIR"
        }[source_dataset],
    })
    return cfg

def setup_sde_ckpt(cfg, dataset, source_dataset):
    # sde_ckpt works with full local paths as well: /path/to/best_model.pkl
    cfg['source_data']["sde_ckpt"] = {
        "gta": "gta_rev2_fc046",
        "syn": "synthia_rev2_89b89",
    }[source_dataset]
    assert dataset == "cityscapes"
    cfg['model']['depth_estimator_weights'] = 'mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd2_crop512x512bs4'
    return cfg

def setup_preselect(cfg, preselect, dataset, n_subset, seed, method="ds_us"):
    cfg['data']['restrict_to_subset']['mode'] = "fixed" if preselect else "random"
    cfg['data']['restrict_to_subset']['n_subset'] = n_subset
    if preselect:
        cfg['data']['restrict_to_subset']['subset'] = preselected_labels(
            {7: 42, 25: 43, 42: 44}[seed], n_subset, dataset, method=method,
        )
    return cfg

def set_segmentation_args(cfg, seg_init, layers, head_inter, output_stride, head_dropout=0.1):
    cfg['model']['segmentation_args'] = {
        'weights': seg_init,
        'layers': layers,
        'head_inter_channels': 64,
        'layer_out_channels': 64,
        'head_dropout': head_dropout,
        'layer_dropout': 0,
        'head_inter': head_inter,
        'output_stride': output_stride
    }
    return cfg


def generate_experiment_cfgs(base_cfg, id):
    cfgs = []
    # Self-supervised depth for source domain
    # Use with configs/sde_dec11.yml. If the training collapses
    # into constant depth predictions, a rerun usually helps.
    if id == 310:
        assert base_cfg["training"]["monodepth_lambda"] == 1.0 and \
               base_cfg["training"]["segmentation_lambda"] == 0.0, \
            "Did you use configs/sde_dec11.yml?"
        for dataset in [
            "gta",
            "synthia"
        ]:
            cfg = deepcopy(base_cfg)
            cfg["data"].update({
                "dataset": {"cs": "cityscapes", "gta": "gtaseq", "synthia": "synthiaseq"}[dataset],
                "path": {"cs": "MachineConfig.CITYSCAPES_DIR", "gta": "MachineConfig.GTASEQ_DIR",
                         "synthia": "MachineConfig.SYNTHIA_DIR"}[dataset]
            })
            cfg_name = f'{dataset}_rev2'
            cfg['tag'] = cfg_name
            cfgs.append(cfg)
    # SSDA with Transfer Learning
    elif id == 260:
        dataset = "cityscapes"
        sde_w = f'mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd2_crop512x512bs4'
        for source_dataset, seed, n_subset in itertools.product([
            "gta",
            "syn"
        ], [
            # 7,
            # 25,
            42
        ], [
            100,
            # 500,
            # 200,
            # 2975
        ]):
            for name,                           init,   ema,    mix_mask,       intra_mix,  inter_mix,  match_geom, preselect in [
                ('baseline',                    'none', False,  None,           False,      False,      False,      False),
                ('sel',                         'none', False,  None,           False,      False,      False,      True),
                ("classmix",                    'none', True,   "class",        False,      True,       False,      False),
                ("depthmix_cdm",                'none', True,   "depthcomp",    True,       False,      False,      False),
                ("depthmix_tdm",                'none', True,   "depthcomp",    False,      True,       False,      False),
                ("depthmix_tdm_cdm",            'none', True,   "depthcomp",    True,       True,       False,      False),
                ("depthmix_tdm_cdm_mg",         'none', True,   "depthcomp",    True,       True,       True,       False),
            ]:
                if preselect and n_subset == 2975:
                    continue
                name = name.replace('.', '').replace(' ', '').replace(',', 'i').replace('(', 'I').replace(')', 'I')
                unlab_cfg = {
                    "inter_domain_mix": inter_mix,
                    "intra_domain_mix": intra_mix,
                    "mix_mask": mix_mask,
                    "only_unlabeled": mix_mask is None,
                    "mix_use_gt": mix_mask is not None,
                    "depthcomp_margin": 0.03,
                    "debug_image": True
                } if ema else None

                cfg = deepcopy(base_cfg)
                cfg['tag'] = f"{source_dataset}2{dataset}_{name}_D{n_subset}_S{seed}"
                cfg['model']['variant'] = name
                cfg, load_backbone = decoder_variant(cfg, dec=6, crop=(512, 512))
                cfg['model']['backbone_pretraining'] = sde_w if (
                        load_backbone and init != "none") else "imnet"
                cfg['model']['depth_pretraining'] = init
                cfg = setup_optimizer(cfg, "sgd", lr=1e-2, blr=1e-3, plr=None, slr=None, gclip=10)
                cfg = setup_dataset(cfg, dataset, (512, 512), "stepx")
                cfg = setup_source_data(cfg, source_dataset)
                cfg = setup_sde_ckpt(cfg, dataset, source_dataset)
                cfg = setup_preselect(cfg, preselect, dataset, n_subset, seed)
                cfg['matching_geometry']['enabled'] = match_geom
                cfg['training']['unlabeled_segmentation'] = unlab_cfg
                cfg['seed'] = seed
                cfg = set_segmentation_args(cfg, seg_init=init, layers=[9], head_inter=False, output_stride=1)
                cfgs.append(cfg)
    # SSDA with Multi Task Learning
    elif id == 262:
        dataset = "cityscapes"
        sde_w = f'mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd2_crop512x512bs4'
        for source_dataset, seed, n_subset in itertools.product([
            "gta",
            "syn"
        ], [
            # 7,
            # 25,
            42
        ], [
            100,
            # 500,
            # 200,
            # 2975
        ]):
            for name,                           ema,    mix_mask,       intra_mix,  inter_mix,  match_geom, preselect in [
                # ('mtl',                         False,  None,           False,      False,      False,      False),
                # ("sel_depthmix_tdm_mtl",        True,   "depthcomp",    True,       False,      False,      True),
                # ("depthmix_tdm_cdm_mg_mtl",     True,   "depthcomp",    True,       True,       True,       False),
                ('sel_depthmix_tdm_cdm_mg_mtl', True,   "depthcomp",    True,       True,       True,       True),
            ]:
                if preselect and n_subset == 2975:
                    continue
                name = name.replace('.', '').replace(' ', '').replace(',', 'i').replace('(', 'I').replace(')', 'I')
                unlab_cfg = {
                    "inter_domain_mix": inter_mix,
                    "intra_domain_mix": intra_mix,
                    "mix_mask": mix_mask,
                    "depthmix_online_depth": False,  # Changed for SSDA
                    "only_unlabeled": mix_mask is None,
                    "mix_use_gt": mix_mask is not None,
                    "depthcomp_margin": 0.03,
                    "debug_image": True
                } if ema else None

                cfg = deepcopy(base_cfg)
                cfg['tag'] = f"{source_dataset}2{dataset}_{name}_D{n_subset}_S{seed}"
                cfg['model']['variant'] = name
                cfg['model']['segmentation_name'] = 'mtl_pad'
                cfg['model']['backbone_name'] = f"resnet101"
                cfg, load_backbone = decoder_variant(cfg, 6, (512, 512))
                cfg['model']['backbone_pretraining'] = sde_w
                cfg['model']['depth_pretraining'] = sde_w
                cfg['model']['pose_pretraining'] = sde_w
                cfg['model']['disable_pose'] = False
                cfg['model']['disable_monodepth'] = False
                cfg['training']['segmentation_lambda'] = 1
                cfg['training']['monodepth_lambda'] = 1
                cfg = setup_optimizer(cfg, "sgd", 1e-2, 1e-3, 1e-6, None, 10)
                cfg = setup_dataset(cfg, dataset, (512, 512), "stepx")
                cfg = setup_source_data(cfg, source_dataset)
                cfg = setup_sde_ckpt(cfg, dataset, source_dataset)
                cfg = setup_preselect(cfg, preselect, dataset, n_subset, seed)
                cfg['matching_geometry']['enabled'] = match_geom
                cfg['training']['unlabeled_segmentation'] = unlab_cfg
                cfg['seed'] = seed
                cfg['model']['segmentation_args'] = {
                    'weights': sde_w,
                    'output_stride': 1,
                    'distillation_layer': 7,
                    'side_output': True,
                    'final_layer': 9
                }
                cfgs.append(cfg)
    else:
        raise NotImplementedError("Unknown id {}".format(id))

    return cfgs
