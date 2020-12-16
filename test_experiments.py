import argparse
import os
from datetime import datetime

import torch
import yaml

import label_selection
import train
from configs.machine_config import MachineConfig
from experiments import generate_experiment_cfgs
from utils.cluster_utils import CustomVariantGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/cityscapes_joint.yml",
        help="Base config file to use",
    )
    parser.add_argument(
        "--machine",
        type=str,
        choices=["ws", "slurm", "dgx", "marvin", "aws"],
    )
    args = parser.parse_args()

    MachineConfig(args.machine)

    with open(args.config) as fp:
        base_cfg = yaml.safe_load(fp)

    for exp in [210, 211, 212]:
        cfgs = generate_experiment_cfgs(base_cfg, exp)

        def trial_name_string(trial):
            if 'tag' in trial.config:
                return trial.config['tag']
            else:
                return trial.config['general']['tag']

        experiment_name = "{}_{}".format(
            args.config.rsplit("/", 1)[1].split(".")[0],
            exp
        )

        print("Start experiment {}".format(experiment_name))

        run_id = experiment_name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print("Run ID: " + str(run_id))

        variant_generator = CustomVariantGenerator()
        unresolved_spec = {
            "config": cfgs
        }
        for i, variant in enumerate(variant_generator._generate_resolved_specs(1, unresolved_spec)):
            print("Dispatch job {}".format(variant["experiment_tag"]))
            cfg = variant["spec"]["config"]
            cfg["name"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + variant["experiment_tag"]
            cfg["machine"] = args.machine
            cfg["training"]["log_path"] = os.path.join(cfg["training"]["log_path"], "test", experiment_name) + "/"
            cfg["training"]["print_interval"] = 1
            cfg["training"]["val_interval"] = {"0": 1}
            cfg["training"]["train_iters"] = 2

            if exp == 211:
                cfg['label_selection'].update({
                    'label_steps': [25, 50],
                    'train_iters': [2, 2],
                })
                cfg["training"]["lr_schedule"]["max_iter"] = 2
                label_selection.label_selection_main(cfg)
            else:
                train.train_main(cfg)

            torch.cuda.empty_cache()
