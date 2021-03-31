import argparse
import os
import traceback
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
        "--exp",
        nargs="?",
        type=int,
        help="Experiment id as defined in cluster_experiment.py",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
    )
    parser.add_argument(
        "--machine",
        type=str,
        choices=MachineConfig.AVAIL_MACHINES
    )
    parser.add_argument(
        "--run",
        type=str,
        default="all",
        help="Run id within an experiment. If not specified, run all."
    )
    args = parser.parse_args()
    if args.run == "all":
        pass
    elif "-" in args.run:
        low, up = args.run.split("-")
        args.run = list(range(int(low), int(up)))
    else:
        args.run = [int(i) for i in args.run.split(",")]
    MachineConfig(args.machine)

    with open(args.config) as fp:
        base_cfg = yaml.safe_load(fp)

    cfgs = generate_experiment_cfgs(base_cfg, args.exp)


    def trial_name_string(trial):
        if 'tag' in trial.config:
            return trial.config['tag']
        else:
            return trial.config['general']['tag']


    experiment_name = "{}_{}".format(
        args.config.rsplit("/", 1)[1].split(".")[0],
        args.exp
    )

    print("Start experiment {}".format(experiment_name))

    run_id = experiment_name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print("Run ID: " + str(run_id))
    out_dir = os.path.join("$HOME/dispatcher", run_id)
    out_dir = os.path.expandvars(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    variant_generator = CustomVariantGenerator()
    unresolved_spec = {
        "config": cfgs
    }
    for i, variant in enumerate(variant_generator._generate_resolved_specs(1, unresolved_spec)):
        if args.run != "all" and i not in args.run:
            continue
        print("Dispatch job {}".format(variant["experiment_tag"]))
        cfg = variant["spec"]["config"]
        cfg["name"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + variant["experiment_tag"]
        cfg["machine"] = args.machine
        cfg["training"]["log_path"] = os.path.join(cfg["training"]["log_path"], experiment_name) + "/"
        out_file = os.path.join(out_dir, variant["trial_id"] + ".yaml")
        with open(out_file, 'w') as of:
            yaml.safe_dump(cfg, of, default_flow_style=False)
        if not args.dry:
            try:
                if args.exp == 211:
                    label_selection.label_selection_main(cfg)
                else:
                    train.train_main(cfg)
            except Exception:
                print(traceback.format_exc())
                print("Continue with next experiment.")
            torch.cuda.empty_cache()
