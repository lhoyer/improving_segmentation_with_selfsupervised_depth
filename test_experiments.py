import argparse
import os
import uuid
from datetime import datetime

import torch
import yaml

import train
from configs.machine_config import MachineConfig
from experiments import generate_experiment_cfgs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--machine",
        type=str,
        choices=MachineConfig.AVAIL_MACHINES,
    )
    args = parser.parse_args()

    MachineConfig(args.machine)

    for exp, base_cfg_file in [
        (310, "configs/sde_dec11.yml"),
        (260, "configs/ssda.yml"),
        (262, "configs/ssda.yml")
    ]:
        with open(base_cfg_file) as fp:
            base_cfg = yaml.safe_load(fp)
        cfgs = generate_experiment_cfgs(base_cfg, exp)

        experiment_name = "{}_{}".format(
            base_cfg_file.rsplit("/", 1)[1].split(".")[0],
            exp
        )
        print("Start experiment {}".format(experiment_name))
        run_id = experiment_name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print("Run ID: " + str(run_id))

        for i, cfg in enumerate(cfgs):
            print("Dispatch job {}".format(cfg["tag"]))
            cfg["experiment_name"] = experiment_name
            cfg["name"] = f'{datetime.now().strftime("%y%m%d_%H%M")}_{cfg["tag"]}_{str(uuid.uuid4())[:5]}'
            cfg["machine"] = args.machine
            cfg["training"]["log_path"] = os.path.join(cfg["training"]["log_path"], "test", experiment_name) + "/"
            cfg["training"]["print_interval"] = 1
            cfg["training"]["val_interval"] = {"0": 1}
            cfg["training"]["train_iters"] = 2
            cfg["data"]["num_val_samples"] = 10

            train.train_main(cfg)

            torch.cuda.empty_cache()
