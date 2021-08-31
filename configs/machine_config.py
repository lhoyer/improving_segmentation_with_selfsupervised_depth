import os


class MachineConfig:
    DOWNLOAD_MODEL_DIR = None
    CITYSCAPES_DIR = None
    GTASEQ_DIR = None
    GTASEG_DIR = None
    SYNTHIA_DIR = None
    GENERATED_DEPTH_DIR = None
    GENERATED_DEPTH_ARCHIVE_DIR = None
    LOG_DIR = None
    AVAIL_MACHINES = ["ws"]
    GLOBAL_TRAIN_CFG = None
    MACHINE = None

    def __init__(self, machine):
        MachineConfig.MACHINE = machine
        if machine == "ws":
            MachineConfig.DOWNLOAD_MODEL_DIR = "models/"
            MachineConfig.CITYSCAPES_DIR = "datasets/Cityscapes/"
            MachineConfig.GTASEQ_DIR = "datasets/GTASeq/"
            MachineConfig.GTASEG_DIR = "datasets/GTASeg/"
            MachineConfig.SYNTHIA_DIR = "datasets/Synthia/"
            MachineConfig.GENERATED_DEPTH_DIR = "datasets/depth"
            MachineConfig.GENERATED_DEPTH_ARCHIVE_DIR = "datasets/depth-archives"
            MachineConfig.LOG_DIR = "results/"
        else:
            raise NotImplementedError(machine)
