class MachineConfig:
    DOWNLOAD_MODEL_DIR = None
    CITYSCAPES_DIR = None
    GENERATED_DEPTH_DIR = None
    LOG_DIR = None
    AVAIL_MACHINES = ["ws"]

    def __init__(self, machine):
        if machine == "ws":
            MachineConfig.DOWNLOAD_MODEL_DIR = "models/"
            MachineConfig.CITYSCAPES_DIR = "datasets/Cityscapes/"
            MachineConfig.CAMVID_DIR = "datasets/CamVid/"
            MachineConfig.MAPILLARY_DIR = "datasets/Mapillary-Vistas/"
            MachineConfig.GENERATED_DEPTH_DIR = "generated_depth/"
            MachineConfig.LOG_DIR = "results/"
        else:
            raise NotImplementedError(machine)
