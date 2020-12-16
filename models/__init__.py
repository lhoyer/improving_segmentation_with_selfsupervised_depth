import copy

from models.joint_segmentation_depth import joint_segmentation_depth


def get_model(model_dict, n_classes):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    model = model(name=name, num_classes=n_classes, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "joint_segmentation_depth": joint_segmentation_depth,
        }[name]
    except:
        raise NotImplementedError("Model {} not available".format(name))
