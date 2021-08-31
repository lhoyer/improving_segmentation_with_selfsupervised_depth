import contextlib
import datetime
import logging
import os
import tarfile

import numpy as np


@contextlib.contextmanager
def np_local_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

def is_source_file(x):
    if x.isdir() or x.name.endswith((".py", ".sh", ".yml", ".json", ".txt")):
        # print(x.name)
        return x
    else:
        return None

def gen_code_archive(out_dir, id):
    archive = os.path.join(out_dir, f"code_{id}.tar.gz")
    os.makedirs(os.path.dirname(archive), exist_ok=True)
    with tarfile.open(archive, mode='w:gz') as tar:
        tar.add(".", filter=is_source_file)
    return archive

def get_logger(logdir):
    logger = logging.getLogger("segsde")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger
