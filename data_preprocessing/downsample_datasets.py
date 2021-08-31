import argparse
import glob
import multiprocessing
import os
from functools import partial

from PIL import Image
from tqdm import tqdm

from configs.machine_config import MachineConfig


# Downscale images
def process_images(fs, in_dir, out_dir, res, replace=False):
    for f in fs:
        new_f = f.replace(in_dir, out_dir)
        assert f != new_f, "{} and {} are the same.".format(f, new_f)
        new_f = new_f.replace('.png', '.jpg')
        # print('Old', f)
        # print('New', new_f)
        if os.path.isfile(new_f) and not replace:
            # print('Already exists.')
            continue

        os.makedirs(os.path.dirname(new_f), exist_ok=True)

        with open(f, 'rb') as fp:
            with Image.open(fp) as img:
                img = img.resize(res, Image.ANTIALIAS)
                # almost no compression artifacts when visually
                # compared with downscaled png
                img.save(new_f, subsampling=0, quality=98)

# Check for corrupted files
def repair(files, in_dir, out_dir, res):
    for f in files:
        new_f = f.replace(in_dir, out_dir)
        assert f != new_f, "{} and {} are the same.".format(f, new_f)
        new_f = new_f.replace('.png', '.jpg')

        try:
            with open(new_f, 'rb') as fp:
                img = Image.open(fp).convert("RGB")
        except:
            print("Try again to process {}".format(f))
            process_images([f], in_dir, out_dir, res, replace=True)
            with open(new_f, 'rb') as fp:
                img = Image.open(fp).convert("RGB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--machine",
        type=str,
        default="",
        choices=[*MachineConfig.AVAIL_MACHINES, ""]
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
    )
    args = parser.parse_args()
    MachineConfig(args.machine)
    n = args.chunk_size

    CONVERT_LIST = [
        (MachineConfig.CITYSCAPES_DIR + "leftImg8bit_trainvaltest/leftImg8bit/",
         MachineConfig.CITYSCAPES_DIR + "leftImg8bit_small/", (1024, 512)),
        (MachineConfig.CITYSCAPES_DIR + "leftImg8bit_sequence",
         MachineConfig.CITYSCAPES_DIR + "leftImg8bit_sequence_small", (1024, 512)),
        (MachineConfig.GTASEG_DIR + "images/",
         MachineConfig.GTASEG_DIR + "images_small/", (1024, 512)),
        (MachineConfig.SYNTHIA_DIR + "video/",
         MachineConfig.SYNTHIA_DIR + "video_small/", (1024, 512)),
        (MachineConfig.SYNTHIA_DIR + "RGB/",
         MachineConfig.SYNTHIA_DIR + "RGB_small/", (1024, 512)),
    ]

    # Convert files
    for in_dir, out_dir, res in CONVERT_LIST:
        files = [f for f in glob.glob(in_dir + "**/*.png", recursive=True)]
        files = [f for f in files if "/test" not in f]

        print(f'Convert {len(files)} files for {out_dir}.')

        pool = multiprocessing.Pool(args.threads)
        chunks = [files[i:i + n] for i in range(0, len(files), n)]
        partial_process_images = partial(process_images, in_dir=in_dir, out_dir=out_dir, res=res)
        r = list(tqdm(pool.imap(partial_process_images, chunks), total=len(chunks)))

    # Verify and repair files
    for in_dir, out_dir, res in CONVERT_LIST:
        files = [f for f in glob.glob(in_dir + "**/*.png", recursive=True)]
        files = [f for f in files if "/test" not in f]

        print(f'Verify {len(files)} files for {out_dir}.')

        pool = multiprocessing.Pool(args.threads)
        chunks = [files[i:i + n] for i in range(0, len(files), n)]
        partial_repair = partial(repair, in_dir=in_dir, out_dir=out_dir, res=res)
        r = list(tqdm(pool.imap(partial_repair, chunks), total=len(chunks)))