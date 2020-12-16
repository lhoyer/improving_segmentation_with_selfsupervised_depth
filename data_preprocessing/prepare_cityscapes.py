import glob
import os

import ray
from PIL import Image
from tqdm import tqdm


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

def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])

# Downscale images
@ray.remote(num_cpus=4)
def process_images_ray(*args, **kwargs):
    return process_images(*args, **kwargs)

# Check for corrupted files
@ray.remote(num_cpus=4)
def repair_ray(files, in_dir, out_dir, res):
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
    CITYSCAPES_ROOT = "datasets/Cityscapes/"
    CONVERT_LIST = [
        ("leftImg8bit_trainvaltest/leftImg8bit/", "leftImg8bit_small/", (1024, 512)),
        ("leftImg8bit_sequence", "leftImg8bit_sequence_small", (1024, 512)),
    ]

    # Convert files
    ray.shutdown()
    ray.init()
    for in_dir, out_dir, res in CONVERT_LIST:
        in_dir = CITYSCAPES_ROOT + in_dir
        out_dir = CITYSCAPES_ROOT + out_dir
        files = [f for f in glob.glob(in_dir + "**/*.png", recursive=True)]
        files = [f for f in files if "/test" not in f]

        print(f'Convert {len(files)} files for {out_dir}.')

        obj_ids = []
        n = 100
        for fs in [files[i:i + n] for i in range(0, len(files), n)]:
            obj_ids.append(process_images_ray.remote(fs, in_dir, out_dir, res))
        for x in tqdm(to_iterator(obj_ids), total=len(obj_ids)):
            pass

    # Verify and repair files
    for in_dir, out_dir, res in CONVERT_LIST:
        in_dir = CITYSCAPES_ROOT + in_dir
        out_dir = CITYSCAPES_ROOT + out_dir
        files = [f for f in glob.glob(in_dir + "**/*.png", recursive=True)]
        files = [f for f in files if "/test" not in f]

        print(f'Verify {len(files)} files for {out_dir}.')

        obj_ids = []
        n = 100
        for fs in [files[i:i + n] for i in range(0, len(files), n)]:
            obj_ids.append(repair_ray.remote(fs, in_dir, out_dir, res))
        for x in tqdm(to_iterator(obj_ids), total=len(obj_ids)):
            pass

    ray.shutdown()
