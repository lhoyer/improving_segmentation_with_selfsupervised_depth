## Improving Semi-Supervised and Domain-Adaptive Semantic Segmentation with Self-Supervised Depth Estimation

This is the official pytorch implementation of our paper 
[Improving Semi-Supervised and Domain-Adaptive Semantic Segmentation with Self-Supervised Depth Estimation](https://arxiv.org/pdf/2108.12545.pdf).

Training deep networks for semantic segmentation requires large amounts of labeled training data, which presents a major
challenge in practice, as labeling segmentation masks is a highly labor-intensive process. To address this issue, we 
present a framework for semi-supervised and domain-adaptive semantic segmentation, which is enhanced by self-supervised 
monocular depth estimation (SDE) trained only on unlabeled image sequences.

In particular, we propose four key contributions:

1. We automatically select the most useful samples to be annotated for semantic segmentation based on the correlation 
   of sample diversity and difficulty between SDE and semantic segmentation. 
2. We implement a strong data augmentation by mixing images and labels using the structure of the scene.
3. We transfer knowledge from features learned during SDE to semantic segmentation by means of transfer and 
   multi-task learning.
4. We exploit additional labeled synthetic data with Cross-Domain DepthMix and Matching Geometry Sampling to align
   synthetic and real data.

We validate the proposed model on the Cityscapes dataset, where all four contributions demonstrate significant 
performance gains, and achieve state-of-the-art results for semi-supervised semantic segmentation as well as for 
semi-supervised domain adaptation. In particular, with only 1/30 of the Cityscapes labels, our method achieves 92% 
of the fully-supervised baseline performance and even 97% when exploiting additional data from GTA.

Below, you can see the qualitative results of our model trained with only 100 annotated semantic segmentation samples.

<p align="center">
  <img src="demo.gif" alt="example input output gif" width="1200" />
</p>

If you find this code useful in your research, please consider citing:

```
@article{hoyer2021improving,
  title={Improving Semi-Supervised and Domain-Adaptive Semantic Segmentation with Self-Supervised Depth Estimation},
  author={Hoyer, Lukas and Dai, Dengxin and Wang, Qin and Chen, Yuhua and Van Gool, Luc},
  journal={arXiv preprint arXiv:2108.12545 [cs]},
  year={2021}
}
```
```
@inproceedings{hoyer2021three,
  title={Three Ways to Improve Semantic Segmentation with Self-Supervised Depth Estimation},
  author={Hoyer, Lukas and Dai, Dengxin and Chen, Yuhua and Köring, Adrian and Saha, Suman and Van Gool, Luc},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={11130--11140},
  year={2021}
}
```

### Setup Environment

To install all requirements, you can run:

```
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

### Download Datasets

For this project, the following datasets are required:

* **Cityscapes**: Please download Cityscapes from https://www.cityscapes-dataset.com/downloads/. 
  We require following packages: gtFine_trainvaltest.zip, leftImg8bit_trainvaltest.zip, and 
  leftImg8bit_sequence_trainvaltest.zip.
* **GTA5 Segmentation**: Please download the GTA5 segmentation dataset from https://download.visinf.tu-darmstadt.de/data/from_games/.
  For convenience, you can also use the script [data_processing/gta_seg_download.sh](data_processing/gta_seg_download.sh).
* **GTA5 Video Sequences** (optional): Please download the GTA5 video sequences dataset from https://playing-for-benchmarks.org/download/.
  For convenience, you can also use the script [data_processing/gta_seq_download.sh](data_processing/gta_seq_download.sh).
  After the download and extraction of the dataset, please remove the corrupted file `train/img/052/052_00064.jpg`.
  This dataset is only required to train self-supervised depth estimation. You can skip it if you want to use our pretrained depth model.
* **SYNTHIA-RAND-CITYSCAPES**: Please download SYNTHIA-RAND-CITYSCAPES from http://synthia-dataset.net/download/808/. The synthia 
  labels mapped to cityscapes, are available [here](https://drive.google.com/file/d/1cwGJGzE8yEvKTzF2KeaBcydnVZmng-KL/view?usp=sharing).
* **SYNTHIA Video Sequences** (optional): Please download the Synthia Video Sequences from http://synthia-dataset.net/downloads/.
  For convencience, you can also use the script [data_processing/synthia_seq_download.sh](data_processing/synthia_seq_download.sh).
  This dataset is only required to train self-supervised depth estimation. You can skip it if you want to use our pretrained depth model.

Please extract the datasets to construct the following folder structure:

```
datasets/cityscapes/
- gtFine/
- leftImg8bit_trainvaltest/
- leftImg8bit_sequence/
datasets/gta_seg/
- images/
- labels/
datasets/gta_seq/
- img/ (optional)
- cls/ (optional)
datasets/synthia/
- RGB/
- video/ (optional)
- synthia_mapped_to_cityscapes/
```

For performance reasons, we work on downsampled copies of the dataset. 
To downsample the datasets, you can run
```
python -m data_preprocessing.downsample_datasets --machine ws
```

### Optional: Pretrain Self-Supervised Depth on GTA and Synthia

To train your own self-supervised depth estimation model on GTA and Synthia, you
can run the following training:

```
python run_experiments.py --machine ws --config configs/sde_dec11.yml --exp 310
```

The logs and checkpoints are saved in the directory `results/`

You can also skip this section if you want to use our pretrained self-supervised 
depth estimation model. It will be downloaded automatically in the following section. 
If you want to use your own pretrained model, you can adapt `setup_sde_ckpt()` in
[experiments.py#L113](experiments.py#L113) so that the model references point to 
your own model checkpoints.

### Run Semi-Supervised Domain Adaptation Experiments

The semi-supervised domain adaptation semantic segmentation model can be trained
using:

```
python run_experiments.py --machine ws --exp 262
```

If you want to change the number of labeled target samples, the source dataset,
or run it on multiple seeds, please refer to [experiments.py#222](experiments.py#222).

Further ablations that include multi-task learning can be run using the same
command. For that purpose, please comment in the desired configurations in
[experiments.py#222](experiments.py#222) for `id == 262`.

To run ablations that do not use multi-task learning, you can refer to 
[experiments.py#167](experiments.py#167) `id == 260` and run:

```
python run_experiments.py --machine ws --exp 260
```

Please note that this branch does not include the semi-supervised semantic
segmentation experiments. Please refer to the `master` branch of this repository
to run them.