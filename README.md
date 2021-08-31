## Three Ways to Improve Semantic Segmentation with Self-Supervised Depth Estimation

This is the official pytorch implementation of our CVPR21 paper 
[Three Ways to Improve Semantic Segmentation with Self-Supervised Depth Estimation](https://arxiv.org/pdf/2012.10782.pdf)
and its extension to semi-supervised domain adaptation 
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
@inproceedings{hoyer2021three,
  title={Three Ways to Improve Semantic Segmentation with Self-Supervised Depth Estimation},
  author={Hoyer, Lukas and Dai, Dengxin and Chen, Yuhua and Köring, Adrian and Saha, Suman and Van Gool, Luc},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={11130--11140},
  year={2021}
}
```
```
@article{hoyer2021improving,
  title={Improving Semi-Supervised and Domain-Adaptive Semantic Segmentation with Self-Supervised Depth Estimation},
  author={Hoyer, Lukas and Dai, Dengxin and Wang, Qin and Chen, Yuhua and Van Gool, Luc},
  journal={arXiv preprint arXiv:2108.12545 [cs]},
  year={2021}
}
```

### Setup Environment

To install all requirements, you can run:

```
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

Please download Cityscapes from https://www.cityscapes-dataset.com/downloads/. We require following packages:
gtFine_trainvaltest.zip, leftImg8bit_trainvaltest.zip, and leftImg8bit_sequence_trainvaltest.zip. 
For performance reasons, we work on a downsampled copy of Cityscapes. Please refer to 
[data_preprocessing/prepare_cityscapes.py](data_preprocessing/prepare_cityscapes.py) for
more information.

Before continuing, you should have following folder structure prepared:

```
CITYSCAPES_DIR/
- gtFine/
- leftImg8bit_small/
- leftImg8bit_sequence_small/
```

You can setup the paths for data and logging in the machine config ``configs/machine_confg.py``.
In the following, we assume that you have called your machine `ws`.

### Inference with a Pretrained Model

If you want to test our pretrained model (trained on 372 Cityscapes images) on some of your own images, 
you can download the checkpoint [here](https://drive.google.com/file/d/1vnQAF_BeQWZagH-izSQIBRlSVDWDJME2/view?usp=sharing),
unzip it, and run it using:

```
python inference.py --machine ws --model /path/to/checkpoint/dir/ --data /path/to/data/dir/
```

### Pretrain Self-supervised Depth on Cityscapes

To run the two phases of the self-supervised depth estimation pretraining (first 300k iterations with frozen encoder and 
50k iterations with ImageNet feature distance loss), you can execute:

```
python train.py --machine ws --config configs/cityscapes_monodepth_highres_dec5_crop.yml
python train.py --machine ws --config configs/cityscapes_monodepth_highres_dec6_crop.yml
```

You can also skip this section if you want to use our pretrained self-supervised depth estimation model. 
It will be downloaded automatically in the following section. If you want to use your own pretrained
model, you can adapt [models/utils.py#L108](models/utils.py#L108) that the model references point to your
own model on Google drive.

### Run Semi-Supervised Experiments

The semi-supervised experiments can be executed using:

```
python run_experiments.py --machine ws --exp EXP_ID
```

The EXP_ID corresponds to the experiment defined in [experiments.py](experiments.py).
Following experiments are relevant for the paper:

* **Experiment 210:** All configurations that are only based on transfer learning.
* **Experiment 211:** Automatic data selection for annotation configurations.
* **Experiment 212:** Configurations that involve multi-task learning.

To use the labels selected by the automatic data selection for the other experiments, please copy the content of 
`nlabelsXXX_subset.json` from the log directory to [loader/preselected_labels.py](loader/preselected_labels.py)
after running experiment 211.
For better reproducibility, we have stored our results there as well.
Table 3 is generated using experiment 210 with the config sel_{pres_method}_scratch. 

Be aware that running all experiments takes multiple weeks on a single GPU. 
For that reason, we have commented out all but one subset size and seed as well as minor ablations.

### Run Semi-Supervised Domain Adaptation Experiments

In order to run our framework extension to semi-supervised domain adaptation,
please switch to the `ssda` branch and follow its README.md instructions.

### Framework Structure

##### Experiments and Configurations
* *configs/machine_config.yml:* Definition of data and log paths for different machines.
* *configs/cityscapes_monodepth\*:* Configurations for monodepth pretraining on Cityscapes.
* *configs/cityscapes_joint.yml:* Base configuration for all semi-supervised segmentation experiments.
* *experiments.py:* Generation of derivative configurations from cityscapes_joint.yml for the different experiments.
* *run_experiments.py:* Execution of experiments defined in experiments.py.

##### Training Logic

* *train.py:* Training script for a specific configuration. It contains the main training logic for self-supervised
depth estimation, semi-supervised semantic segmentation, and DepthMix.
* *label_selection.py:* Logic for automatic data selection for annotation.
* *monodepth_loss.py:* Loss for self-supervised depth estimation.

##### Models

* *models/joint_segmentation_depth.py:* Combined model for depth estimation, pose prediction, and semantic segmentation.
* *models/joint_segmentation_depth_decoder.py:* Segmentation decoders for transfer learning from self-supervised depth
and multi-task learning.
* *models/depth_decoder.py:* Multi-scale depth decoder.
* *models/monodepth_layers.py:* Operations necessary for self-supervised depth estimation.

##### Data Loading

* *loader/sequence_segmentation_loader.py:* Base class for loading image sequences with segmentation labels.
* *loader/cityscapes_loader.py:* Implementation for loading Cityscapes.
* *loader/depth_estimator.py:* Generate depth estimates from pretrained self-supervised depth model and store them
that they can be loaded by sequence_segmentation_loader as pseudo depth label.
* *loader/preselected_labels.py:* A selection of annotated samples obtained with label_selection.py

