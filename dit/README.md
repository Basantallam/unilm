# [DiT: Self-Supervised Pre-Training for Document Image Transformer](https://arxiv.org/abs/2203.02378)

DiT (Document Image Transformer) is a self-supervised pre-trained Document Image Transformer model using large-scale unlabeled text images for Document AI tasks, which is essential since no supervised counterparts ever exist due to the lack of human labeled document images. 
# In this repository:
## We demonstrate the evaluation of 2 DiT pre-trained models.

1. Document Classification DiT model
2. Graphical Document Object Detection DiT model

Additionally, we fine-tuned the Graphical Document Object Detection DiT model to perform the task of Form Understanding on FUNSD Dataset. We test and evaluate its performance. 

# Training

The training code for each of the three models architecture can be found in [here](https://github.com/Basantallam/unilm/tree/master/dit) which is divided into 2 directories:

1. [object detection](https://github.com/Basantallam/unilm/tree/master/dit/object_detection) contains the Graphical Document Object Detection DiT model configurations and training code as well as the FUNSD fine-tuned form understanding model configurations and training code.

2. [classification](https://github.com/Basantallam/unilm/tree/master/dit/classification) contains the Document Classification DiT model configurations and training code.

<div align="center">
  <img src="https://user-images.githubusercontent.com/45008728/157173825-0949218a-61f5-4acb-949b-bbc499ab49f2.png" width="500" /><img src="https://user-images.githubusercontent.com/45008728/157173843-796dc878-2607-48d7-85cb-f54a2c007687.png" width="500"/> Model outputs with PubLayNet (left) and ICDAR 2019 cTDaR (right)
</div>


## Pretrained models

We used DiT weights pretrained on [IIT-CDIP Test Collection 1.0](https://dl.acm.org/doi/10.1145/1148170.1148307). The models were pretrained with 224x224 resolution.

`DiT-base`: #layer=12; hidden=768; FFN factor=4x; #head=12; patch=16x16 (#parameters: 86M)


## Setup

First, clone the repo and install required packages:
```
git clone https://github.com/Basantallam/unilm
cd unilm/dit
pip install -r requirements.txt
```

The required packages including: [Pytorch](https://pytorch.org/) version 1.9.0, [torchvision](https://pytorch.org/vision/stable/index.html) version 0.10.0 and [Timm](https://github.com/rwightman/pytorch-image-models) version 0.5.4, etc.

For mixed-precision training, please install [apex](https://github.com/NVIDIA/apex)
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
For object detection, please additionally install detectron2 library and shapely. Refer to the [Detectron2's INSTALL.md](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md).

```bash
# Install `detectron2`
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

# Install `shapely`
pip install shapely
```

## Fine-tuning on RVL-CDIP (Document Image Classification)

We summarize used the following mdels. The detailed instructions to reproduce the results can be found at [`classification/README.md`](classification/README.md).

| name | initialized checkpoint | resolution | accuracy  | weight |
|------------|:----------------------------------------|:----------:|:-------:|-----|
| DiT-base | [dit_base_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-base-224-p16-500k-62d53a.pth) | 224x224 | 92.11 | [link](https://layoutlm.blob.core.windows.net/dit/dit-fts/rvlcdip_dit-b.pth) |


## Fine-tuning on PubLayNet (Document Layout Analysis)

We summarize used the following mdels. The detailed instructions to reproduce the results can be found at [`object_detection/README.md`](object_detection/README.md).

| name | initialized checkpoint | detection algorithm  |  mAP| weight |
|------------|:----------------------------------------|:----------:|-------------------|-----|
| DiT-base | [dit_base_patch16_224](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-base-224-p16-500k-62d53a.pth) | Mask R-CNN | 0.935 |  [link](https://layoutlm.blob.core.windows.net/dit/dit-fts/publaynet_dit-b_mrcnn.pth) |





```
@misc{li2022dit,
    title={DiT: Self-supervised Pre-training for Document Image Transformer},
    author={Junlong Li and Yiheng Xu and Tengchao Lv and Lei Cui and Cha Zhang and Furu Wei},
    year={2022},
    eprint={2203.02378},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```


## Acknowledgement

This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, the [detectron2](https://github.com/facebookresearch/detectron2) library, the [DeiT](https://github.com/facebookresearch/deit) repository, the [Dino](https://github.com/facebookresearch/dino) repository, the [BEiT](https://github.com/microsoft/unilm/tree/master/beit) repository and the [MPViT](https://github.com/youngwanLEE/MPViT) repository.


## License

The content of this project itself is licensed under the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)，[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)


### Contact Information

For help or issues using DiT models, please submit a GitHub issue.

For other communications related to DiT, please contact Lei Cui (`lecu@microsoft.com`), Furu Wei (`fuwei@microsoft.com`).
