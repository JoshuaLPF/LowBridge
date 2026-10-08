# LowBridge
## Bridging the Inter-Domain Gap through Low-Level Features for Cross-Modal Medical Image Segmentation [[arXiv]](https://arxiv.org/abs/2505.11909)
- ![Framework](https://github.com/JoshuaLPF/LowBridge/blob/main/Figure/framework.png)
- LowBridge leverages edge (low‑level structural features) as domain‑invariant representation across modalities:
1, Train a generative model to reconstruct source‑style images from edge maps extracted from source images.
2, Train segmentation network on these reconstructed source images with original labels.
3, At inference: extract edges from target images → feed edges into pretrained generator to synthesize source‑style target images → predict segmentation mask with pretrained segmentor.
- LowBridge achieves state‑of‑the‑art results on liver (CHAOS) and cardiac sub‑structure (MMWHS) segmentation, outperforming 10 existing SOTA UDA/DG approaches. It is model‑agnostic, compatible with various generative and segmentation backbones.
- Please cite our paper if you find it useful for your research.
```
@article{lyu2025efficient,
  title={Efficient Fourier Filtering Network with Contrastive Learning for UAV-based Unaligned Bi-modal Salient Object Detection},
  author={Lyu, Pengfei and Yeung, Pak-Hei and Yu, Xiaosheng and Cheng, Xiufei and Wu, Chengdong and Rajapakse, Jagath C},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  publisher={IEEE}
}
```

## Requirements

List of prerequisites or required libraries for the project to run:

- Pytorch 2.0.0
- Cuda 11.8
- Python 3.8 or higher
- tensorboardX
- opencv-python
- timm==0.6.13
- thop
- numpy

## Datasets in paper
- Please resize the bimodal images to the same size before training.
- UAV RGB-T 2400: [link](https://github.com/VDT-2048/UAV-RGB-T-2400);
- UNVT821, UNVT1000, UNVT5000: [link](https://github.com/lz118/Deep-Correlation-Network).

## Results
The results of our AlignSal and other SOTA models
### UAV RGB-T 2400:
- AlignSal: [link](https://pan.baidu.com/s/1M2xWybKfdOV3GLhnxFQlQg?pwd=rxyj);
- Comparison model: [link](https://pan.baidu.com/s/165OwbmbMzwb5gPvwzBSpOQ?pwd=28f5).
### UNVT821, UNVT1000, UNVT5000
- AlignSal: [link](https://pan.baidu.com/s/1hhboN8oskn4JPgXPgZ6kaA?pwd=8fvr);
- Comparison model: [link](https://pan.baidu.com/s/1oHcMoWgNS_0Ep43fegFUNA?pwd=nuns).

## Evaluation Metrics Toolbox
- The Evaluation Metrics Toolbox is available here: [link](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox).

## Acknowledgements
- Thanks to all the seniors, and projects (*e.g.*, [MROS](https://github.com/VDT-2048/UAV-RGB-T-2400), [ContrastAlign](https://github.com/modaxiansheng/ContrastAlign/), [DCNet](https://github.com/lz118/Deep-Correlation-Network), and [SwinNet](https://github.com/liuzywen/SwinNet)).

## Contact Us
If you have any questions, please contact us (lvpengfei1995@163.com).
