# LowBridge
## Bridging the Inter-Domain Gap through Low-Level Features for Cross-Modal Medical Image Segmentation [[arXiv]](https://arxiv.org/abs/2505.11909)

**LowBridge** leverages **edge (low‑level structural features)** as domain‑invariant representation across modalities:
1. Train a generative model to reconstruct source‑style images from edge maps extracted from source images.
2. Train segmentation network on these reconstructed source images with original labels.
3. At inference: extract edges from target images → feed edges into pretrained generator to synthesize source‑style target images → predict segmentation mask with pretrained segmentor.

LowBridge achieves state‑of‑the‑art results on liver (CHAOS) and cardiac sub‑structure (MMWHS) segmentation, outperforming 10 existing SOTA UDA/DG approaches. It is **model‑agnostic**, compatible with various generative and segmentation backbones.

![Framework](https://github.com/JoshuaLPF/LowBridge/blob/main/Figure/framework.png)
> Figure: Overview of LowBridge pipeline (Training Phase & Testing Phase)


## Dataset Preparation
We evaluate our method on two public medical segmentation benchmarks:
1. **CHAOS** (Abdominal Liver Segmentation, MRI-CT) | ISBI 2019 CHAOS Challenge
   Download from: https://chaos.grand-challenge.org/Combined_Healthy_Abdominal_Organ_Segmentation/
2. **MMWHS 2017** (Cardiac Sub-structure Segmentation, MRI-CT)
   Download instructions: https://github.com/cchen-cc/SIFA#readme

## Data Resources
### Generation Stage
Edge maps, generated images and raw images for the generative training stage:
- Train set: [Baidu Cloud](https://pan.baidu.com/s/1KcQN7wzbShDsd9f7ahu7Yw?pwd=kr4m)
- Test set: [Baidu Cloud](https://pan.baidu.com/s/1nQZRFqnn9GRCjsPSRjEwbQ?pwd=56gx)

### Segmentation Stage
Synthesized images and corresponding ground-truth labels for segmentation training:
[Baidu Cloud](https://pan.baidu.com/s/1metjgnp90a54h43XHllCeA?pwd=4d6q)

## Pre-trained Checkpoints
- Generator checkpoint (generation stage): [Baidu Cloud](https://pan.baidu.com/s/1N-q0sw9N1bETwXRvyV5cFw?pwd=v2k8)
- Segmentor checkpoint (segmentation stage): [Baidu Cloud](https://pan.baidu.com/s/19P_JdtSvaDMjD7-aG_5FUw?pwd=yfky)

## Prediction Results
Segmentation outputs on test sets: [Baidu Cloud](https://pan.baidu.com/s/1EDIH7kWtO4Ex1HDwmtIu3A?pwd=wgpy)

## Please cite our paper if you find it useful for your research.
```
@article{lyu2025bridging,
  title={Bridging the inter-domain gap through low-level features for cross-modal medical image segmentation},
  author={Lyu, Pengfei and Yeung, Pak-Hei and Xia, Jing and Hu, De and Yu, Xiaosheng and Chi, Jianning and Wu, Chengdong and Rajapakse, Jagath C},
  journal={arXiv preprint arXiv:2505.11909},
  year={2025}
}
```

## Contact Us
If you have any questions, please contact us (lyupengfei1995@outlook.com).
