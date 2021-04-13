<img src='https://github.com/Ha0Tang/BiGraphGAN/blob/master/imgs/face_results.jpeg' width=1200>

## Installation

Clone this repo.
```bash
git clone https://github.com/Ha0Tang/BiGraphGAN
cd BiGraphGAN/facial
```

## Dataset Preparation

Please follow [C2GAN](https://github.com/Ha0Tang/C2GAN#dataset-preparation) to directly download the facial dataset.

This repository uses the same dataset format as [SelectionGAN](https://github.com/Ha0Tang/SelectionGAN/tree/master/person_transfer#data-preperation) and [C2GAN](https://github.com/Ha0Tang/C2GAN#dataset-preparation). so you can use the same data for all these methods.

## Generating Images Using Pretrained Model
```bash
cd scripts/
sh download_bigraphgan_model.sh facial
```
Then,
1. Change several parameters in `test_facial.sh`.
2. Run `sh test_facial.sh` for testing.

## Train and Test New Models
1. Change several parameters in `train_facial.sh`.
2. Run `sh train_facial.sh` for training.
3. Change several parameters in `test_facial.sh`.
4. Run `sh test_facial.sh` for testing.

## Download Images Produced by the Authors
**For your convenience, you can directly download the images produced by the authors for qualitative comparisons in your own papers!!!**

```bash
cd scripts/
sh download_bigraphgan_result.sh facial
```

## Evaluation
We adopt SSIM, PSNR, LPIPS for evaluation of Market-1501. Please refer to [C2GAN](https://github.com/Ha0Tang/C2GAN) for more details.
 
## Acknowledgments
This source code is inspired by both [C2GAN](https://github.com/Ha0Tang/C2GAN), and [SelectionGAN](https://github.com/Ha0Tang/SelectionGAN). 

## Related Projects
**[XingGAN](https://github.com/Ha0Tang/XingGAN) | [GestureGAN](https://github.com/Ha0Tang/GestureGAN) | [C2GAN](https://github.com/Ha0Tang/C2GAN) | [SelectionGAN](https://github.com/Ha0Tang/SelectionGAN) | [Guided-I2I-Translation-Papers](https://github.com/Ha0Tang/Guided-I2I-Translation-Papers)**

## Citation
If you use this code for your research, please cite our papers.

BiGraphGAN
```
@inproceedings{tang2020bipartite,
  title={Bipartite Graph Reasoning GANs for Person Image Generation},
  author={Tang, Hao and Bai, Song and Torr, Philip HS and Sebe, Nicu},
  booktitle={BMVC},
  year={2020}
}
```

If you use the original [XingGAN](https://github.com/Ha0Tang/XingGAN), [GestureGAN](https://github.com/Ha0Tang/GestureGAN), [C2GAN](https://github.com/Ha0Tang/C2GAN), and [SelectionGAN](https://github.com/Ha0Tang/SelectionGAN) model, please cite the following papers:

XingGAN
```
@inproceedings{tang2020xinggan,
  title={XingGAN for Person Image Generation},
  author={Tang, Hao and Bai, Song and Zhang, Li and Torr, Philip HS and Sebe, Nicu},
  booktitle={ECCV},
  year={2020}
}
```

GestureGAN
```
@article{tang2019unified,
  title={Unified Generative Adversarial Networks for Controllable Image-to-Image Translation},
  author={Tang, Hao and Liu, Hong and Sebe, Nicu},
  journal={IEEE Transactions on Image Processing (TIP)},
  year={2020}
}

@inproceedings{tang2018gesturegan,
  title={GestureGAN for Hand Gesture-to-Gesture Translation in the Wild},
  author={Tang, Hao and Wang, Wei and Xu, Dan and Yan, Yan and Sebe, Nicu},
  booktitle={ACM MM},
  year={2018}
}
```

C2GAN
```
@inproceedings{tang2019cycleincycle,
  title={Cycle In Cycle Generative Adversarial Networks for Keypoint-Guided Image Generation},
  author={Tang, Hao and Xu, Dan and Liu, Gaowen and Wang, Wei and Sebe, Nicu and Yan, Yan},
  booktitle={ACM MM},
  year={2019}
}
```

SelectionGAN
```
@inproceedings{tang2019multi,
  title={Multi-channel attention selection gan with cascaded semantic guidance for cross-view image translation},
  author={Tang, Hao and Xu, Dan and Sebe, Nicu and Wang, Yanzhi and Corso, Jason J and Yan, Yan},
  booktitle={CVPR},
  year={2019}
}

@article{tang2020multi,
  title={Multi-channel attention selection gans for guided image-to-image translation},
  author={Tang, Hao and Xu, Dan and Yan, Yan and Corso, Jason J and Torr, Philip HS and Sebe, Nicu},
  journal={arXiv preprint arXiv:2002.01048},
  year={2020}
}
```

## Contributions
If you have any questions/comments/bug reports, feel free to open a github issue or pull a request or e-mail to the author Hao Tang ([hao.tang@unitn.it](hao.tang@unitn.it)).
