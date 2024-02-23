# ZS6D

![teaser](./assets/overview.png)

We demonstrate the effectiveness of deep features extracted from self-supervised, pre-trained Vision Transformer (ViT) for Zero-shot 6D pose estimation. For more detailed information check out the corresponding [[paper](https://arxiv.org/pdf/2309.11986.pdf)].

![pipeline](./assets/ZS6D_pipeline.png)

## Installation:

### Docker setup:

### ROS integration:

## Template rendering:
To generate templates from a object model to perform inference, we refer to the [ZS6D_template_rendering](https://github.com/haberger/ZS6D_template_rendering) repository.

## Template preparation:


## Inference:

## Evaluation on BOP Datasets:


## Acknowledgements
This project is built upon [dino-vit-features](https://github.com/ShirAmir/dino-vit-features), which performed a very comprehensive study about features of self-supervised pretrained Vision Transformers and their applications, including local correspondence matching. Here is a link to their [paper](https://arxiv.org/abs/2112.05814). We thank the authors for their great work and repo.

## Citation
If you found this repository useful please consider starring ⭐ and citing :

```
@article{ausserlechner2023zs6d,
  title={ZS6D: Zero-shot 6D Object Pose Estimation using Vision Transformers},
  author={Ausserlechner, Philipp and Haberger, David and Thalhammer, Stefan and Weibel, Jean-Baptiste and Vincze, Markus},
  journal={arXiv preprint arXiv:2309.11986},
  year={2023}
}
```
