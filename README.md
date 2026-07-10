# SGP-Net

This is the code of SGP-Net.

## Dependencies

Please install following essential dependencies:

```text
dcm2nii
json5==0.13.0
nibabel==2.5.1
opencv-python-headless==5.0.0.93
Pillow==9.0.1
sacred==0.8.2
scikit-image==0.18.3
SimpleITK==1.2.3
torch==1.11.0
torchvision==0.12.0
```

## Data sets and pre-processing

Download:
- **CHAOS-MRI**: [Combined Healthy Abdominal Organ Segmentation data set](https://chaos.grand-challenge.org/)
- **Synapse-CT**: [Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/218292)
- **CMR**: [Multi-sequence Cardiac MRI Segmentation data set](https://zmiclab.github.io/projects/mscmrseg19/) (bSSFP fold)

Pre-processing is performed according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535) and we follow the procedure on their github repository.

## Training

1. Compile `./data/supervoxels/felzenszwalb_3d_cy.pyx` with cython (`python ./data/supervoxels/setup.py build_ext --inplace`) and run `./data/supervoxels/generate_supervoxels.py`.
2. Download pre-trained ResNet-101 weights [vanilla version](https://download.pytorch.org/models/resnet101-63fe2227.pth) or [deeplabv3 version](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and put your checkpoints folder, then replace the absolute path in the code `./models/encoder.py`.
3. Run `./script/train.sh`.

## Inference

Run `./script/evaluate.sh`.
