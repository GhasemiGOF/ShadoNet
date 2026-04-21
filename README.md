# ShadoNet: Shape-Aware Nucleus Detection and Classification for Ki-67 Histopathology

## Overview
ShadoNet is a deep learning framework for joint nucleus detection and classification in Ki-67â€“stained histopathology images.

Instead of relying on explicit segmentation, ShadoNet formulates the problem as a class-aware structured regression task, predicting proximity maps that encode nucleus centers, spatial structure, and class identity.

To incorporate morphology without requiring manual boundary annotations, ShadoNet introduces shape-aware supervision using:
- SAM-derived masks as geometric priors  
- Rotation-aware SIoU loss for orientation alignment  
- Hausdorff Distance Transform (HDT) loss for boundary consistency  

This design enables shape-informed learning while remaining efficient and annotation-light.

## Key Features
- Single-stage framework for detection + classification  
- No segmentation labels required  
- Class-specific proximity map regression  
- Shape-aware training via SAM priors  
- Robust in dense and heterogeneous tissue regions  
- Designed for Ki-67 histopathology (PanNET, breast cancer)  

## Repository Structure
```
ShadoNet/
â”‚â”€â”€ data/
â”‚â”€â”€ models/
â”‚â”€â”€ losses/
â”‚â”€â”€ utils/
â”‚â”€â”€ sam/
â”‚â”€â”€ train.py
â”‚â”€â”€ inference.py
â”‚â”€â”€ evaluation.py
â”‚â”€â”€ configs/
â”‚â”€â”€ checkpoints/
â”‚â”€â”€ README.md
```

## Installation
```bash
git clone https://github.com/your-username/ShadoNet.git
cd ShadoNet

conda create -n shadonet python=3.9
conda activate shadonet

pip install -r requirements.txt
```

## Training
```bash
python train.py --config configs/train.yaml
```

## Inference
```bash
python inference.py --checkpoint checkpoints/model.pth
```

## Evaluation
```bash
python evaluation.py
```

## Citation
```
@article{ghasemi2025shadonet,
  title={ShadoNet: A Nucleus Detection and Classification Framework for Ki-67 Pathology Images},
  author={Ghasemi, Mahsa and Xing, Fuyong and Cornish, Toby C and Ghosh, Debashis and Bian, Jiang and Zhang, Xuhong},
  journal={Bioinformatics},
  year={2025}
}
```

## Contact
zhangxuh@iu.edu
