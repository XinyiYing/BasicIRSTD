# RDIAN

Official implementation for TGRS paper :
“Receptive-field and Direction Induced Attention Network for Infrared Dim Small Target Detection with a Large-scale Dataset IRDST”

## Citation

Please cite our paper in your publications if our work helps your research. BibTeX reference is as follows.

```
@article{TGRS23RDIAN,
 author={Sun, Heng and Bai, Junxiang and Yang, Fan and Bai, Xiangzhi},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Receptive-Field and Direction Induced Attention Network for Infrared Dim Small Target Detection With a Large-Scale Dataset IRDST}, 
  year={2023},
  volume={61},
  number={},
  pages={1-13},
  doi={10.1109/TGRS.2023.3235150}
}
```

And if the implementation of this repo is helpful to you, just star it.

## Requirements

* Python 3.9
* Pytorch 1.8.0

## Dataset

The IRDST dataset: <http://xzbai.buaa.edu.cn/datasets.html>

## Experiments 

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with a single GeForce RTX 3090 GPU of 24 GB Memory.

The trained model params are in `./params`

## Training
```
python train.py 
```

## Testing
```
python test.py
```

## License
MIT License
