# Local Context-Aware Active Domain Adaptation               


## Abstract
Active Domain Adaptation (ADA) queries the label of selected target samples to help adapting a model from a related source domain to a target domain. It has attracted increasing attention recently due to its promising performance with minimal labeling cost. Nevertheless, existing ADA methods have not fully exploited the local context of queried data, which is important to ADA, especially when the domain gap is large.

In this paper, we propose a novel framework of Local context-aware Active Domain Adaptation (LADA), which is composed of two key modules. The Local context-aware Active Selection (LAS) module selects target samples whose class probability predictions are inconsistent with their neighbors. The Local context-aware Model Adaptation (LMA) module refines a model with both queried samples and their expanded neighbors, regularized by a context-preserving loss.

Extensive experiments show that LAS selects more informative samples than existing active selection strategies. Furthermore, equipped with LMA, the full LADA method outperforms state-of-the-art ADA solutions on various benchmarks.

<p align="center">
    <img src="fig/framework.png" width="900"> <br>
</p>


## Usage
### Prerequisites
We experimented with python==3.8, pytorch==1.8.0, cudatoolkit==11.1. 

To start, download the [office31](https://faculty.cc.gatech.edu/~judy/domainadapt/), [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html), [VisDA](https://ai.bu.edu/visda-2017/) datasets and set up the path in ./data folder.

### Training
To obtain results of baseline active selection criteria on office home,
```shell
for ADA_DA in 'ft' 'mme'; do
  for ADA_AL in 'random' 'entropy' 'margin' 'coreset' 'leastConfidence' 'BADGE' 'AADA' 'CLUE'; do
    python main.py --cfg configs/officehome.yaml --gpu 0 --log log/oh/baseline  ADA.AL $ADA_AL  ADA.DA $ADA_DA
  done
done
```

To reproduce results of LADA on office home,
```shell
# LAS + fine-tuning with CE loss
python main.py --cfg configs/officehome.yaml --gpu 0 --log log/oh/LADA  ADA.AL LADA  ADA.DA ft
# LAS + MME model adaptation
python main.py --cfg configs/officehome.yaml --gpu 0 --log log/oh/LADA  ADA.AL LADA  ADA.DA mme
# LAS + LMA w/o random data augmentation
python main.py --cfg configs/officehome.yaml --gpu 0 --log log/oh/LADA  ADA.AL LADA  ADA.DA LADA
# LAS + LMA w/ random data augmentation
python main.py --cfg configs/officehome.yaml --gpu 0 --log log/oh/LADA  ADA.AL LADA  ADA.DA LADA  LADA.A_RAND_NUM 1
```

## Acknowledgements
The pipline and implementation of baseline methods are adapted from [CLUE](https://github.com/virajprabhu/CLUE). We adopt configuration files as [EADA](https://github.com/BIT-DA/EADA).


## Citation
If you find our paper and code useful for your research, please consider citing
```bibtex
@article{sun2022prior,
    author    = {Sun, Tao and Lu, Cheng and Ling, Haibin},
    title     = {Local Context-Aware Active Domain Adaptation},
    journal   = {arXiv preprint arXiv:2208.12856},
    year      = {2022}
}
```