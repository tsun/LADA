# Local Context-Aware Active Domain Adaptation               

Pytorch implementation of LADA. 
> [Local Context-Aware Active Domain Adaptation](https://arxiv.org/abs/2208.12856)                 
> Tao Sun, Cheng Lu, and Haibin Ling                 
> *ICCV 2023* 
> 
## Abstract
Active Domain Adaptation (ADA) queries the labels of a small number of selected target samples to help adapting a model from a source domain to a target domain. The local context of queried data is important, especially when the domain gap is large. However, this has not been fully explored by existing ADA works. 

In this paper, we propose a Local context-aware ADA framework, named LADA, to address this issue. To select informative target samples, we devise a novel criterion based on the local inconsistency of model predictions. Since the labeling budget is usually small, fine-tuning model on only queried data can be inefficient. We progressively augment labeled target data with the confident neighbors in a class-balanced manner. 

Experiments validate that the proposed criterion chooses more informative target samples than existing active selection strategies. Furthermore, our full method surpasses recent ADA arts on various benchmarks. 
<p align="center">
    <img src="fig/framework.png" width="900"> <br>
</p>


## Usage
### Prerequisites
We experimented with python==3.8, pytorch==1.8.0, cudatoolkit==11.1. 

To start, download the [office31](https://faculty.cc.gatech.edu/~judy/domainadapt/), [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html), [VisDA](https://ai.bu.edu/visda-2017/) datasets and set up the path in ./data folder.

### Supported methods
| Active Criteria |                                                                                 Paper                                                                                 |           Implementation           |
|-----------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------:|
| Random          |                                                                                   -                                                                                   |  [random](active/sampler.py)      | 
| Entropy         |                                                                                   -                                                                                   |  [entropy](active/sampler.py)      |
| Margin          |                                                                                   -                                                                                   |  [margin](active/sampler.py)      |
| LeastConfidence |                                                                                   -                                                                                   |  [leastConfidence](active/sampler.py)      | 
| CoreSet         |                                                         [ICLR 2018](https://openreview.net/pdf?id=H1aIuk-RW)                                                          |  [coreset](active/sampler.py)      |
| AADA            |                    [WACV 2020](https://openaccess.thecvf.com/content_WACV_2020/papers/Su_Active_Adversarial_Domain_Adaptation_WACV_2020_paper.pdf)                    |  [AADA](active/sampler.py)      |
| BADGE           |                                                         [ICLR 2020](https://openreview.net/pdf?id=ryghZJBKPS)                                                         |  [BADGE](active/sampler.py)      |                              
| CLUE            | [ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Prabhu_Active_Domain_Adaptation_via_Clustering_Uncertainty-Weighted_Embeddings_ICCV_2021_paper.pdf) |  [CLUE](active/sampler.py)      |
| MHP             |     [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_MHPL_Minimum_Happy_Points_Learning_for_Active_Source_Free_Domain_CVPR_2023_paper.pdf)      | [MHP](active/MHPsampler.py) |
| LAS (ours)      |                                                             [ICCV 2023](https://arxiv.org/abs/2208.12856)                                                             | [LAS](active/LASsampler.py) |


| Domain Adaptation             |                                                                              Paper                                                                               |        Implementation        |
|-------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------:|
| Fine-tuning (joint label set) |                                                                                -                                                                                 | [ft_joint](active/solver.py) |
| Fine-tuning                   |                                                                                -                                                                                 |    [ft](active/solver.py)    |
| DANN                          |      [JMLR 2016](https://jmlr.org/papers/volume17/15-239/15-239.pdf)                                                                           |   [dann](active/solver.py)   |
| MME                           |       [ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Saito_Semi-Supervised_Domain_Adaptation_via_Minimax_Entropy_ICCV_2019_paper.pdf)        |  [mme](active/MMEsolver.py)  |
| MCC                           |                                       [ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660460.pdf)                                        |  [MCC](active/MCCsolver.py)  |
| CDAC                          | [CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Cross-Domain_Adaptive_Clustering_for_Semi-Supervised_Domain_Adaptation_CVPR_2021_paper.pdf) | [CDAC](solver/CDACsolver.py) | 
| RAA (ours)                    |                                                          [ICCV 2023](https://arxiv.org/abs/2208.12856)                                                           |  [RAA](solver/PAAsolver.py)  | 
| LAA (ours)                    |                                                          [ICCV 2023](https://arxiv.org/abs/2208.12856)                                                           |  [LAA](solver/PAAsolver.py)  | 




### Training
To obtain results of baseline active selection criteria on office home with 5% labeling budget,
```shell
for ADA_DA in 'ft' 'mme'; do
  for ADA_AL in 'random' 'entropy' 'margin' 'coreset' 'leastConfidence' 'BADGE' 'AADA' 'CLUE' 'MHP'; do
    python main.py --cfg configs/officehome.yaml --gpu 0 --log log/oh/baseline  ADA.AL $ADA_AL  ADA.DA $ADA_DA
  done
done
```

To reproduce results of LADA on office home with 5% labeling budget,
```shell
# LAS + fine-tuning with CE loss
python main.py --cfg configs/officehome.yaml --gpu 0 --log log/oh/LADA  ADA.AL LAS  ADA.DA ft
# LAS + MME model adaptation
python main.py --cfg configs/officehome.yaml --gpu 0 --log log/oh/LADA  ADA.AL LAS  ADA.DA mme
# LAS + Random Anchor set Augmentation (RAA)
python main.py --cfg configs/officehome.yaml --gpu 0 --log log/oh/LADA  ADA.AL LAS  ADA.DA RAA
# LAS + Local context-aware Anchor set Augmentation (LAA)
python main.py --cfg configs/officehome.yaml --gpu 0 --log log/oh/LADA  ADA.AL LAS  ADA.DA LAA 
```

More commands can be found in *run.sh*.

## Acknowledgements
The pipline and implementation of baseline methods are adapted from [CLUE](https://github.com/virajprabhu/CLUE) and [deep-active-learning](https://github.com/ej0cl6/deep-active-learning). We adopt configuration files as [EADA](https://github.com/BIT-DA/EADA).


## Citation
If you find our paper and code useful for your research, please consider citing
```bibtex
@article{sun2022local,
    author    = {Sun, Tao and Lu, Cheng and Ling, Haibin},
    title     = {Local Context-Aware Active Domain Adaptation},
    journal   = {IEEE/CVF International Conference on Computer Vision},
    year      = {2023}
}
```