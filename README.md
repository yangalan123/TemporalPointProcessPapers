# Must-read Papers on Temporal Point Process (TPP)
![](https://img.shields.io/badge/PRs-Welcome-red)

Mainly Contributed by Chenghao Yang, Hongyuan Mei and Jason Eisner.

Thanks for all great [contributors](#acknowledgements) on GitHub!

### Contents
* [0. Toolkits](#0-toolkits)
* [1. Survey Papers](#1-survey-papers)
* [2. Modeling Papers](#2-modeling-papers)
    * [2.1 Hawkes Process Modeling](#21-hawkes-process-modeling)
    * [2.2 Neural-Symbolic Methods](#22-neural-symobolic-methods)
* [3. Algorithm Papers](#3-algorithm-papers)
* [4. Application Papers](#4-application-papers)
* [Acknowledgements](#acknowledgements)

## 0. Toolkits
1. **PoPPy: A Point Process Toolbox Based on PyTorch**. *Hongteng Xu*. arXiv 2018. [[website](https://github.com/HongtengXu/PoPPy)] [[paper](https://arxiv.org/pdf/1810.10122.pdf)]
1. **THAP: A Matlab Toolkit for Learning with Hawkes Processes**. *Hongteng Xu, Hongyuan Zha*. arXiv 2017. [[website](https://github.com/HongtengXu/Hawkes-Process-Toolkit)] [[pdf](https://arxiv.org/pdf/1708.09252.pdf)]
1. **Tick: a Python library for statistical learning, with a particular emphasis on time-dependent modelling**. *Emmanuel Bacry, Martin Bompaire, Stéphane Gaïffas, Soren Poulsen*. arXiv 2017. [[website](https://x-datainitiative.github.io/tick/index.html)] [[pdf](https://arxiv.org/pdf/1707.03003.pdf)]

## 1. Survey Papers
1. **Neural Temporal Point Processes: A Review**. *Oleksandr Shchur, Ali Caner Türkmen, Tim Januschowski, Stephan Günnemann*. IJCAI 2021. [[pdf](https://arxiv.org/pdf/2104.03528.pdf)]
1. **Recent Advance in Temporal Point Process: from Machine Learning Perspective**. *Junchi Yan*. SJTU Technical Report 2019. [[pdf](https://thinklab.sjtu.edu.cn/src/pp_survey.pdf)]

## 2. Modeling Papers


### 2.1 Hawkes Process Modeling
1. **Transformer Hawkes Process**. *Simiao Zuo, Haoming Jiang, Zichong Li, Tuo Zhao, Hongyuan Zha*. ICML 2020. [[pdf](https://arxiv.org/pdf/2002.09291.pdf)] [[code](https://github.com/SimiaoZuo/Transformer-Hawkes-Process)]
1. **Self-Attentive Hawkes Process**. *Qiang Zhang, Aldo Lipani, Omer Kirnap, Emine Yilmaz*. ICML 2020. [[pdf](https://arxiv.org/pdf/1907.07561.pdf)] [[code](https://github.com/QiangAIResearcher/sahp_repo)]
1. **Fully Neural Network based Model for General Temporal Point Processes**. *Takahiro Omi, Naonori Ueda, Kazuyuki Aihara*. NeurIPS 2019. [[pdf](https://arxiv.org/pdf/1905.09690.pdf)] [[code](https://github.com/omitakahiro/NeuralNetworkPointProcess)]
1. **The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process**. *Hongyuan Mei, Jason Eisner*. NeurIPS 2017. [[pdf](https://arxiv.org/pdf/1612.09328.pdf)] [[code](https://github.com/HMEIatJHU/neurawkes)] [[spotlight](https://www.cs.jhu.edu/~hmei/papers/mei+eisner.nips17.video.html)]
1. **Wasserstein Learning of Deep Generative Point Process Models**. *Shuai Xiao, Mehrdad Farajtabar, Xiaojing Ye, Junchi Yan, Le Song, Hongyuan Zha*. NeurIPS 2017. [[pdf](https://arxiv.org/pdf/1705.08051.pdf)] [[code]()]
1. **Recurrent Marked Temporal Point Processes: Embedding Event History to Vector**. *Nan Du, Hanjun Dai, Rakshit Trivedi, Utkarsh Upadhyay, Manuel Gomez-Rodriguez, and Le Song*. KDD 2016. [[pdf](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf)] [[code](https://github.com/dunan/NeuralPointProcess)]
1. **Isotonic Hawkes Processes**. *Yichen Wang, Bo Xie, Nan Du, Le Song*. ICML 2016. [[pdf](http://proceedings.mlr.press/v48/wangg16.pdf)]
1. **Hawkes Processes with Stochastic Excitations**. *Young Lee, Kar Wai Lim, Cheng Soon Ong*. ICML 2016. [[pdf](https://arxiv.org/pdf/1609.06831.pdf)]
1. **Learning Triggering Kernels for Multi-dimensional Hawkes Processes**. *Ke Zhou, Hongyuan Zha, and Le Song*. ICML 2013. [[paper](http://proceedings.mlr.press/v28/zhou13.pdf)]

### 2.2 Neural-Symbolic Methods
1. **Neural Datalog Through Time: Informed Temporal Modeling via Logical Specification**. *Hongyuan Mei, Guanghui Qin, Minjie Xu, Jason Eisner*. ICML 2020. [[pdf](https://arxiv.org/pdf/2006.16723.pdf)] [[code](https://github.com/HMEIatJHU/neural-datalog-through-time)] [[slides](https://www.cs.jhu.edu/~hmei/papers/mei+qin+xu+eisner.icml20.talk.pdf)]
1. **Temporal Logic Point Processes**。 *Shuang Li, Lu Wang, Ruizhi Zhang, Xiaofu Chang, Xuqin Liu, Yao Xie, Yuan Qi, Le Song*. ICML 2020. [[pdf](http://proceedings.mlr.press/v119/li20p/li20p.pdf)] [[slideslive](https://slideslive.com/38922890/temporal-logic-point-processes)]

## 3. Algorithm Papers
1. **Noise-Contrastive Estimation for Multivariate Point Processes**. *Hongyuan Mei, Tom Wan, Jason Eisner*. NeurIPS 2020. [[pdf](https://arxiv.org/pdf/2011.00717.pdf)] [[code](https://github.com/HMEIatJHU/nce-mpp)] [[slides](https://www.cs.jhu.edu/~hmei/papers/mei+wan+eisner.neurips20.talk.pdf)]
1. **Imputing Missing Events in Continuous-Time Event Streams**. *Hongyuan Mei, Guanghui Qin, Jason Eisner*. ICML 2019. [[pdf](https://arxiv.org/pdf/1905.05570.pdf)] [[code](https://github.com/HMEIatJHU/neurawkes)] [[slides](https://www.cs.jhu.edu/~hmei/papers/mei+qin+eisner.icml19.talk.pdf)]
1. **Learning Temporal Point Processes via Reinforcement Learning**. *Shuang Li, Shuai Xiao, Shixiang Zhu, Nan Du, Yao Xie, Le Song*. NeurIPS 2018. [[pdf](https://papers.nips.cc/paper/2018/file/5d50d22735a7469266aab23fd8aeb536-Paper.pdf)]

## 4. Application Papers
1. **Personalized Dynamic Treatment Regimes in Continuous Time: A Bayesian Joint Model for Optimizing Clinical Decisions with Timing**. *William Hua, Hongyuan Mei, Sarah Zohar, Magali Giral, Yanxun Xu*. Bayesian Analysis (2021). [[pdf](https://arxiv.org/pdf/2007.04155.pdf)] [[code](https://github.com/YanxunXu/doct)]
1. **Neural Temporal Point Processes For Modelling Electronic Health Records**. *Joseph Enguehard, Dan Busbridge, Adam Bozson, Claire Woodcock, Nils Hammerla*. Machine Learning for Health 2020. [[pdf](http://proceedings.mlr.press/v136/enguehard20a/enguehard20a.pdf)]
1. **A Dirichlet Mixture Model of Hawkes Processes for Event Sequence Clustering**. *Hongteng Xu, Hongyuan Zha*. NeurIPS 2017. [[pdf](https://arxiv.org/pdf/1701.09177)] 
1. **Learning Granger Causality for Hawkes Processes**. *Hongteng Xu, Mehrdad Farajtabar, Hongyuan Zha*. ICML 2016. [[pdf](https://arxiv.org/pdf/1602.04511.pdf)]
1. **Hawkes processes for Continuous Time Sequence Classification: An Application to Rumour Stance Classification in Twitter.**. *Michal Lukasik, P. K. Srijith, Duy Vu, Kalina Bontcheva, Arkaitz Zubiaga, Trevor Cohn*. ACL 2016. [[pdf](https://aclanthology.org/P16-2064.pdf)] [[code](https://github.com/mlukasik/seqhawkes)]
1. **Learning Network of Multivariate Hawkes Processes: A Time Series Approach**. *Jalal Etesami, Negar Kiyavash, Kun Zhang, Kushagra Singhal*. UAI 2016. [[pdf](https://arxiv.org/pdf/1603.04319)]
1. **The Bayesian Echo Chamber: Modeling Social Influence via Linguistic Accommodation**. *Fangjian Guo, Charles Blundell, Hanna Wallach, Katherine Heller*. AISTATS 2015. [[pdf](https://arxiv.org/pdf/1411.2674)]
1. **Constructing Disease Network and Temporal Progression Model via Context-sensitive Hawkes Process**. *Edward Choi, Nan Du, Robert Chen, Le Song, Jimeng Sun*. ICDM 2015. [[pdf](https://ieeexplore.ieee.org/document/7373379)]
1. **Time-sensitive Recommendation from Recurrent User Activities**. *Nan Du, Yichen Wang, Niao He, Jimeng Sun, and Le Song*. NeurIPS 2015. [[pdf](https://papers.nips.cc/paper/2015/file/136f951362dab62e64eb8e841183c2a9-Paper.pdf)]
1. **Dirichlet-Hawkes Processes with Applications to Clustering Continuous-time Document Streams**. *Nan Du, Mehrdad Farajtabar, Amr Ahmed, Alexander J Smola, Le Song*. KDD 2015. [[pdf](https://dl.acm.org/doi/pdf/10.1145/2783258.2783411)]
1. **Hawkestopic: A Joint Model for Network Inference and Topic Modeling from Text-based Cascades**. *Xinran He, Theodoros Rekatsinas, James Foulds, Lise Getoor, Yan Liu*. ICML 2015. [[pdf](http://proceedings.mlr.press/v37/he15.pdf)]
1. **Mixture of Mutually Exciting Processes for Viral Diffusion**. *Shuang-Hong Yang, Hongyuan Zha*. ICML 2013. [[pdf](http://proceedings.mlr.press/v28/yang13a.pdf)]

## Acknowledgements
Great thanks to other contributors AAA, BBB, CCC! (names are not listed in particular order)

Please contact us if we miss your names in this list, we will add you back ASAP!