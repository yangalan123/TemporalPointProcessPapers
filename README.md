# Recommended Reading on Temporal Point Process (TPP)
![](https://img.shields.io/badge/PRs-Welcome-red)

Mainly Contributed by [Chenghao Yang](https://yangalan123.github.io/), [Hongyuan Mei](https://www.hongyuanmei.com/) and [Jason Eisner](https://www.cs.jhu.edu/~jason/).

Thanks for all great [contributors](#acknowledgements) on GitHub!

### Contents
* [0. Toolkits](#0-toolkits)
* [1. Survey Papers](#1-survey-papers)
* [2. Modeling Papers](#2-modeling-papers)
    * [2.1 Temporal Point Process Modeling](#21-temporal-point-process-modeling)
    * [2.2 Structured Temporal Point Process Modeling](#22-structured-temporal-point-process-modeling)
* [3. Algorithm Papers](#3-algorithm-papers)
    * [3.1 Training Algorithm](#31-training-algorithm)
    * [3.2 Imputing](#32-imputing)
    * [3.3 Anomaly Detection](#33-anomaly-detection)
    * [3.4 Clustering](#34-clustering)
    * [3.5 Data Augmentation](#35-data-augmentation)
    * [3.6 Denoising](#36-denoising)
    * [3.7 Querying](#37-querying)
* [4. Application Papers](#4-application-papers)
    * [4.1 Social Media](#41-social-media)
    * [4.2 Clinical Health](#42-clinical-health)
    * [4.3 E-commerce](#43-e-commerce)
    * [4.4 Causality Discovery](#44-causality-discovery)
    * [4.5 Audio Processing](#45-audio-processing)
    * [4.6 Natural Language Processing](#46-natural-language-processing)
    * [4.7 Computer Vision](#47-computer-vision)
    * [4.8 Network Structure Discovery](#48-network-structure-discovery)
    * [4.9 Science](#49-science)
    * [4.10 Reinforcement Learning](#410-reinforcement-learning)
* [5. Benchmark](#5-benchmark)
* [6. Research Opportunities](#6-research-opportunities)
* [Acknowledgements](#acknowledgements)

## 0. Toolkits
1. **EasyTPP: Towards Open Benchmarking the Temporal Point Processes**. *Siqiao Xue, Xiaoming Shi, Zhixuan Chu, Yan Wang, Fan Zhou, Hongyan Hao, Caigao Jiang, Chen Pan, Yi Xu, James Y. Zhang, Qingsong Wen, Jun Zhou, Hongyuan Mei*. ICLR 2024. [[pdf](https://arxiv.org/pdf/2307.08097.pdf)] [[code](https://github.com/ant-research/easytemporalpointprocess)]
1. **TPPToolkits: Toolkits for Temporal Point Process**. *Hongyuan Mei, Chenghao Yang*. [[website](https://github.com/yangalan123/TPPToolkits)]
1. **PoPPy: A Point Process Toolbox Based on PyTorch**. *Hongteng Xu*. arXiv 2018. [[website](https://github.com/HongtengXu/PoPPy)] [[pdf](https://arxiv.org/pdf/1810.10122.pdf)]
1. **THAP: A Matlab Toolkit for Learning with Hawkes Processes**. *Hongteng Xu, Hongyuan Zha*. arXiv 2017. [[website](https://github.com/HongtengXu/Hawkes-Process-Toolkit)] [[pdf](https://arxiv.org/pdf/1708.09252.pdf)]
1. **Tick: a Python library for statistical learning, with a particular emphasis on time-dependent modelling**. *Emmanuel Bacry, Martin Bompaire, Stéphane Gaïffas, Soren Poulsen*. arXiv 2017. [[website](https://x-datainitiative.github.io/tick/index.html)] [[pdf](https://arxiv.org/pdf/1707.03003.pdf)]

## 1. Survey Papers
1. **Transformers in Time Series: A Survey**. *Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi Chen, Ziqing Ma, Junchi Yan, Liang Sun*. IJCAI 2023. [[pdf](https://arxiv.org/pdf/2202.07125.pdf)]
1. **Neural Temporal Point Processes: A Review**. *Oleksandr Shchur, Ali Caner Türkmen, Tim Januschowski, Stephan Günnemann*. IJCAI 2021. [[pdf](https://arxiv.org/pdf/2104.03528.pdf)]
1. **Recent Advance in Temporal Point Process: from Machine Learning Perspective**. *Junchi Yan*. SJTU Technical Report 2019. [[pdf](https://thinklab.sjtu.edu.cn/src/pp_survey.pdf)]

## 2. Modeling Papers


### 2.1 Temporal Point Process Modeling
1. **Neural Jump-Diffusion Temporal Point Processes.** *Shuai Zhang, Chuan Zhou, Yang Aron Liu, Peng Zhang, Xixun Lin, Zhi-Ming Ma*. ICML 2024. [[paper](https://openreview.net/pdf?id=d1P6GtRzuV)] [[code](https://github.com/Zh-Shuai/NJDTPP)]
1. **Language Models Can Improve Event Prediction by Few-Shot Abductive Reasoning.** *Xiaoming Shi, Siqiao Xue, Kangrui Wang, Fan Zhou, James Y. Zhang, Jun Zhou, Chenhao Tan, Hongyuan Mei*. NeurIPS 2023. [[paper](https://arxiv.org/pdf/2305.16646.pdf)] [[code](https://github.com/iLampard/lamp)]
1. **Prompt-augmented Temporal Point Process for Streaming Event Sequence.** *Siqiao Xue, Yan Wang, Zhixuan Chu, Xiaoming Shi, Caigao Jiang, Hongyan Hao, Gangwei Jiang, Xiaoyun Feng, James Y. Zhang, Jun Zhou*. NeurIPS 2023. [[paper](https://arxiv.org/pdf/2310.04993.pdf)] [[code](https://github.com/yanyanSann/PromptTPP)]
1. **Integration-free Training for Spatio-temporal Multimodal Covariate Deep Kernel Point Processes**. *Yixuan Zhang, Quyu Kong, Feng Zhou*. NeurIPS 2023. [[paper](https://arxiv.org/pdf/2310.05485.pdf)]
1. **Sparse Transformer Hawkes Process for Long Event Sequences**. *Zhuoqun Li, Mingxuan Sun*. ECML-PKDD 2023. [[Springer](https://link.springer.com/chapter/10.1007/978-3-031-43424-2_11)]
1. **Intensity-free Convolutional Temporal Point Process: Incorporating Local and Global Event Contexts**. *Wang-Tao Zhou, Zhao Kang, Ling Tian, Yi Su*. Information Sciences 2023.  [[pdf](https://arxiv.org/pdf/2306.14072.pdf)] 
1. **Meta Temporal Point Processes**. *Wonho Bae, Mohamed Osama Ahmed, Frederick Tung, Gabriel L. Oliveira*. ICLR 2023. [[pdf](https://arxiv.org/pdf/2301.12023.pdf)]
1. **HYPRO: A Hybridly Normalized Probabilistic Model for Long-Horizon Prediction of Event Sequences**. *Siqiao Xue, Xiaoming Shi, James Y Zhang, Hongyuan Mei*. NeurIPS 2022. [[pdf](https://arxiv.org/pdf/2210.01753.pdf)] [[code (iLampard)](https://github.com/iLampard/hypro_tpp)] [[code (alipay)](https://github.com/alipay/hypro_tpp)]
1. **Exploring Generative Neural Temporal Point Process**. *Haitao Lin, Lirong Wu, Guojiang Zhao, Pai Liu, Stan Z. Li*. TMLR 2022. [[pdf](https://arxiv.org/pdf/2208.01874.pdf)] [[code](https://github.com/BIRD-TAO/GNTPP)]
1. **Transformer Embeddings of Irregularly Spaced Events and Their Participants**. *Chenghao Yang, Hongyuan Mei, Jason Eisner*. ICLR 2022. [[pdf](https://arxiv.org/pdf/2201.00044.pdf)] [[code](https://github.com/yangalan123/anhp-andtt)]
1. **Long Horizon Forecasting with Temporal Point Processes**. *Prathamesh Deshpande, Kamlesh Marathe, Abir De, Sunita Sarawagi*. WSDM 2021. [[pdf](https://dl.acm.org/doi/10.1145/3437963.3441740)]
1. **Deep Fourier Kernel for Self-Attentive Point Processes**. *Shixiang Zhu, Minghe Zhang, Ruyi Ding, Yao Xie*. AISTATS 2021. [[pdf](http://proceedings.mlr.press/v130/zhu21b.html)]
1. **Neural Spatio-Temporal Point Processes**. *Ricky T. Q. Chen, Brandon Amos, Maximilian Nickel*. ICLR 2021. [[pdf](https://arxiv.org/pdf/2011.04583.pdf)] [[code](https://github.com/facebookresearch/neural_stpp)]
1. **Transformer Hawkes Process**. *Simiao Zuo, Haoming Jiang, Zichong Li, Tuo Zhao, Hongyuan Zha*. ICML 2020. [[pdf](https://arxiv.org/pdf/2002.09291.pdf)] [[code](https://github.com/SimiaoZuo/Transformer-Hawkes-Process)]
1. **Self-Attentive Hawkes Process**. *Qiang Zhang, Aldo Lipani, Omer Kirnap, Emine Yilmaz*. ICML 2020. [[pdf](https://arxiv.org/pdf/1907.07561.pdf)] [[code](https://github.com/QiangAIResearcher/sahp_repo)]
1. **Intensity-Free Learning of Temporal Point Processes**. *Oleksandr Shchur, Marin Biloš, Stephan Günnemann*. ICLR 2020. [[pdf](https://arxiv.org/abs/1909.12127)] [[code](https://github.com/shchur/ifl-tpp)]
1. **Fast and Flexible Temporal Point Processes with Triangular Maps**. *Oleksandr Shchur, Nicholas Gao, Marin Biloš, Stephan Günnemann*. NeurIPS 2020. [[pdf](https://arxiv.org/pdf/2006.12631.pdf)] [[code + data](https://www.in.tum.de/daml/triangular-tpp/)]
1. **Uncertainty on Asynchronous Time Event Prediction**. *Marin Biloš, Bertrand Charpentier, Stephan Günnemann*. NeurIPS 2019. [[pdf](https://papers.nips.cc/paper/2019/file/78efce208a5242729d222e7e6e3e565e-Paper.pdf)] [[code](https://github.com/sharpenb/Uncertainty-Event-Prediction)]
1. **Latent ODEs for Irregularly-Sampled Time Series**. *Yulia Rubanova, Ricky T. Q. Chen, David Duvenaud*. NeurIPS 2019. [[pdf](https://papers.nips.cc/paper/2019/file/42a6845a557bef704ad8ac9cb4461d43-Paper.pdf)] [[code](https://github.com/YuliaRubanova/latent_ode)]
1. **Neural Jump Stochastic Differential Equations**. *Junteng Jia, Austin R. Benson*. NeurIPS 2019. [[pdf](https://arxiv.org/pdf/1905.10403.pdf)] [[code](https://github.com/000Justin000/torchdiffeq/tree/jj585)]
1. **Fully Neural Network based Model for General Temporal Point Processes**. *Takahiro Omi, Naonori Ueda, Kazuyuki Aihara*. NeurIPS 2019. [[pdf](https://arxiv.org/pdf/1905.09690.pdf)] [[code](https://github.com/omitakahiro/NeuralNetworkPointProcess)]
1. **Deep Reinforcement Learning of Marked Temporal Point Processes**. *Utkarsh Upadhyay, Abir De, Manuel Gomez-Rodriguez*. NeurIPS 2018. [[pdf](https://arxiv.org/pdf/1805.09360.pdf)] [[code](https://github.com/Networks-Learning/tpprl)]
1. **Learning Conditional Generative Models for Temporal Point Processes**. *Shuai Xiao, Hongteng Xu, Junchi Yan, Mehrdad Farajtabar, Xiaokang Yang, Le Song, Hongyuan Zha*. AAAI 2018. [[pdf](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16163/16203)]
1. **The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process**. *Hongyuan Mei, Jason Eisner*. NeurIPS 2017. [[pdf](https://arxiv.org/pdf/1612.09328.pdf)] [[code](https://github.com/HMEIatJHU/neurawkes)] [[spotlight](https://www.cs.jhu.edu/~hmei/papers/mei+eisner.nips17.video.html)]
1. **Wasserstein Learning of Deep Generative Point Process Models**. *Shuai Xiao, Mehrdad Farajtabar, Xiaojing Ye, Junchi Yan, Le Song, Hongyuan Zha*. NeurIPS 2017. [[pdf](https://arxiv.org/pdf/1705.08051.pdf)] [[code](https://github.com/xiaoshuai09/Wasserstein-Learning-For-Point-Process)]
1. **Cascade Dynamics Modeling with Attention-based Recurrent Neural Network**. *Yongqing Wang, Huawei Shen, Shenghua Liu, Jinhua Gao, Xueqi Cheng*. IJCAI 2017. [[pdf](https://www.ijcai.org/proceedings/2017/0416.pdf)]
1. **Modeling The Intensity Function Of Point Process Via Recurrent Neural Networks**. *Shuai Xiao, Junchi Yan, Stephen M. Chu, Xiaokang Yang, Hongyuan Zha*. AAAI 2017. [[pdf](https://arxiv.org/pdf/1705.08982.pdf)] [[code](https://github.com/xiaoshuai09/Recurrent-Point-Process)]
1. **Recurrent Marked Temporal Point Processes: Embedding Event History to Vector**. *Nan Du, Hanjun Dai, Rakshit Trivedi, Utkarsh Upadhyay, Manuel Gomez-Rodriguez, and Le Song*. KDD 2016. [[pdf](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf)] [[code](https://github.com/dunan/NeuralPointProcess)]
1. **Isotonic Hawkes Processes**. *Yichen Wang, Bo Xie, Nan Du, Le Song*. ICML 2016. [[pdf](http://proceedings.mlr.press/v48/wangg16.pdf)]
1. **Hawkes Processes with Stochastic Excitations**. *Young Lee, Kar Wai Lim, Cheng Soon Ong*. ICML 2016. [[pdf](https://arxiv.org/pdf/1609.06831.pdf)]
1. **Learning Triggering Kernels for Multi-dimensional Hawkes Processes**. *Ke Zhou, Hongyuan Zha, and Le Song*. ICML 2013. [[pdf](http://proceedings.mlr.press/v28/zhou13.pdf)]

### 2.2 Structured Temporal Point Process Modeling
1. **Neuro-Symbolic Temporal Point Processes**. *Yang Yang, Chao Yang, Boyang Li, Yinghao Fu, Shuang Li*. ICML 2024. [[pdf](https://arxiv.org/pdf/2406.03914)]
1. **A Variational Autoencoder for Neural Temporal Point Processes with Dynamic Latent Graphs.** *Sikun Yang, Hongyuan Zha*. AAAI 2024. [[paper](https://arxiv.org/abs/2312.16083)]
1. **Transformer Embeddings of Irregularly Spaced Events and Their Participants**. *Chenghao Yang, Hongyuan Mei, Jason Eisner*. ICLR 2022. [[pdf](https://arxiv.org/pdf/2201.00044.pdf)] [[code](https://github.com/yangalan123/anhp-andtt)] (cross-listed as this paper covers both structured and unstructured TPP modeling. )
1. **Learning Neural Point Processes with Latent Graphs**. *Qiang Zhang, Aldo Lipani, and Emine Yilmaz*. WWW 2021. [[pdf](https://discovery.ucl.ac.uk/id/eprint/10122006/1/2021_WWW_Learning_Neural_Point_Processes_with_Latent_Graphs.pdf)]
1. **User-Dependent Neural Sequence Models for Continuous-Time Event Data**. *Alex Boyd, Robert Bamler, Stephan Mandt, Padhraic Smyth*. NeurIPS 2020. [[pdf](https://arxiv.org/pdf/2011.03231.pdf)] [[code](https://github.com/ajboyd2/vae_mpp)]
1. **Neural Datalog Through Time: Informed Temporal Modeling via Logical Specification**. *Hongyuan Mei, Guanghui Qin, Minjie Xu, Jason Eisner*. ICML 2020. [[pdf](https://arxiv.org/pdf/2006.16723.pdf)] [[code](https://github.com/HMEIatJHU/neural-datalog-through-time)] [[slides](https://www.cs.jhu.edu/~hmei/papers/mei+qin+xu+eisner.icml20.talk.pdf)]
1. **Temporal Logic Point Processes**. *Shuang Li, Lu Wang, Ruizhi Zhang, Xiaofu Chang, Xuqin Liu, Yao Xie, Yuan Qi, Le Song*. ICML 2020. [[pdf](http://proceedings.mlr.press/v119/li20p/li20p.pdf)] [[slideslive](https://slideslive.com/38922890/temporal-logic-point-processes)]
1. **DyRep: Learning Representations over Dynamic Graphs**. *Rakshit Trivedi, Mehrdad Farajtabar, Prasenjeet Biswal, Hongyuan Zha*. ICLR 2019. [[pdf](https://openreview.net/pdf?id=HyePrhR5KX)] [[code](https://github.com/uoguelph-mlrg/LDG)]
1. **Deep Mixture Point Processes: Spatio-temporal Event Prediction with Rich Contextual Information**. *Maya Okawa, Tomoharu Iwata, Takeshi Kurashima, Yusuke Tanaka, Hiroyuki Toda, Naonori Ueda*. KDD 2019. [[pdf](https://arxiv.org/pdf/1906.08952.pdf)]
1. **Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs**. *Rakshit Trivedi, Hanjun Dai, Yichen Wang, Le Song*. ICML 2017. [[pdf](https://proceedings.mlr.press/v70/trivedi17a/trivedi17a.pdf)]

## 3. Algorithm Papers
### 3.1 Training Algorithm
1. **Learning Multivariate Temporal Point Processes via the Time-Change Theorem**. *Guilherme Augusto Zagatti, See Kiong Ng, Stéphane Bressan*. AISTATS 2024. [[pdf](https://proceedings.mlr.press/v238/augusto-zagatti24a/augusto-zagatti24a.pdf)] [[code](https://github.com/NUS-IDS/multi-ttpp)]
1. **SMURF-THP: Score Matching-based UnceRtainty quantiFication for Transformer Hawkes Process**. *Zichong Li, Yanbo Xu, Simiao Zuo, Haoming Jiang, Chao Zhang, Tuo Zhao, Hongyuan Zha*. ICML 2023. [[pdf](https://arxiv.org/pdf/2310.16336.pdf)] [[code](https://github.com/zichongli5/SMURF-THP)]
1. **Noise-Contrastive Estimation for Multivariate Point Processes**. *Hongyuan Mei, Tom Wan, Jason Eisner*. NeurIPS 2020. [[pdf](https://arxiv.org/pdf/2011.00717.pdf)] [[code](https://github.com/HMEIatJHU/nce-mpp)] [[slides](https://www.cs.jhu.edu/~hmei/papers/mei+wan+eisner.neurips20.talk.pdf)]
1. **Learning Temporal Point Processes via Reinforcement Learning**. *Shuang Li, Shuai Xiao, Shixiang Zhu, Nan Du, Yao Xie, Le Song*. NeurIPS 2018. [[pdf](https://papers.nips.cc/paper/2018/file/5d50d22735a7469266aab23fd8aeb536-Paper.pdf)]
1. **INITIATOR: Noise-contrastive Estimation for Marked Temporal Point Process**. *Ruocheng Guo, Jundong Li, and Huan
Liu*. IJCAI 2018. [[pdf](https://www.ijcai.org/proceedings/2018/0303.pdf)]
1. **Improving Maximum Likelihood Estimation of Temporal Point Process via Discriminative and Adversarial Learning**. *Junchi Yan, Xin Liu, Liangliang Shi, Changsheng Li, Hongyuan Zha*. IJCAI 2018. [[pdf](https://www.ijcai.org/Proceedings/2018/0409.pdf)]

### 3.2 Imputing
1. **Imputing Missing Events in Continuous-Time Event Streams**. *Hongyuan Mei, Guanghui Qin, Jason Eisner*. ICML 2019. [[pdf](https://arxiv.org/pdf/1905.05570.pdf)] [[code](https://github.com/HMEIatJHU/neurawkes)] [[slides](https://www.cs.jhu.edu/~hmei/papers/mei+qin+eisner.icml19.talk.pdf)]

### 3.3 Anomaly Detection
1. **Detecting Anomalous Event Sequences with Temporal Point Processes**. *Oleksandr Shchur, Ali Caner Türkmen, Tim Januschowski, Jan Gasthaus, Stephan Günnemann*. NeurIPS 2021. [[pdf](https://arxiv.org/pdf/2106.04465.pdf)]
1. **Sequential Adversarial Anomaly Detection for One-Class Event Data**. *Shixiang Zhu, Henry Shaowu Yuchi, Yao Xie*. ICASSP 2020. [[pdf](https://arxiv.org/pdf/1910.09161.pdf)]

### 3.4 Clustering
1. **C-NTPP: Learning Cluster-Aware Neural Temporal Point Process**. *Fangyu Ding, Junchi Yan, Haiyang Wang*. AAAI 2023. [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/25897/25669)]
1. **Learning Mixture of Neural Temporal Point Processes for Multi-dimensional Event Sequence Clustering**. *Yunhao Zhang, Junchi Yan, Xiaolu Zhang, Jun Zhou, Xiaokang Yang*. IJCAI 2022. [[pdf](https://www.ijcai.org/proceedings/2022/0523.pdf)]
1. **A Dirichlet Mixture Model of Hawkes Processes for Event Sequence Clustering**. *Hongteng Xu, Hongyuan Zha*. NeurIPS 2017. [[pdf](https://arxiv.org/pdf/1701.09177)] [[code](https://github.com/HongtengXu/Hawkes-Process-Toolkit)]
1. **Definition of Distance for Nonlinear Time Series Analysis of Marked Point Process Data**. *Koji Iwayama, Yoshito Hirata, Kazuyuki Aihara*. Physics Letters A 2017. [[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S037596011631564X)] 

### 3.5 Data Augmentation
1. **Benefits from superposed hawkes processes**. *Hongteng Xu, Dixin Luo, Xu Chen, Lawrence Carin*. AISTATS 2018. [[pdf](https://arxiv.org/pdf/1710.05115.pdf)] [[code](https://github.com/HongtengXu/Hawkes-Process-Toolkit)]
1. **Learning Hawkes processes from short doubly-censored event sequences**. *Hongteng Xu, Dixin Luo, Hongyuan Zha*. ICML 2017. [[pdf](https://arxiv.org/pdf/1702.07013.pdf)]
1. **Transforming spatial point processes into Poisson processes using random superposition**. *Jesper Møller, Kasper K. Berthelsen*. Advances in Applied Probability 2012. [[pdf](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/AFDAE6427340C7D79D2168E5613F0F5C/S0001867800005449a.pdf/transforming-spatial-point-processes-into-poisson-processes-using-random-superposition.pdf)]

### 3.6 Denoising
1. **Learning Hawkes processes under synchronization noise**. *William Trouleau, Jalal Etesami, Matthias Grossglauser, Negar Kiyavash, Patrick Thiran*. ICML 2019. [[pdf](http://proceedings.mlr.press/v97/trouleau19a/trouleau19a.pdf)]
1. **Learning registered point processes from idiosyncratic observations**. *Hongteng Xu, Lawrence Carin, Hongyuan Zha*. ICML 2018. [[pdf](http://people.ee.duke.edu/~lcarin/Hongteng_ICML18.pdf)]

### 3.7 Querying
1. **Probabilistic Querying of Continuous-Time Event Sequences**. *Alex Boyd, Yuxin Chang, Stephan Mandt, Padhraic Smyth*. AISTATS 2023. [[pdf](https://proceedings.mlr.press/v206/boyd23a/boyd23a.pdf)] [[code](https://github.com/ajboyd2/point_process_queries)]

## 4. Application Papers

### 4.1 Social Media
1. **Identifying Coordinated Accounts on Social Media through Hidden Influence and Group Behaviours**. *Karishma Sharma, Yizhou Zhang, Emilio Ferrara, Yan Liu*. KDD 2021. [[pdf](https://arxiv.org/pdf/2008.11308.pdf)] [[code](https://github.com/USC-Melady/AMDN-HAGE-KDD21)]
1. **COEVOLVE: A Joint Point Process Model for Information Diffusion and Network Co-evolution**. *Mehrdad Farajtabar, Yichen Wang, Manuel Gomez Rodriguez, Shuang Li, Hongyuan Zha, Le Song*. JMLR 2017. [[pdf](https://www.jmlr.org/papers/volume18/16-132/16-132.pdf)] [[code](https://github.com/Networks-Learning/Coevolution)]
1. **DeepHawkes: Bridging the Gap between Prediction and Understanding of Information Cascades**. *Qi Cao, Huawei Shen, Keting Cen, Wentao Ouyang, Xueqi Cheng*. CIKM 2017. [[pdf](http://www.bigdatalab.ac.cn/~shenhuawei/publications/2017/cikm-cao.pdf)] [[code](https://github.com/CaoQi92/DeepHawkes)]
1. **The Bayesian Echo Chamber: Modeling Social Influence via Linguistic Accommodation**. *Fangjian Guo, Charles Blundell, Hanna Wallach, Katherine Heller*. AISTATS 2015. [[pdf](https://arxiv.org/pdf/1411.2674)]
1. **Learning Social Infectivity in Sparse Low-rank Networks Using Multi-dimensional Hawkes Processes**. *Ke Zhou, Hongyuan Zha, Le Song*. AISTATS 2013. [[pdf](http://proceedings.mlr.press/v31/zhou13a.pdf)]
### 4.2 Clinical Health
1. **Continuous-Time Decision Transformer for Healthcare Applications**. *Zhiyue Zhang, Hongyuan Mei, Yanxun Xu*. AISTATS 2023. [[pdf](https://proceedings.mlr.press/v206/zhang23i/zhang23i.pdf)] [[code](https://github.com/ZhiyueZ/CTDT)]
1. **Personalized Dynamic Treatment Regimes in Continuous Time: A Bayesian Joint Model for Optimizing Clinical Decisions with Timing**. *William Hua, Hongyuan Mei, Sarah Zohar, Magali Giral, Yanxun Xu*. Bayesian Analysis (2021). [[pdf](https://arxiv.org/pdf/2007.04155.pdf)] [[code](https://github.com/YanxunXu/doct)]
1. **Neural Temporal Point Processes For Modelling Electronic Health Records**. *Joseph Enguehard, Dan Busbridge, Adam Bozson, Claire Woodcock, Nils Hammerla*. Machine Learning for Health 2020. [[pdf](http://proceedings.mlr.press/v136/enguehard20a/enguehard20a.pdf)] [[code](https://github.com/babylonhealth/neuralTPPs)]
1. **Patient Flow Prediction via Discriminative Learning of Mutually-Correcting Processes**. *Hongteng Xu, Weichang Wu, Shamim Nemati, Hongyuan Zha*. TKDE 2016. [[pdf](https://arxiv.org/pdf/1602.05112.pdf)]
1. **Constructing Disease Network and Temporal Progression Model via Context-sensitive Hawkes Process**. *Edward Choi, Nan Du, Robert Chen, Le Song, Jimeng Sun*. ICDM 2015. [[pdf](https://ieeexplore.ieee.org/document/7373379)]
### 4.3 E-commerce
1. **Time is of the Essence: a Joint Hierarchical RNN and Point Process Model for Time and Item Predictions**. *Bjørnar Vassøy, Massimiliano Ruocco, Eliezer de Souza da Silva, Erlend Aune*. WSDM 2019. [[pdf](https://arxiv.org/pdf/1812.01276.pdf)] [[code](https://github.com/BjornarVass/Recsys)]
1. **Intermittent Demand Forecasting with Deep Renewal Processes**. *Ali Caner Turkmen, Yuyang Wang, Tim Januschowski*. TPP@NeurIPS 2019. [[pdf](https://arxiv.org/pdf/1911.10416.pdf)]
1. **Recurrent Spatio-Temporal Point Process for Check-in Time Prediction**. *Guolei Yang, Ying Cai, Chandan K. Reddy*. CIKM 2018. [[pdf](https://dl.acm.org/doi/pdf/10.1145/3269206.3272003)]
1. **Time-sensitive Recommendation from Recurrent User Activities**. *Nan Du, Yichen Wang, Niao He, Jimeng Sun, and Le Song*. NeurIPS 2015. [[pdf](https://papers.nips.cc/paper/2015/file/136f951362dab62e64eb8e841183c2a9-Paper.pdf)]
### 4.4 Causality Discovery
1. **Structural Hawkes Processes for Learning Causal Structure from Discrete-Time Event Sequences**. *Jie Qiao, Ruichu Cai, Siyu Wu, Yu Xiang, Keli Zhang, Zhifeng Hao*. IJCAI 2023. [[pdf](https://arxiv.org/abs/2305.05986)]
1. **CAUSE: Learning Granger Causality from Event Sequences using Attribution Methods**. *Wei Zhang, Thomas Kobber Panum, Somesh Jha, Prasad Chalasani, David Page*. ICML 2020. [[pdf](https://arxiv.org/pdf/2002.07906.pdf)] [[slideslive](https://slideslive.com/s/prasad-chalasani-30427)] [[code & data](https://github.com/razhangwei/CAUSE)]
1. **Uncovering Causality from Multivariate Hawkes Integrated Cumulants**. *Massil Achab, Emmanuel Bacry, Stéphane Gaïffas, Iacopo Mastromatteo, Jean-Francois Muzy*. ICML 2017. [[pdf](https://arxiv.org/pdf/1607.06333.pdf)] [[code](https://github.com/achab/nphc)]
1. **Graphical Modeling for Multivariate Hawkes Processes with Nonparametric Link Functions**. *Michael Eichler, Rainer Dahlhaus, Johannes Dueck*. Journal of Time Series Analysis 2017. [[pdf](https://arxiv.org/pdf/1605.06759.pdf)]
1. **Learning Granger Causality for Hawkes Processes**. *Hongteng Xu, Mehrdad Farajtabar, Hongyuan Zha*. ICML 2016. [[pdf](https://arxiv.org/pdf/1602.04511.pdf)]
1. **Learning Network of Multivariate Hawkes Processes: A Time Series Approach**. *Jalal Etesami, Negar Kiyavash, Kun Zhang, Kushagra Singhal*. UAI 2016. [[pdf](https://arxiv.org/pdf/1603.04319)]
### 4.5 Audio Processing
1. **Recurrent Poisson Process Unit for Speech Recognition**. *Hengguan Huang, Hao Wang, Brian Mak*. AAAI 2019. [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/4620/4498)]
### 4.6 Natural Language Processing
1. **Early Rumor Detection Using Neural Hawkes Process with a New Benchmark Dataset**. *Fengzhu Zeng, Wei Gao*. NAACL 2022. [[pdf](https://aclanthology.org/2022.naacl-main.302.pdf)] [[code & dataset](https://github.com/znhy1024/HEARD)]
1. **Hawkes processes for Continuous Time Sequence Classification: An Application to Rumour Stance Classification in Twitter**. *Michal Lukasik, P. K. Srijith, Duy Vu, Kalina Bontcheva, Arkaitz Zubiaga, Trevor Cohn*. ACL 2016. [[pdf](https://aclanthology.org/P16-2064.pdf)] [[code](https://github.com/mlukasik/seqhawkes)]
1. **Dirichlet-Hawkes Processes with Applications to Clustering Continuous-time Document Streams**. *Nan Du, Mehrdad Farajtabar, Amr Ahmed, Alexander J Smola, Le Song*. KDD 2015. [[pdf](https://dl.acm.org/doi/pdf/10.1145/2783258.2783411)]
1. **Hawkestopic: A Joint Model for Network Inference and Topic Modeling from Text-based Cascades**. *Xinran He, Theodoros Rekatsinas, James Foulds, Lise Getoor, Yan Liu*. ICML 2015. [[pdf](http://proceedings.mlr.press/v37/he15.pdf)]
### 4.7 Computer Vision
1. **Egocentric Activity Prediction via Event Modulated Attention**. *Yang Shen, Bingbing Ni, Zefan Li, Ning Zhuang*. ECCV 2018. [[pdf](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yang_Shen_Egocentric_Activity_Prediction_ECCV_2018_paper.pdf)]

### 4.8 Network Structure Discovery
1. **Discovering Latent Network Structure in Point Process Data**. *Scott Linderman and Ryan Adams*. ICML 2014. [[pdf](http://proceedings.mlr.press/v32/linderman14.pdf)] [[code](https://github.com/slinderman/pyhawkes)]
### 4.9 Science
1. **Weather Knows What Will Occur: Urban Public Nuisance Events Prediction and Control with Meteorological Assistance**. *Yi Xie, Tianyu Qiu, Yun Xiong, Xiuqi Huang, Xiaofeng Gao, Chao Chen, Qiang Wang, Haihong Li*. KDD 2024. [[acm page](https://dl.acm.org/doi/abs/10.1145/3637528.3671639)] [[video](https://www.youtube.com/watch?v=ozxcqD5cS3k)]
1. **Point process models for sequence detection in high-dimensional neural spike trains**. *Alex Williams, Anthony Degleris, Yixin Wang, Scott Linderman*. NeurIPS 2021. [[pdf](https://arxiv.org/pdf/2010.04875.pdf)] [[code](https://github.com/lindermanlab/PPSeq.jl)]
1. **Point Process Latent Variable Models of Larval Zebrafish Behavior**. *Anuj Sharma, Robert Johnson, Florian Engert, Scott Linderman*. NeurIPS 2018. [[pdf](https://papers.nips.cc/paper/2018/file/e02af5824e1eb6ad58d6bc03ac9e827f-Paper.pdf)]
1. **Mixture of Mutually Exciting Processes for Viral Diffusion**. *Shuang-Hong Yang, Hongyuan Zha*. ICML 2013. [[pdf](http://proceedings.mlr.press/v28/yang13a.pdf)]
### 4.10 Reinforcement Learning
1. **Bellman Meets Hawkes: Model-Based Reinforcement Learning via Temporal Point Processes**. *Chao Qu, Xiaoyu Tan, Siqiao Xue, Xiaoming Shi, James Zhang, Hongyuan Mei*. AAAI 2023. [[pdf](https://arxiv.org/pdf/2201.12569.pdf)] [[code](https://github.com/Event-Driven-rl/Event-Driven-RL)]


## 5. Benchmark
1. **EasyTPP: Towards Open Benchmarking the Temporal Point Processes**. *Siqiao Xue, Xiaoming Shi, Zhixuan Chu, Yan Wang, Fan Zhou, Hongyan Hao, Caigao Jiang, Chen Pan, Yi Xu, James Y. Zhang, Qingsong Wen, Jun Zhou, Hongyuan Mei*. Arxiv 2023. [[pdf](https://arxiv.org/pdf/2307.08097.pdf)] [[code](https://github.com/ant-research/easytemporalpointprocess)]

## 6. Research Opportunities
Note: Papers listed in this section are loosely related to TPP (e.g., may not contain continuous-time modeling), but we find them insightful and open up new research opportunities. 

1. **Predictive Querying for Autoregressive Neural Sequence Models**. *Alex Boyd, Sam Showalter, Stephan Mandt, Padhraic Smyth*. NeurIPS 2022. [[pdf](https://arxiv.org/pdf/2210.06464.pdf)] [[code](https://github.com/ajboyd2/prob_seq_queries)] <details><summary>Why this is relevant? See here for compilers' comments (<i>click to expand</i>)</summary>
(Comments: This is for general autoregressive models and the authors does not do experiments on TPP, but TPP practioners may find it interesting to think about how we can deal with queries beyond one-step-ahead, like `How likely is event A to occur before event B?` and `How likely is event C to occur (once of more) within the next K steps of the sequence?`. An example application in TPP is published in AISTATS 2023, which is listed in Section ``Querying'' of this paper list. )
</details>


<!-- 1. **Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks**. *Srijan Kumar, Xikun Zhang, and Jure Leskovec.* KDD 2019. [[pdf](https://arxiv.org/pdf/1908.01207.pdf)] [[code & data](http://snap.stanford.edu/jodie/)] -->















## Acknowledgements
We thank all the contributors to this list. And more contributions are very welcome.

<a href="https://github.com/yangalan123/TemporalPointProcessPapers/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=yangalan123/TemporalPointProcessPapers" />
</a>

