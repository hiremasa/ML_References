# ML_References
ML/DLのリファレンス集<br>
自分が読んで使ったものを追加していく<br>
フォーマットは[ここ](https://github.com/phalanx-hk/kaggle_cv_pipeline/blob/master/kaggle_tips.md#pretrained_model_library) を参考
- [よく参照する論文等](#journal)
  - [pretrained model library](#pretrained_model_library)
  - [data augmentation library](#data_augmentation_library)
  - [classification models](#image_classification_models)
  - [optimizer](#optimizer)
  - [reguralization](#reguralization)
  - [batch normalization](#batch_norm)
  - [hyperparameter tuning](#hyperparameter_tuning)
  - [visualizatioo](#visualization)
  - [imbalanced data](#imbalanced_data)
  - [semi-supervised learning](#semi_supervised_learning)
  - [unsupervised learning](#unsupervised_learning)
  - [multi task learning](#multi_task_learning)
  - [fine grained visual classification](#fine_grained_visual_classification)
  - [knowledge distillation](#knowledge_distillation)
  - [domain adaptation](#domain_adaptation)
  - [metric learning / face recognition / re-id](metric_learning)　
  - [survey](#survey)
  - [competition solution](#solution)
  - [tracking ML experiment](#ml_experiment)

- [お役立ちサイト集](#ref_blogs)
  - [Qiita](#qiita)
  - [Blog](#blog)
  - [Kaggle](#kaggle)
---
<a name="journal"></a>
# ML/DLのリファレンス集
- [] 論文概要とソースコード

<a name="pretrained_model_library"></a>

# pretrained models
<!-- - [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch) -->
- [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
<!-- - [EfficientNet-Pytorch](https://github.com/lukemelas/EfficientNet-PyTorch) -->

<a name="data_augmentation_library"></a>

# data augmentation library
[albumentations](https://github.com/albumentations-team/albumentations)
<!-- - [albumentations](https://github.com/albumentations-team/albumentations)
- [dali](https://github.com/NVIDIA/DALI)
- [kornia](https://github.com/kornia/kornia)
- [rising](https://github.com/PhoenixDL/rising)
- [solt](https://github.com/MIPT-Oulu/solt) -->

<a name="image_classification_models"></a>

# classification models (読んだら更新していく)
## must
<!-- - [Inceptionv3](https://arxiv.org/abs/1512.00567) (CVPR'16)
- [ResNet](https://arxiv.org/abs/1512.03385) (CVPR'16)
- [DenseNet](https://arxiv.org/abs/1608.06993) (CVPR'17)
- [ResNeXt](https://arxiv.org/abs/1611.05431) (CVPR'17)
- [SENet](https://arxiv.org/abs/1709.01507) (CVPR'18)
- [coord conv](https://arxiv.org/abs/1807.03247) (NeurIPS'18)
- [HRNet](https://arxiv.org/abs/1908.07919) (CVPR'19)  -->
- [EfficientNet](https://arxiv.org/abs/1905.11946) (ICML'19)
<!-- - [ECA-Net](https://arxiv.org/abs/1910.03151) (CVPR'20)
- [ResNeSt](https://arxiv.org/abs/2004.08955) (arxiv)
- [octave conv](https://arxiv.org/abs/1904.05049) (ICCV'19) -->

## try
<!-- - [HarDNet](https://arxiv.org/abs/1909.00948) (ICCV'19)
- [RegNet](https://arxiv.org/abs/2003.13678) (CVPR'20)
- [CSPNet](https://arxiv.org/abs/1911.11929) (CVPRW'20)
- [Spatially Attentive Output Layer](https://arxiv.org/abs/2004.07570) (CVPR'20)
- [Improved ResNet](https://arxiv.org/abs/2004.04989) (arxiv)
- [SlimConv](https://arxiv.org/abs/2003.07469) (arxiv)
- [Visual Transformers](https://arxiv.org/abs/2006.03677) (arxiv)
- [URIE](https://arxiv.org/abs/2007.08979) (ECCV'20) -->

<a name="optimizer"></a>

# optimizer
- [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412) (ICLR 2021)

<a name="reguralization"></a>

# reguralization
## label smoothing
[When Does Label thing Help?](https://arxiv.org/abs/1906.02629) (NeurIPS'19)

## dropout
<!-- - [dropout](https://jmlr.org/papers/v15/srivastava14a.html) (JMLR'14)
- [dropblock](https://arxiv.org/abs/1810.12890) (NeurIPS'18) -->

## data augmentatoin
### model robustness under data shift
<!-- - [Generalisation in humans and deep neural networks](https://arxiv.org/abs/1808.08750) (NeurIPS'18)
- [IMAGENET-TRAINED CNNS ARE BIASED TOWARDS TEXTURE](https://arxiv.org/abs/1811.12231) (ICLR'19)
- [BENCHMARKING NEURAL NETWORK ROBUSTNESS TO COMMON CORRUPTIONS AND PERTURBATIONS](https://arxiv.org/abs/1903.12261) (ICLR'19)
- [Why do deep convolutional networks generalize so poorly to small image transformations?](https://arxiv.org/abs/1805.12177) (JMLR'19) -->

### augmentation method
- [Bag of Tricks](https://arxiv.org/pdf/1812.01187.pdf)
<!-- - [mixup](https://arxiv.org/abs/1710.09412) (ICLR'18)
- [CutMix](https://arxiv.org/abs/1905.04899) (ICCV'19)
- [Manifold Mixup](https://arxiv.org/abs/1806.05236) (ICML'19)
- [Fast AutoAugment](https://arxiv.org/abs/1905.00397) (NeurIPS'19)
- [Implicit Semantic Data Augmentation](https://arxiv.org/abs/1909.12220) (NeurIPS'19)
- [Population Based Augmentation](https://arxiv.org/abs/1905.05393) (ICML'19)
- [RandAugment](https://arxiv.org/abs/1909.13719) (CVPRW'20)
- [SmoothMix](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w45/Lee_SmoothMix_A_Simple_Yet_Effective_Data_Augmentation_to_Train_Robust_CVPRW_2020_paper.pdf) (CVPRW'20)
- [Adversarial AutoAugment](https://arxiv.org/abs/1912.11188) (ICLR'20)
- [AUGMIX](https://arxiv.org/abs/1912.02781) (ICLR'20)
- [Puzzle Mix](https://proceedings.icml.cc/static/paper_files/icml/2020/6618-Paper.pdf) (ICML'20)
- [Attribute Mix](https://arxiv.org/abs/2004.02684) (arxiv)
- [Attentive CutMix](https://arxiv.org/abs/2003.13048) (arxiv)
- [FMix](https://arxiv.org/abs/2002.12047) (arxiv)
- [Momentum Exchenge](https://arxiv.org/abs/2002.11102) (arxiv)
- [patch gaussian](https://arxiv.org/abs/1906.02611) (arxiv) -->

<a name="batch_norm"></a>

# batch normalization
<!-- - [Instance Normalization](https://arxiv.org/abs/1701.02096) (CVPR'17)
- [Group Normalization](https://arxiv.org/abs/1803.08494) (ECCV'18)
- [Filter Response Normalization](https://arxiv.org/abs/1911.09737) (CVPR'20)
- [Evolving Normalization](https://arxiv.org/abs/2004.02967) (arxiv) -->

<a name="hyperparameter_tuning"></a>

# hyperparameter tuning
## library
<!-- - [optuna](https://optuna.org/) -->
## journal
<!-- - [RETHINKING THE HYPERPARAMETERS
FOR FINE-TUNING](https://arxiv.org/abs/2002.11770) (ICLR'20)
- [HyperSTAR](https://arxiv.org/abs/2005.10524) (CVPR'20) -->

<a name="imbalanced_data"></a>

# imbalanced data
<!-- - [pc-softmax](https://arxiv.org/abs/1911.10688) (arxiv)
- [focal loss](https://arxiv.org/abs/1708.02002) (ICCV'17)
- [reduced focal loss](https://arxiv.org/abs/1903.01347) (arxiv)
- [Class-Balanced Loss](https://arxiv.org/abs/1901.05555) (CVPR'19)
- [Bilateral-Branch Network](https://arxiv.org/abs/1912.02413) (CVPR'20)
- [Rebalanced mixup](https://arxiv.org/abs/2007.03943) (arxiv)
- [M2m](https://arxiv.org/abs/2004.00431) (CVPR'20) -->

  - [visualizatioo](#visualization)
<a name="visualization"></a>

# visualization
- [Grad-Cam](https://arxiv.org/pdf/1610.02391.pdf)

<a name="semi_supervised_learning"></a>

# semi-supervised learning
<!-- - [Pseudo-label](http://deeplearning.net/wp-content/uploads/2013/03/pseudo_label_final.pdf)
- [Noisy Student](https://arxiv.org/abs/1911.04252) (CVPR'20)
- [Mean Teacher](https://arxiv.org/abs/1703.01780) (NIPS'17)
- [MixMatch](https://arxiv.org/abs/1905.02249) (NIPS'19)
- [FixMatch](https://arxiv.org/abs/2001.07685) (arxiv)
- [FeatMatch](https://arxiv.org/abs/2007.08505) (ECCV'20) -->

<a name="unsupervised_learning"></a>

# unsupervised learning
<!-- - [SCAN](https://arxiv.org/abs/2005.12320) (ECCV'20) -->

<a name="multi_task_learning"></a>

# multi task learning
<!-- - [Dynamic Weight Average](https://arxiv.org/abs/1803.10704) (CVPR'19)
- [NDDR-CNN](https://arxiv.org/abs/1801.08297) (CVPR'19)
- [ML-GCN](https://arxiv.org/abs/1904.03582) (CVPR'19) -->

<a name="fine_grained_visual_classification"></a>

# fine grained visual classification
<!-- - [Facing the Hard Problems in FGVC](https://arxiv.org/abs/2006.13190) (arxiv)
- [DFL-CNN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Learning_a_Discriminative_CVPR_2018_paper.pdf) (CVPR'18)
- [Destruction and Construction Learning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Destruction_and_Construction_Learning_for_Fine-Grained_Image_Recognition_CVPR_2019_paper.pdf) (CVPR'19)
- [Look-Into-Object](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Look-Into-Object_Self-Supervised_Structure_Modeling_for_Object_Recognition_CVPR_2020_paper.pdf) (CVPR'20) -->

<a name="knowledge_distillation"></a>

# knowledge distillation
<!-- - [Learning What and Where to Transfer](https://arxiv.org/abs/1905.05901) (ICML'19)
- [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068) (CVPR'19)
- [Attention Transfer](https://arxiv.org/abs/1612.03928) (ICLR'17)
- [Noisy Student](https://arxiv.org/abs/1911.04252) (CVPR'20)
- [Mean Teacher](https://arxiv.org/abs/1703.01780) (NIPS'17) -->

<a name="domain_adaptation"></a>

# domain adaptation
<!-- - [domain adversarial neural network](https://arxiv.org/abs/1505.07818) (JMLR'16)
- [REVISITING BATCH NORMALIZATION FOR PRACTICAL DOMAIN ADAPTATION](https://arxiv.org/abs/1603.04779) (ICLR'17)
- [MUNIT](https://arxiv.org/abs/1804.04732) (ECCV'18)
- [Style Normalization and Restitution](https://arxiv.org/abs/2005.11037) (CVPR'20) -->

<a name="metric_learning"></a>

# metric learning / face recognition / re-id
<!-- ## library
- [torch reid](https://github.com/KaiyangZhou/deep-person-reid)
- [insightface](https://github.com/deepinsight/insightface)
- [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline)
- [cnnimageretrieval-pytorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch)

## journal
- [center loss](https://ydwen.github.io/papers/WenECCV16.pdf) (ECCV'16)
- [Fine-tuning CNN Image Retrieval with No Human Annotation](https://arxiv.org/abs/1711.02512) (TPAMI'18)
- [arcface](https://arxiv.org/abs/1801.07698) (CVPR'19)
- [AdaCos](https://arxiv.org/abs/1905.00292) (CVPR'19)
- [MS-Loss](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf) (CVPR'19)
- [Bag of Tricks and A Strong Baseline for Deep Person Re-identification](https://arxiv.org/abs/1903.07071) (CVPRW'19)
- [AP-Loss](https://arxiv.org/abs/1906.07589) (ICCV'19)
- [SoftTriple Loss](https://arxiv.org/abs/1909.05235) (ICCV'19)
- [Circle Loss](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Circle_Loss_A_Unified_Perspective_of_Pair_Similarity_Optimization_CVPR_2020_paper.pdf) (CVPR'20)
- [Cross-Batch Memory](https://arxiv.org/abs/1912.06798) (CVPR'20)
- [Unifying Deep Local and Global Features for Image Search](https://arxiv.org/abs/2001.05027) (ECCV'20) -->



<a name="survey"></a>

# survey
<!-- - [Noisy Labels](https://arxiv.org/abs/2007.08199)
- [data augmentation](https://link.springer.com/article/10.1186/s40537-019-0197-0)
- [face recognition](https://arxiv.org/abs/1804.06655)
- [metric learning](https://www.mdpi.com/2073-8994/11/9/1066) -->


---
<a name="ref_blogs"></a>
# お役立ちサイト集
- [] 随時追加していく


<a name="qiita"></a>

# Qiita

<a name="blog"></a>

# Blog

<a name="kaggle"></a>

#Kaggle
- [EfficientNet-PyTorch Speed up and Memory usage](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/111292)
