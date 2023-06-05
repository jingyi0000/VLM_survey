## Vision Language Models for Vision Tasks: A Survey
This is the repository of **Vision Language Models for Vision Tasks: a Survey**, a systematic survey of VLM studies in various visual recognition tasks including image classification, object detection, semantic segmentation, etc. For details, please refer to:

**Vision-Language Models for Vision Tasks: A Survey**  
 [[Paper](https://arxiv.org/abs/2304.00685)]
 
[![arXiv](https://img.shields.io/badge/arXiv-2304.00685-b31b1b.svg)](https://arxiv.org/abs/2304.00685) 
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) 
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com) 
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
<!-- [![made-with-Markdown](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg)](http://commonmark.org) -->
<!-- [![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest) -->

## News

## Abstract

Most visual recognition studies rely heavily on crowd-labelled data in deep neural networks (DNNs) training, and they usually train a DNN for each single visual recognition task, leading to a laborious and time-consuming visual recognition paradigm. To address the two challenges, Vision Language Models (VLMs) have been intensively investigated recently, which learns rich vision-language correlation from web-scale image-text pairs that are almost infinitely available on the Internet and enables zero-shot predictions on various visual recognition tasks with a single VLM. This paper provides a systematic review of visual language models for various visual recognition tasks, including: (1) the background that introduces the development of visual recognition paradigms; (2) the foundations of VLM that summarize the widely-adopted network architectures, pre-training objectives, and downstream tasks; (3) the widely adopted datasets in VLM pre-training and evaluations; (4) the review and categorization of existing VLM pre-training methods, VLM transfer learning methods, and VLM knowledge distillation methods; (5) the benchmarking, analysis and discussion of the reviewed methods; (6) several research challenges and potential research directions that could be pursued in the future VLM studies for visual recognition.

## Citation
If you find our work useful in your research, please consider citing:
```
@article{zhang2023vision,
  title={Vision-Language Models for Vision Tasks: A Survey},
  author={Zhang, Jingyi and Huang, Jiaxing and Jin, Sheng and Lu, Shijian},
  journal={arXiv preprint arXiv:2304.00685},
  year={2023}
}
```

## Menu
- [Datasets](#datasets)
  - [Datasets for VLM Pre-training](#datasets-for-vlm-pre-training)
  - [Datasets for VLM Evaluation](#datasets-for-vlm-evaluation)
- [Vision-Language Pre-training Methods](#vision-language-pre-training-methods)
  - [Pre-training with Contrastive Objective](#pre-training-with-contrastive-objective)
  - [Pre-training with Generative Objective](#pre-training-with-generative-objective)
  - [Pre-training with Alignment Objective](#pre-training-with-alignment-objective)
- [Vision-Language Model Transfer Learning Methods](#vision-language-model-transfer-learning-methods)
- [Vision-Language Model Knowledge Distillation Methods](#vision-language-model-knowledge-distillation-methods)

## Datasets

### Datasets for VLM Pre-training


| Dataset                                             |  Year  |     Num of Image-Text Paris     |     Language     | Project |                                  
|-----------------------------------------------------|:------:|:-------------------------------:|:----------------:|:------------:|
|[SBU Caption](https://proceedings.neurips.cc/paper_files/paper/2011/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf)|2011|1M|English|[Project](https://www.cs.rice.edu/~vo9/sbucaptions/)|
|[COCO Caption](https://arxiv.org/pdf/1504.00325v2.pdf)|2016|1.5M|English|[Project](https://github.com/tylin/coco-caption)|
|[Yahoo Flickr Creative Commons 100 Million](https://arxiv.org/pdf/1503.01817v2.pdf)|2016|100M|English|[Project](http://projects.dfki.uni-kl.de/yfcc100m/)|
|[Visual Genome](https://arxiv.org/pdf/1602.07332v1.pdf)|2017|5.4M|English|[Project](http://visualgenome.org/)|
|[Conceptual Captions 3M](https://aclanthology.org/P18-1238.pdf)|2018|3.3M|English|[Project](https://ai.google.com/research/ConceptualCaptions/)|
|[Localized Narratives](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500630.pdf)|2020|0.87M|English|[Project](https://google.github.io/localized-narratives/)|
|[Conceptual 12M](https://openaccess.thecvf.com/content/CVPR2021/papers/Changpinyo_Conceptual_12M_Pushing_Web-Scale_Image-Text_Pre-Training_To_Recognize_Long-Tail_Visual_CVPR_2021_paper.pdf)|2021|12M|English|[Project](https://github.com/google-research-datasets/conceptual-12m)|
|[Wikipedia-based Image Text](https://arxiv.org/pdf/2103.01913v2.pdf)|2021|37.6M|108 Languages|[Project](https://github.com/google-research-datasets/wit)|
|[Red Caps](https://arxiv.org/pdf/2111.11431v1.pdf)|2021|12M|English|[Project](https://redcaps.xyz/)|
|[LAION400M](https://arxiv.org/pdf/2111.02114v1.pdf)|2021|400M|English|[Project](https://laion.ai/blog/laion-400-open-dataset/)|
|[LAION5B](https://arxiv.org/pdf/2210.08402.pdf)|2022|5B|Over 100 Languages|[Project](https://laion.ai/blog/laion-5b/)|
|[WuKong](https://arxiv.org/pdf/2202.06767.pdf)|2022|100M|Chinese|[Project](https://wukong-dataset.github.io/wukong-dataset/)|
||||||
|[CLIP](https://arxiv.org/pdf/2103.00020.pdf)|2021|400M|English|-|
|[ALIGN](https://arxiv.org/pdf/2102.05918.pdf)|2021|1.8B|English|-|
|[FILIP](https://arxiv.org/pdf/2111.07783.pdf)|2021|300M|English|-|
|[WebLI](https://arxiv.org/pdf/2209.06794.pdf)|2022|12B|English|-|



### Datasets for VLM Evaluation

| Dataset                                             |  Year  | Classes | Training | Testing | Evaluation Task |Evaluation Metric| Project|                                  
|-----------------------------------------------------|:------:|:-------:|:--------:|:-------:|:----:|:------:|:-----------:|
|MNIST|1998|10|60,000|10,000|Image Classification|Accuracy|[Project](http://yann.lecun.com/exdb/mnist/)|
|Caltech-101|2004|102|3,060|6,085|Image Classification|Mean Per Class|[Project](https://data.caltech.edu/records/mzrjq-6wc02)|
|PASCAL VOC 2007|2007|20|5,011|4,952|Image Classification|11-point mAP|[Project](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)|
|Oxford 102 Flowers|2008|102|2,040|6,149|Image Classification|Mean Per Class|[Project](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)|
|CIFAR-10|2009|10|50,000|10,000|Image Classification|Accuracy|[Project](https://www.cs.toronto.edu/~kriz/cifar.html)|
|CIFAR-100|2009|100|50,000|10,000|Image Classification|Accuracy|[Project](https://www.cs.toronto.edu/~kriz/cifar.html)|
|ImageNet-1k|2009|1000|1,281,167|50,000|Image Classification|Accuracy|[Project](https://www.image-net.org/)|
|SUN397|2010|397|19,850|19,850|Image Classification|Accuracy|[Project](https://vision.princeton.edu/projects/2010/SUN/)|
|SVHN|2011|10|73,257|26,032|Image Classification|Accuracy|[Project](http://ufldl.stanford.edu/housenumbers/)|
|STL-10|2011|10|1,000|8,000|Image Classification|Accuracy|[Project](https://cs.stanford.edu/~acoates/stl10/)|
|GTSRB|2011|43|26,640|12,630|Image Classification|Accuracy|[Project](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)|
|KITTI Distance|2012|4|6,770|711|Image Classification|Accuracy|[Project](https://github.com/harshilpatel312/KITTI-distance-estimation)|
|IIIT5k|2012|36|2,000|3,000|Image Classification|Accuracy|[Project](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset)|
|Oxford-IIIT PETS|2012|37|3,680|3,669|Image Classification|Mean Per Class|[Project](https://www.robots.ox.ac.uk/~vgg/data/pets/)|
|Stanford Cars|2013|196|8,144|8,041|Image Classification|Accuracy|[Project](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)|
|FGVC Aircraft|2013|100|6,667|3,333|Image Classification|Mean Per Class|[Project](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)|
|Facial Emotion|2013|8|32,140|3,574|Image Classification|Accuracy|[Project](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data)|
|Rendered SST2|2013|2|7,792|1,821|Image Classification|Accuracy|[Project](https://github.com/openai/CLIP/blob/main/data/rendered-sst2.md)|
|Describable Textures|2014|47|3,760|1,880|Image Classification|Accuracy|[Project](https://www.robots.ox.ac.uk/~vgg/data/dtd/)|
|Food-101|2014|101|75,750|25,250|Image Classification|Accuracy|[Project](https://www.kaggle.com/datasets/dansbecker/food-101)|
|Birdsnap|2014|500|42,283|2,149|Image Classification|Accuracy|[Project](https://thomasberg.org/)|
|RESISC45|2017|45|3,150|25,200|Image Classification|Accuracy|[Project](https://pan.baidu.com/s/1mifR6tU?_at_=1679281159364#list/path=%2F)|
|CLEVR Counts|2017|8|2,000|500|Image Classification|Accuracy|[Project](https://cs.stanford.edu/people/jcjohns/clevr/)|
|PatchCamelyon|2018|2|294,912|32,768|Image Classification|Accuracy|[Project](https://github.com/basveeling/pcam)|
|EuroSAT|2019|10|10,000|5,000|Image Classification|Accuracy|[Project](https://github.com/phelber/eurosat)|
|Hateful Memes|2020|2|8,500|500|Image Classification|ROC AUC|[Project](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set/)|
|Country211|2021|211|43,200|21,100|Image Classification|Accuracy|[Project](https://github.com/openai/CLIP/blob/main/data/country211.md)|
||||||||
|Flickr30k|2014|-|31,783|-|Image-Text Retrieval|Recall|[Project](https://shannon.cs.illinois.edu/DenotationGraph/)
|COCO Caption|2015|-|82,783|5,000|Image-Text Retrieval|Recall|[Project](https://github.com/tylin/coco-caption)
||||||||
|UCF101|2012|101|9,537|1,794|Action Recognition|Accuracy|[Project](https://www.crcv.ucf.edu/data/UCF101.php)|
|Kinetics700|2019|700|494,801|31,669|Action Recognition|Mean (top1, top5)|[Project](https://www.deepmind.com/open-source/kinetics)|
|RareAct|2020|122|7,607|-|Action Recognition|mWAP, mSAP|[Project](https://github.com/antoine77340/RareAct)|
||||||||
|COCO 2014 Detection|2014|80|83,000|41,000|Object Detection|Box mAP|[Project](https://www.kaggle.com/datasets/jeffaudi/coco-2014-dataset-for-yolov3)|
|COCO 2017 Detection|2017|80|118,000|5,000|Object Detection|Box mAP|[Project](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)|
|LVIS|2019|1203|118,000|5,000|Object Detection|Box mAP|[Project](https://www.lvisdataset.org/)|
|ODinW|2022|314|132,413|20,070|Object Detection|Box mAP|[Project](https://eval.ai/web/challenges/challenge-page/1839/overview)|
||||||||
|PASCAL VOC 2012|2012|20|1,464|1,449|Semantic Segmentation|mIoU|[Project](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)|
|PASCAL Content|2014|459|4,998|5,105|Semantic Segmentation|mIoU|[Project](https://www.cs.stanford.edu/~roozbeh/pascal-context/)|
|Cityscapes|2016|19|2,975|500|Semantic Segmentation|mIoU|[Project](https://www.cityscapes-dataset.com/)|
|ADE20k|2017|150|25,574|2,000|Semantic Segmentation|mIoU|[Project](https://groups.csail.mit.edu/vision/datasets/ADE20K/)|

## Vision-Language Pre-training Methods

### Pre-training with Contrastive Objective
1. Learning Transferable Visual Models From Natural Language Supervision (CLIP) [[Paper](https://arxiv.org/pdf/2103.00020.pdf)][[Code](https://github.com/openai/CLIP)]
2. Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision (ALIGN) [[Paper](https://arxiv.org/pdf/2102.05918.pdf)]
3. Data Efficient Language-Supervised Zero-Shot Recognition with Optimal Transport Distillation (OTTER) [[Paper](https://arxiv.org/abs/2112.09445)][[Code](https://github.com/facebookresearch/OTTER)]
4. Supervision Exists Everywhere: A Data Efficient Contrastive Language-Image Pre-training Paradigm (DeCLIP) [[Paper](https://arxiv.org/abs/2110.05208)][[Code](https://github.com/Sense-GVT/DeCLIP)]
5. Contrastive Vision-Language Pre-training with Limited Resources (ZeroVL) [[Paper](https://arxiv.org/abs/2112.09331)][[Code](https://github.com/zerovl/ZeroVL)]
6. FILIP: Fine-grained Interactive Language-Image Pre-Training [[Paper](https://arxiv.org/abs/2111.07783)]
7. Unified Contrastive Learning in Image-Text-Label Space (UniCL) [[Paper](https://arxiv.org/abs/2204.03610)][[Code](https://github.com/microsoft/UniCL)]
8. Florence: A New Foundation Model for Computer Vision [[Paper](https://arxiv.org/abs/2111.11432)]
9. SLIP: Self-supervision meets Language-Image Pre-training [[Paper](https://arxiv.org/abs/2112.12750)][[Code](https://github.com/facebookresearch/SLIP)]
10. PyramidCLIP: Hierarchical Feature Alignment for Vision-language Model Pretraining [[Paper](https://arxiv.org/abs/2204.14095)]
11. Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese [[Paper](https://arxiv.org/abs/2211.01335)][[Code](https://github.com/OFA-Sys/Chinese-CLIP)]
12. LiT: Zero-Shot Transfer with Locked-image text Tuning [[Paper](https://arxiv.org/abs/2111.07991)][[Code](https://google-research.github.io/vision_transformer/lit/)]
13. AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities [[Paper](https://arxiv.org/abs/2211.06679)][[Code](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP)]
14. FLAVA: A Foundational Language And Vision Alignment Model [[Paper](https://arxiv.org/abs/2112.04482)][[Code](https://github.com/facebookresearch/multimodal/tree/main/examples/flava)]
15. Large-scale Bilingual Language-Image Contrastive Learning (KELIP)[[Paper](https://arxiv.org/abs/2203.14463)][[Code](https://github.com/navervision/KELIP)]
16. CoCa: Contrastive Captioners are Image-Text Foundation Models [[Paper](https://arxiv.org/abs/2205.01917)][[Code](https://github.com/lucidrains/CoCa-pytorch)]
17. Non-Contrastive Learning Meets Language-Image Pre-Training (nCLIP) [[Paper](https://arxiv.org/abs/2210.09304)]
18. K-LITE: Learning Transferable Visual Models with External Knowledge [[Paper](https://arxiv.org/abs/2204.09222)][[Code](https://github.com/microsoft/klite)]
19. NLIP: Noise-robust Language-Image Pre-training [[Paper](https://arxiv.org/abs/2212.07086)]
20. UniCLIP: Unified Framework for Contrastive Language-Image Pre-training [[Paper](https://arxiv.org/abs/2209.13430)]

### Pre-training with Generative Objective

### Pre-training with Alignment Objective

## Vision-Language Model Transfer Learning Methods

## Vision-Language Model Knowledge Distillation Methods
