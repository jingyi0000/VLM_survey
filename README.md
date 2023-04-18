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
- [Vision-Language Model Transfer Learning Methods](#vision-language-model-transfer-learning-methods)
- [Vision-Language Model Knowledge Distillation Methods](#vision-language-model-knowledge-distillation-methods)

## Datasets

### Datasets for VLM Pre-training

#### Public Datasets
1. SBU Caption [[Paper](https://proceedings.neurips.cc/paper_files/paper/2011/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf)] [[Project Page](https://www.cs.rice.edu/~vo9/sbucaptions/)]
2. COCO Caption [[Paper](https://arxiv.org/pdf/1504.00325v2.pdf)] [[Project Page](https://github.com/tylin/coco-caption)]
3. Yahoo Flickr Creative Commons 100 Million (YFCC100M) [[Paper](https://arxiv.org/pdf/1503.01817v2.pdf)] [[Project Page](http://projects.dfki.uni-kl.de/yfcc100m/)]
4. Visual Genome [[Paper](https://arxiv.org/pdf/1602.07332v1.pdf)] [[Project Page](http://visualgenome.org/)]
5. Conceptual Captions (CC3M) [[Paper](https://aclanthology.org/P18-1238.pdf)] [[Project Page](https://ai.google.com/research/ConceptualCaptions/)]
6. Localized Narratives [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500630.pdf)] [[Project Page](https://google.github.io/localized-narratives/)]
7. Conceptual 12M (CC12M) [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Changpinyo_Conceptual_12M_Pushing_Web-Scale_Image-Text_Pre-Training_To_Recognize_Long-Tail_Visual_CVPR_2021_paper.pdf)] [[Project Page](https://github.com/google-research-datasets/conceptual-12m)]
8. Wikipedia-based Image Text (WIT) [[Paper](https://arxiv.org/pdf/2103.01913v2.pdf)] [[Project Page](https://github.com/google-research-datasets/wit)]
9. Red Caps [[Paper](https://arxiv.org/pdf/2111.11431v1.pdf)] [[Project Page](https://redcaps.xyz/)]
10. LAION400M [[Paper](https://arxiv.org/pdf/2111.02114v1.pdf)] [[Project Page](https://laion.ai/blog/laion-400-open-dataset/)]
11. LAION5B [[Paper](https://arxiv.org/pdf/2210.08402.pdf)] [[Project Page](https://laion.ai/blog/laion-5b/)]
12. WuKong [[Paper](https://arxiv.org/pdf/2202.06767.pdf)] [[Project Page](https://wukong-dataset.github.io/wukong-dataset/)]

#### Non-public Datasets
1. CLIP [[Paper](https://arxiv.org/pdf/2103.00020.pdf)]
2. ALIGN [[Paper](https://arxiv.org/pdf/2102.05918.pdf)]
3. FILIP [[Paper](https://arxiv.org/pdf/2111.07783.pdf)]
4. WebLI [[Paper](https://arxiv.org/pdf/2209.06794.pdf)]



### Datasets for VLM Evaluation

#### Image Classification
1. MNIST [[Project Page](http://yann.lecun.com/exdb/mnist/)]
2. Caltech-101 [[Project Page](https://data.caltech.edu/records/mzrjq-6wc02)]
3. PASCAL VOC 2007 Classification [[Project Page](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)]
4. Oxford 102 Folwers [[Project Page](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)]
5. CIFAR-10 [[Project Page](https://www.cs.toronto.edu/~kriz/cifar.html)]
6. CIFAR-100 [[Project Page](https://www.cs.toronto.edu/~kriz/cifar.html)]
7. ImageNet-1k [[Project Page](https://www.image-net.org/)]
8. SUN397 [[Project Page](https://vision.princeton.edu/projects/2010/SUN/)]
9. SVHN [[Project Page](http://ufldl.stanford.edu/housenumbers/)]
10. STL-10 [[Project Page](https://cs.stanford.edu/~acoates/stl10/)]
11. GTSRB [[Project Page](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)]
12. KITTI Distance [[Project Page](https://github.com/harshilpatel312/KITTI-distance-estimation)]
13. IIIT5k [[Project Page](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset)]
14. Oxford-IIIT PETS [[Project Page](https://www.robots.ox.ac.uk/~vgg/data/pets/)]
15. Stanford Cars [[Project Page](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)]
16. FGVC Aircraft [[Project Page](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)]
17. Facial Emotion Recognition 2013 [[Project Page](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data)]
18. Rendered SST2 [[Project Page](https://github.com/openai/CLIP/blob/main/data/rendered-sst2.md)]
19. Describable Textures (DTD) [[Project Page](https://www.robots.ox.ac.uk/~vgg/data/dtd/)]
20. Food-101 [[Project Page](https://www.kaggle.com/datasets/dansbecker/food-101)]
21. Birdsnap [[Project Page](https://thomasberg.org/)]
22. RESISC45 [[Project Page](https://pan.baidu.com/s/1mifR6tU?_at_=1679281159364#list/path=%2F)]
23. CLEVR Counts [[Project Page](https://cs.stanford.edu/people/jcjohns/clevr/)]
24. PatchCamelyon [[Project Page](https://github.com/basveeling/pcam)]
25. EuroSAT [[Project Page](https://github.com/phelber/eurosat)]
26. Hateful Memes [[Project Page](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set/)]
27. Country211 [[Project Page](https://github.com/openai/CLIP/blob/main/data/country211.md)]



#### Image-Text Retrieval
1. Flickr30k [[Project Page](https://shannon.cs.illinois.edu/DenotationGraph/)]
2. COCO Caption [[Project Page](https://github.com/tylin/coco-caption)]

#### Action Recognition
1. UCF101 [[Project Page](https://www.crcv.ucf.edu/data/UCF101.php)]
2. Kinetics700 [[Project Page](https://www.deepmind.com/open-source/kinetics)]
3. RareAct [[Project Page](https://github.com/antoine77340/RareAct)]

#### Object Detection
1. COCO 2014 Detection [[Project Page](https://www.kaggle.com/datasets/jeffaudi/coco-2014-dataset-for-yolov3)]
2. COCO 2017 Detection [[Project Page](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)]
3. LVIS [[Project Page](https://www.lvisdataset.org/)]
4. ODinW [[Project Page](https://eval.ai/web/challenges/challenge-page/1839/overview)]

#### Semantic Segmentation
1. PASCAL VOC 2012 Segmentation [[Project Page](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)]
2. PASCAL Content [[Project Page](https://www.cs.stanford.edu/~roozbeh/pascal-context/)]
2. Cityscapes [[Project Page](https://www.cityscapes-dataset.com/)]
2. ADE20k [[Project Page](https://groups.csail.mit.edu/vision/datasets/ADE20K/)]

## Vision-Language Pre-training Methods

## Vision-Language Model Transfer Learning Methods

## Vision-Language Model Knowledge Distillation Methods
