## Vision Language Models for Vision Tasks: A Survey
This is the repository of **Vision Language Models for Vision Tasks: a Survey**, a systematic survey of VLM studies in various visual recognition tasks including image classification, object detection, semantic segmentation, etc. For details, please refer to:

**Vision-Language Models for Vision Tasks: A Survey**  
 [[Paper](https://arxiv.org/abs/2304.00685)]

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

## Vision-Language Pre-training Methods

## Vision-Language Model Transfer Learning Methods

## Vision-Language Model Knowledge Distillation Methods
