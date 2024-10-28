# LoCATe-GAT

##  :eyeglasses: At a glance
This repository contains the official PyTorch implementation of our paper : LoCATe-GAT: Modeling Multi-Scale Local Context and Action Relationships for Zero-Shot Action Recognition, a work done by Sandipan Sarma, Divyam Singal, and Arijit Sur at [Indian Institute of Technology Guwahati](https://www.iitg.ac.in/cse/). The work has been recently published in the [IEEE Transactions on Emerging Topics in Computational Intelligence](https://ieeexplore.ieee.org/xpl/aboutJournal.jsp?punumber=7433297).

### Motivations

The increasing number of actions in the real world makes it difficult for traditional deep learning models to recognize unseen actions. Recently, this data scarcity gap has been bridged by pretrained vision-language models like CLIP for efficient **zero-shot action recognition**. We have two important observations:

- **Local spatial context**: Existing best methods are transformer-based, which capture global context via self-attention, but miss out on local details.
- **Duality**: Objects and action environments play a dual role of promoting distinguishability and functional similarity, assisting action recognition of both seen and unseen classes.

### Approach
We propose a two-stage framework (as shown in the figure below) that contains a novel transformer called LoCATe and a graph attention network (GAT):

- **Local Context-Aggregating Temporal transformer (LoCATe)**: Captures multi-scale local context using dilated convolutional layers during temporal modeling
- **GAT**: Models semantic relationships between action classes and achieves a strong synergy with the video embeddings produced by LoCATe

### Outcomes
- State-of-the-art/comparable results on four benchmark datasets
- Best results on the recently proposed TruZe evaluation protocol
- Uses 25x fewer parameters than existing methods
- Mitigates the polysemy problem better than previous methods

## 📁 Preparing the datasets

We have evaluated our method on four benchmarks: 
- [UCF-101](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar) and [HMDB-51](serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar) can be directly downloaded from the web.
- Download the json file for [ActivityNet-v1.3](http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json) and then use this script for downloading the videos.

  I followed an example in another repository and wrapped the directory structure within a pair of triple backticks (```):

```
datasets
│   README.md
│   file001.txt    
│
└───folder1
│   │   file011.txt
│   │   file012.txt
│   │
│   └───subfolder1
│       │   file111.txt
│       │   file112.txt
│       │   ...
│   
└───folder2
    │   file021.txt
    │   file022.txt
```
