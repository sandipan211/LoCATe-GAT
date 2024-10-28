# LoCATe-GAT

## 👓 At a glance
This repository contains the official PyTorch implementation of our paper : LoCATe-GAT: Modeling Multi-Scale Local Context and Action Relationships for Zero-Shot Action Recognition, a work done by Sandipan Sarma, Divyam Singal, and Arijit Sur at [Indian Institute of Technology Guwahati](https://www.iitg.ac.in/cse/). The work has been recently published in the [IEEE Transactions on Emerging Topics in Computational Intelligence](https://ieeexplore.ieee.org/xpl/aboutJournal.jsp?punumber=7433297).

### 😍 Motivation

The increasing number of actions in the real world makes it difficult for traditional deep learning models to recognize unseen actions. Recently, this data scarcity gap has been bridged by pretrained vision-language models like CLIP for efficient **zero-shot action recognition**. We have two important observations:

- **Local spatial context**: Existing best methods are transformer-based, which capture global context via self-attention, but miss out on local details.
- **Duality**: Objects and action environments play a dual role of promoting distinguishability and functional similarity, assisting action recognition of both seen and unseen classes.

### 💡 Approach
We propose a two-stage framework (as shown in the figure below) that contains a novel transformer called LoCATe and a graph attention network (GAT):

- **Local Context-Aggregating Temporal transformer (LoCATe)**: Captures multi-scale local context using dilated convolutional layers during temporal modeling
- **GAT**: Models semantic relationships between action classes and achieves a strong synergy with the video embeddings produced by LoCATe

### ✅ Outcomes
- State-of-the-art/comparable results on four benchmark datasets
- Best results on the recently proposed TruZe evaluation protocol
- Uses 25x fewer parameters than existing methods
- Mitigates the polysemy problem better than previous methods

## 📁 Preparing the datasets

We have evaluated our method on four benchmarks: 
- [UCF-101](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar) and [HMDB-51](serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar) can be directly downloaded from the web. Zero-shot splits for both these datasets are provided within ```datasets/Label.mat``` and ```datasets/Split.mat```.
- For [ActivityNet](http://activity-net.org/download.html), fill [this](https://docs.google.com/forms/d/e/1FAIpQLSdxhNVeeSCwB2USAfeNWCaI9saVT6i2hpiiizVYfa3MsTyamg/viewform) form to request for the dataset. Zero-shot splits are provided in the folder ```datasets/ActivityNet_v_1_3```.
- For Kinetics, we followed [ER-ZSAR (ICCV 2021)](https://github.com/DeLightCMU/ElaborativeRehearsal) for obtaining the zero-shot splits. Training is done on the entire Kinetics-400 dataset, and testing is done on subsets of Kinetics-600. Zero-shot splits are provided in the folder ```datasets/kinetics-400``` and ```datasets/kinetics-600```.
      - Kinetics-400 has been downloaded following [this](https://github.com/youngwanLEE/VoV3D/blob/main/DATA.md#kinetics-400) repo.
      - For Kinetics-600, we downloaded the videos of the validate and test sets only. For downloading videos, the ```youtube-dl``` package doesn't work seamlessly anymore, so we switched to using ```yt-dlp```, which can be installed following the commands [here](https://www.rapidseedbox.com/blog/yt-dlp-complete-guide). Then, use the following commands for downloading the videos:

  ```bash
  cd datasets/kinetics-600
  python download.py {dataset_split}.csv <data_dir>
  ```
The final datasets directory should have the following structure:

```
datasets
│   Label.mat
│   Split.mat    
│
└───ActivityNet_v_1_3
│   │   activity_net.v1-3.min.json
│   │   anet_classwise_videos.npy
│   |   anet_splits.npy
│   └───Anet_videos_15fps_short256
│       │   v___c8enCfzqw.mp4
│       │   v___dXUJsj3yo.mp4
│       |   ...
│
└───hmdb
│   └───hmdb51_org
│       └───brush_hair
│       └───cartwheel
│       └───...
│   
└───kinetics-400
│   └───train_256
│   │   └───abseiling
│   │   └───air_drumming
│   │   └───...
│   │
│   └───val_256
│   │   └───abseiling
│   │   └───air_drumming
│   │   └───...
│   └───zsar_kinetics_400
│   
└───kinetics-600
│   │   download.py
│   │   test.csv  
│   │   validate.csv
│   │ 
│   └───test
│   │   └───abseiling
│   │   └───acting in play
│   │   └───...
│   │
│   └───validate
│   │   └───abseiling
│   │   └───acting in play
│   │   └───...
│   └───zsar_kinetics_600
│   
└───ucf
│   └───UCF101
│       └───ApplyEyeMakeup
│       └───ApplyLipstick
│       └───...
```
## 🚄 Training
