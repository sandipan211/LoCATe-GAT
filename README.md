# LoCATe-GAT

##  :eyeglasses: At a glance
This repository contains the official PyTorch implementation of our paper : LoCATe-GAT: Modeling Multi-Scale Local Context and Action Relationships for Zero-Shot Action Recognition, a work done by Sandipan Sarma, Divyam Singal, and Arijit Sur at [Indian Institute of Technology Guwahati](https://www.iitg.ac.in/cse/). The work has been recently published in the [IEEE Transactions on Emerging Topics in Computational Intelligence](https://ieeexplore.ieee.org/xpl/aboutJournal.jsp?punumber=7433297).

### Motivations

The increasing number of actions in the real world makes it difficult for traditional deep learning models to recognize unseen actions. Recently, this data scarcity gap has been bridged by pretrained vision-language models like CLIP for efficient **zero-shot action recognition**. We have two important observations:

- **Local spatial context**: Existing best methods are transformer-based which capture global context via self-attention, but miss out on local details.
- **Duality**: Objects and action environments play a dual role of promoting distinguishability and functional similarity, assisting action recognition of both seen and unseen classes.

- We propose a **generative approach and introduced triplet loss** during feature generation to account for inter-class dissimilarity.

- Moreover, we show that **maintaining cyclic consistency** between the generated visual features and their class semantics is helpful for improving the quality of the generated features.

- **Addressed problems** such as high false positive rate and misclassification of localized objects by resolving semantic confusion, and **comprehensively beat the state-of-the-art methods**.
