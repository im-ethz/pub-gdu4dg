# Gated Domain Units for Multi-source Domain Generalization

In our [paper](https://arxiv.org/abs/2206.12444), we postulate that real-world distributions are composed of elementary distributions that remain invariant across different domains. We call this the **invariant elementary distribution (I.E.D.)** assumption. This invariance thus enables knowledge transfer to unseen domains. To exploit this assumption in domain generalization (DG), we developed a modular neural network layer (the DGLayer) that consists of **Gated Domain Units (GDUs)**. Because our layer is trained with backpropagation, it can be easily integrated into existing deep learning frameworks (see our example bewlow).

## Introducing Gated Domain Units for Domain Generalization

Before we introduce our Gated DOmain Units (GDUs), we will briefly present our theoretical idea **invariant elementary distribution (I.E.D.)** assumption from a practical point of view. 

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/73110207/179177894-0528920c-1063-4834-ab3f-852a0ab2d156.png">
</p>

## Install

Currently we do not support installation via pip or conda. To install and use our DGLayer in the meantime, please clone this repository.
