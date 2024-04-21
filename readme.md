# Attention Is All You Need - PyTorch Transformer Implementation

In this repo I aspire to re-implement using PyTorch the original [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper, laid out a model for translating between german and english. It introduced a new form of the attention mechanism and the creation of the Transformer model. Which kicked of a whole new era of NLP models and beyond.


# Setup Guide
To use this repo follow these steps:

1. Install [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
2. Clone this repo `git clone git@github.com:lhmartin/transformer.git`
3. Entry into the directory with `cd transformer`
4. Install the dependencies with `poetry install`
5. You can then train using by first entering into your enviroment with `poetry shell` or by running `poetry run` before the call to the training script
6. To train use: `python train_script.py --config <path_to_config>`

## What is a Transformer?

A transformer is a model architecture that uses the attention mechanism to draw global dependencies between input and output. One major benefit of this architecture, over recurrent networks such as an RNN, is that the Transformer can easily paralized, and it doesn't require sequence calls during training. This decreases the training time need to reach similar performance to RNNs

<img src="imgs/Figure 1 - The Transformer.png" alt="The Transformer" width="450"/>

### Architecture

A transformer consists of two main componenets. The encoder and the decoder. The encoder takes in the full input sequence, in the form of input ids.



## The Attention Mechanism
<img src="imgs/Figure 2 - Scaled Dot Attention and Multi Head.png" alt="Scaled Dot Attention and Multi Head" width="650"/>



### 

## Training Task and Data

## Components

### Sequential Embeddings

<img src="imgs/pos encoding.png" alt="Positonal Encoding" width="700"/>

### Custom Learning Rate
