# Annotated MPNet

- [Annotated MPNet](#annotated-mpnet)
  - [About](#about)
  - [Installation](#installation)
  - [Pretraining MPNet](#pretraining-mpnet)
    - [HuggingFace dataset](#huggingface-dataset)
    - [Local directory of text files](#local-directory-of-text-files)
  - [Porting a checkpoint to HuggingFace's format](#porting-a-checkpoint-to-huggingfaces-format)

## About

This repository is based very closely on the wonderful code written by the authors of [MPNet: Masked and Permuted Pre-training for Language Understanding](https://arxiv.org/pdf/2004.09297.pdf) (Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu). MPNet employs a pretraining method that has seen a great deal of success in NLP fine-tuning tasks like information retrieval and learning-to-rank.

While many of the fine-tuned applications of this model exist out in the open, pretraining the model remains somewhat esoteric, as this code lives within the researcher's modified [fairseq codebase](https://github.com/microsoft/MPNet/tree/master) (HuggingFace's MPNet can only be used for fine-tuning, since it doesn't properly encode two-stream attention). This codebase works well, but it is fairly bloated and the code is, perhaps unintentionally, obfuscated by the many subdirectories and imports pointing all through the source code.

With `annotated-mpnet`, we are looking to bring the MPNet pretraining code under one, lightweight, clean roof implemented in raw PyTorch. This eliminates the need to use the entirety of `fairseq`, and allows pretraining to occur on nonstandard training devices (i.e. accelerated hardware beyond a GPU).

Additionally, we have gone through the painstaking effort of carefully annotating and commenting on each portion of the model code so that general understanding of the model is more easily conveyed. It is our hope that using this codebase, others will be able to pretrain MPNet using their own data.

## Installation

Install as editable with `pip install -e .`

```bash
git clone https://github.com/pszemraj/annotated-mpnet.git
cd annotated-mpnet
pip install -e .
```

## Pretraining MPNet

> [!NOTE]
> Pretraining can be completed using a directory of text files (_original method_) or a HuggingFace dataset via streaming.

Pretraining is as simple as calling the pretraining entrypoint, which was just installed in the previous step. You can get a rundown of exactly which arguments are provided by typing `pretrain-mpnet -h`, but example commands are shown below.

### HuggingFace dataset

```bash
pretrain-mpnet --dataset-name HuggingFaceFW/fineweb-edu \
--total-updates 10000 --warmup-updates 1000 \
--batch-size 4 --update-freq 4 --lr 0.0002 \
-save_steps 200 -log_dir checkpoints/experiments
```

### Local directory of text files

```bash
pretrain-mpnet \
--train-dir /path/to/train/dir \
--valid-file /path/to/validation/txtfile \
--test-file /path/to/test/txtfile \
--total-updates 10000 \ # total number of updates to run
--warmup-updates 1000 \ # number of updates to warm up the LR
--batch-size 256 \ # actual batch size in memory
--update-freq 4 \ # simulating 1024 bs via gradient accumulation
--lr 0.0002 # the peak learning rate, reached at warmup-updates and then decayed according to the --power arg
```

## Porting a checkpoint to HuggingFace's format

After pretraining is done, you'll probably want to do something with the model! The best way to do this is to port one of your checkpoints to the HuggingFace MPNet format. This will allow you to load the custom model into HuggingFace and do fine-tuning there as you normally would. Below is what a call would look like to do this conversion:

```bash
convert-to-hf \
--mpnet-checkpoint-path ./checkpoints/best_checkpoint.pt \
--hf-model-folder-path ./my_cool_hf_model/
```
