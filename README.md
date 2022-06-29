This is a fork of min(DALL·E), modified so the model only gets loaded once and is compatible with cuda.

# min(DALL·E)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kuprel/min-dalle/blob/main/min_dalle.ipynb)

This is a minimal implementation of [DALL·E Mini](https://github.com/borisdayma/dalle-mini).  It has been stripped to the bare essentials necessary for doing inference, and converted to PyTorch.  The only third party dependencies are `numpy` and `torch` for the torch model and `flax` for the flax model.

### Setup

Run `sh setup.sh` to install dependencies and download pretrained models.  In the bash script, Git LFS is used to download the VQGan detokenizer from Hugging Face and the Weight & Biases python package is used to download the DALL·E Mini and DALL·E Mega transformer models. These models can also be downloaded manually: 
[VQGan](https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384), 
[DALL·E Mini](https://wandb.ai/dalle-mini/dalle-mini/artifacts/DalleBart_model/mini-1/v0/files), 
[DALL·E Mega](https://wandb.ai/dalle-mini/dalle-mini/artifacts/DalleBart_model/mega-1-fp16/v14/files)

### Usage

Use the command line

```
python main.py
```

and choose your prompt when it prompts you to


if you have giant gpu you can do
```
python main.py --mega
```