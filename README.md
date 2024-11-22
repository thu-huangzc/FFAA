# <img src="https://cdn-icons-png.flaticon.com/512/1925/1925270.png" alt="My Icon" style="width: 1em; height: 1em;"> FFAA: Face Forgery Analysis Assistant

**FFAA: Multimodal Large Language Model based Explainable Open-World Face Forgery Analysis Assistant** [[Paper](https://arxiv.org/abs/2408.10072)][[Project Page](https://ffaa-vl.github.io/)]<br>[<u>Zhengchao Huang</u>](https://github.com/thu-huangzc), [<u>Bin Xia</u>](https://github.com/Zj-BinXia), [<u>Zicheng Lin</u>](https://github.com/chenzhiling9954), [<u>Zhun Mou</u>](https://github.com/rexviv), [<u>Wenming Yang</u>](https://scholar.google.com/citations?hl=zh-CN&user=vsE4nKcAAAAJ), [<u>Jiaya Jia</u>](https://scholar.google.com/citations?user=XPAkzTEAAAAJ&hl=en&oi=ao)


## Release

* [2024/09/26] The code and the model weights of FFAA have been made public!
* [2024/08/19] FFAA has been published on Arxiv!

## Contents

* [Install](#install)
* [Model Zoo](docs/MODEL_ZOO.md)
* [Dataset](docs/dataset.md)
* [Train](#train)
* [Evaluation](#evaluation)
* [Inference](#inference)

## Install

1. Clone this repository and navigate to FFAA folder
```
git clone https://github.com/thu-huangzc/FFAA.git
cd FFAA
```

2. Install Package
```
conda create -n ffaa python=3.9 -y
conda activate ffaa
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
```
> The training dataset has not been made public so you don't need to install these packages now.

### Upgrade to latest code base
```
git pull
pip install -e .
```


## Train

FFAA training consists of two stages: (1) Fine-tuning MLLM with hypothetical prompts stage: We introduce hypothetical prompts to the 20K FFA-VQA dataset that presume the face is either real or fake prior to analysis. By fine-tuning the MLLM on this dataset, we enable the generation of answers based on varying hypotheses; (2) Training MIDS with historical answers stage: Utilize the fine-tuned MLLM to extract answers from unused samples in the MA dataset for training.

FFAA is trained on **2 RTX 3090 GPUs with 24GB memory**.

### Hyperparameters
1. LoRA Fine-tuning

| Module                | Global Batch Size | Learning rate | Epochs | LoRA Rank | LoRA alpha |
| --------------------- | ----------------- | ------------- | ------ | --------- | ---------- |
| LLaVA-v1.6-Mistral-7B | 16                | 1e-4          | 3      | 32        | 48         |

2. Train MIDS

| Module | Global Batch Size | Learning rate | Epochs | Weight decay |
| ------ | ----------------- | ------------- | ------ | ------------ |
| MIDS   | 48                | 1e-4          | 2      | 1e-5         |


Other hyperparameters settings can be found at [<u>scripts/train</u>](scripts/train).

### Download LLaVA checkpoints
We utilize LLaVA as our base MLLM module. Certainly, you can choose any other MLLMs as the backbone. In the paper, we select LLaVA-v1.6-Mistral-7B and here are the available download links: [<u>Huggingface</u>](https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b), [<u>Model Scope</u>](https://www.modelscope.cn/models/ai-modelscope/llava-v1.6-mistral-7b)

### Fine-tune MLLM with Hypothetical prompts
Download the 20K FFA-VQA dataset containing hypothetical prompts and place the folder in `./playground/`

Training script with DeepSpeed: [<u>finetune_mistral_lora.sh</u>](scripts/train/finetune_mistral_lora.sh)

### Train MIDS
1. Prepare data

You can use the fine-tuned MLLM to obtain the historical answer data from the unused face images. For each image, we utilize one non-hypothetical prompt and two hypothetical prompts to get three answers. In fact, MIDS is designed to select the correct one from multiple answers. Therefore, the image won't be added into the dataset if the results of the three answers are the same.

Certainly, you can download 90K Mistral-FFA-VQA dataset we provided.

After downloading the dataset, place the folder in `./playground/`

2. Training

Training script with DeepSpeed: [<u>train_mids_v1.sh</u>](scripts/train/train_mids_v1.sh)

## Evaluation

We evaluate models on OW-FFA-Bench, which consists of 6 generalization test sets. 

First, download OW-FFA-Bench and the test set of Multi-attack in [<u>benchmark.md</u>](docs/benchmark.md)

Second, organize the folders in `./benchmark` as follows:
```
benchmark/
  dfd/
    imgs/
    dfd.json
  dfdc/
  dpf/
  ma/
  mffdi/
  pgc/
  wfir/
```
Third, run eval scripts:
1. Evaluate LLaVA only
```
CUDA_VISIBLE_DEVICES=0 python eval_llava.py --benchmark BENCHMARK_NAME --model_name llava-mistral-7b
```
2. Evaluate FFAA
```
CUDA_VISIBLE_DEVICES=0 python eval.py --benchmark BENCHMARK_NAME --model mistral --generate_num 3 --eval_num -1
```
* BENCHMARK_NAME: `dfd`, `dfdc`, `dpf`, `ma`, `mffdi`, `pgc`, `wfir`.
* generate_num: the number of generated answers for each input image. 
* eval_num: the number of images which will be evaluated. `-1` represents all.

Last, the results will be saved in `./results/`.

## Inference

You can place the test images in `./playground/test_images`. The prompts are shown in [<u>prompts.txt</u>](playground/prompts.txt). 

Then you can change the image path which you want to test in `inference.py` and run as follows:
```
CUDA_VISIBLE_DEVICES=0 python inference.py --crop 1 --visualize 1
```
* crop: `1` means the face in the image will be automatically cropped.
* visualize: `1` means the heatmaps of MIDS will be visulized and saved in `./heatmaps/`.

## Citation

If you find FFAA useful for your research and applications, please cite using this BibTeX:

```
@article{huang2024ffaa,
         title={FFAA: Multimodal Large Language Model based Explainable Open-World Face Forgery Analysis Assistant},
         author={Huang, Zhengchao and Xia, Bin and Lin, Zicheng and Mou, Zhun and Yang, Wenming},
         journal={arXiv preprint arXiv:2408.10072},
         year={2024}
}
```


