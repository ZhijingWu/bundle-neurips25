# Code for paper "Text Bundling of LLMs for Zero-Shot Inference on Text-Attributed Graphs"

This repository contains the code for the paper "Text Bundling of LLMs for Zero-Shot Inference on Text-Attributed Graphs". To reproduce the results, please follow the instructions below.

## Environmental Setup
Use the following command in a linux system to prepare the environment:
```bash
conda create -n llmbp python==3.8.18 
conda activate llmbp
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pyg_lib==0.3.1+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html 
pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html 
pip install torch_sparse==0.6.18+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html 
pip install torch_cluster==1.6.3+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html 
pip install torch_spline_conv==1.2.2+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install transformers==4.46.3 
pip install sentence_transformers==2.2.2
pip install dgl==2.4.0+cu121 -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html 
pip install openai 
pip install torch_geometric==2.5.0 
pip install protobuf 
pip install accelerate
```

## Data Setup
Download the data from the following Hugging Face Repository provided by Wang et al. [[Link]](https://huggingface.co/datasets/Graph-COM/Text-Attributed-Graphs)

Put the data in `dataset/` folder.

## Model Setup
Please set your OpenAI API key as follows:
```bash
export OPENAI_API_KEY=<your_api_key>
```

## Running the Code
Please use the following command to run the code:
```bash
python bundle.py --device <your_gpu_id> --dataset <the_dataset_to_run> --bundle_size 5 --num_samples 100 --sample_criterion <neighbor_or_feature_depends_on_the_dataset>  --query_type gpt --model <the_name_of_GPT> --loss_type ranking --gnn_type gcn --stages 300 100 100 --lr 0.001 --wd 0.001 --resample
```
