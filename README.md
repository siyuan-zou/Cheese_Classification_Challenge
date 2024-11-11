# Cheese Classification challenge
This the hydra and wandb configured codebase of the INF473V - Deep Learning in Computer Vision course challenge at École Polytechnique. The goal is to create a cheese classifier without any real training data. The training data were created using Dreambooth and Texual Inversion from Stable Diffusion 2.

In the folder Dreambooth and Texual_Inversion, I have attached used codes for finetuning.

## Instalation

Cloning the repo:
```
git clone https://github.com/siyuan-zou/Cheese_Classification_Challenge.git
cd Cheese_Classification_Challenge
```
Install dependencies:
```
conda create -n cheese_challenge python=3.10
conda activate cheese_challenge
pip install -r requirements.txt
```

Download the data from kaggle and copy them in the dataset folder
The data should be organized as follow: ```dataset/val```, ```dataset/test```. then the generated train sets will go to ```dataset/train/your_new_train_set```

## Using this codebase
This codebase is centered around 2 components: generating your training data and training your model.
Both rely on a config management library called hydra. It allow you to have modular code where you can easily swap methods, hparams, etc

### Training

To train your model you can run 

```
python train.py
```

This will save a checkpoint in checkpoints with the name of the experiment you have. Careful, if you use the same exp name it will get overwritten

to change experiment name, you can do

```
python train.py experiment_name=new_experiment_name
```

### Download the datasets

We have some generated dataset to increase the generalization ability of the model. For example, photos which describe the production process of each cheese[website][https://drive.google.com/drive/folders/1dhTiOr2a-649_3Gm9Gz-R5lVWEslJZcz]

It can be downloaded directly using python package gdown. First do
```
pip install gdown
```

If you want to download the dataset within ```dataset/train```, make sure to do 
```
mkdir dataset/train/name_of_dataset
gdown https://drive.google.com/drive/folders/1dhTiOr2a-649_3Gm9Gz-R5lVWEslJZcz -O path_to_name_of_dataset --folder --remaining-ok
```

If you want to merge the dateset with ```dataset/train/simple_prompts```, you can do 
```
gdown https://drive.google.com/drive/folders/1dhTiOr2a-649_3Gm9Gz-R5lVWEslJZcz -O path_to_simple_prompts --folder --remaining-ok
```

Make sure to set the google drive folder to be accessible by anyone with the link as a viewer. Make sure to copy the link that ends with the folder ID (1dhTiOr2a-649_3Gm9Gz-R5lVWEslJZcz)here without additional characters.

### Generating datasets
You can generate datasets with the following command

```
python generate.py
```

If you want to create a new dataset generator method, write a method that inherits from `data.dataset_generators.base.DatasetGenerator` and create a new config file in `configs/generate/dataset_generator`.
You can then run

```
python generate.py dataset_generator=your_new_generator
```

In the folder ```prompts``` we stored the propmts corresponding to different types of photos.

## Create submition
To create a submition file, you can run 
```
python create_submition.py experiment_name="name_of_the_exp_you_want_to_score" model=config_of_the_exp
```

Make sure to specify the name of the checkpoint you want to score and to have the right model config

## Acknowledgement

Yining.C
