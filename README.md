# Convolution Recurrent Neural Nets
First of all, i would like to thank Jieru Mei. My repo was heavily based on his repo.

This repo implemented convolution recurrent neural nets(CRNN) for handwritten recognition, optical character recognition. CRNN can use for many text levels: character, word, or even a text line.

In this repo, I try to provide a simple API which help everyone can train their own model. This model, which i have used for many industry projects, can guarantee to work well in many cases.

# Dataset
You must organize your dataset in the below structure
```
├── 23963.jpg
├── 23963.txt
├── char
├── test
└── train
```
Here you must put your images with it's labels in the same folder. Each image is a text line, so just put only one line of text in corresponding label file.

*train*: list of all training files <br />
*test*: list of all testing files <br />
*char*: list of all characters in labels <br />

You can look at data *folder* to see an example. 

# Training
## Training options

## Train
Training process was simplified, just type the command in your terminal

```
python train.py --root {root_folder} --train train --val test --alphabet char --pretrain {pretrain_model}
```
To use GPU to speed up your training, just pass ``--cuda`` in options. Your trained model will be exported your `--save_model` in options. Please see the code for more optitions.
# Predict

# Dependences
# Any Problems:
