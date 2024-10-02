# QLoRA-Based Sequence Classification on IMDb with DistilBERT
This repository demonstrates how to fine-tune the DistilBERT model for sequence classification on the IMDb dataset using QLoRA (Quantized Low-Rank Adaptation) for more efficient training. The project leverages tools from Hugging Face's transformers, datasets, accelerate, and PEFT libraries to streamline model training and evaluation.

## Introduction
This project focuses on performing efficient sequence classification with a lightweight variant of BERT, namely DistilBERT. Using the QLoRA technique, we enable parameter-efficient fine-tuning to reduce the computational overhead while maintaining model performance. The IMDb dataset is utilized for the task, with accuracy as the primary evaluation metric.

By using the accelerate library, we optimize training via mixed precision, ensuring that this project can be run on both low and high-end hardware configurations.

## Dependencies
The following Python packages are required for running the project:

transformers 
datasets 
evaluate 
accelerate 
peft

## Dataset
We are using the IMDb dataset from Hugging Face's datasets library. This dataset contains 25,000 training and 25,000 test samples of movie reviews with binary sentiment labels (positive/negative).

## Model Architecture
We use DistilBERT from Hugging Face, which is a smaller, faster version of BERT, designed for efficiency without sacrificing too much performance.
