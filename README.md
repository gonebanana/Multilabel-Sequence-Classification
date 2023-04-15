# Multilabel Sequence Classification Interface with Application To Sentiment Analysis

### 1. Train and save the model
Provided notebook contains an interface for training (`class MultilabelTrainer`) and inference(`class MultilabelClassifier`) of a BERT-based multilabel classification model. The shown example involves Emotion Detection Task for the textual content based on the [sem_eval_2018_task_1](https://huggingface.co/datasets/sem_eval_2018_task_1) dataset.  

[Open Colab](https://colab.research.google.com/drive/1MTZfRvWgJTtqoUoSWaCHh5LPE08_DY3w?usp=sharing)

### 2. Create Docker Container
Following command allows to create a container for deploying the finetuned model. In the Sentiment Classification Task provided model aims to predict scores for each emotion from the pre-defined set of classification labels. To perform conteinerization finetuned model `bert-finetuned-sem_eval-english-0.1.0` should be located in the `app` directory. 

```bash
docker build -t sentiment-classifier-app .
docker run -p 80:80 sentiment-classifier-app
```
