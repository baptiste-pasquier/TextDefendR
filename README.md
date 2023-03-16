# NLP adversarial attacks <!-- omit from toc -->

[![Code quality](https://github.com/baptiste-pasquier/nlp-adversarial-attacks/actions/workflows/quality.yml/badge.svg)](https://github.com/baptiste-pasquier/nlp-adversarial-attacks/actions/workflows/quality.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

- [1. Installation](#1-installation)
- [2. Usage](#2-usage)
  - [2.1. DistilCamemBERT fine-tuning on Allociné](#21-distilcamembert-fine-tuning-on-allociné)
    - [2.1.1. Model fine-tuning (optional)](#211-model-fine-tuning-optional)
    - [2.1.2. Model evaluation](#212-model-evaluation)
  - [2.2. TCAB Dataset Generation](#22-tcab-dataset-generation)
    - [2.2.1. Download the Allociné dataset](#221-download-the-allociné-dataset)
    - [2.2.2. Run some attacks with TextAttack (optional)](#222-run-some-attacks-with-textattack-optional)
    - [2.2.3. Run attacks for TCAB dataset](#223-run-attacks-for-tcab-dataset)
  - [2.3. TCAB Benchmark](#23-tcab-benchmark)
    - [2.3.1. Generate the whole dataset](#231-generate-the-whole-dataset)
    - [2.3.2. Encode the dataset with feature extraction](#232-encode-the-dataset-with-feature-extraction)
    - [2.3.3. Compile the joblib features in a dataframe](#233-compile-the-joblib-features-in-a-dataframe)

## 1. Installation

1. Clone the repository
```bash
git clone https://github.com/baptiste-pasquier/nlp-adversarial-attacks
```

2. Install the project
- With `poetry` ([installation](https://python-poetry.org/docs/#installation)):
```bash
poetry install
```
- With `pip` :
```bash
pip install -e .
```

3. Install pre-commit
```bash
pre-commit install
```

4. (Optional) Install Pytorch CUDA
```bash
poe torch_cuda
```

## 2. Usage

### 2.1. DistilCamemBERT fine-tuning on Allociné

#### 2.1.1. Model fine-tuning (optional)



- Fine-tune with TextAttack:
```{bash}
textattack train --model cmarkea/distilcamembert-base --dataset allocine --num-epochs 3 --learning_rate 5e-5 --num_warmum_steps 500 --weight_decay 0.01 --per-device-train-batch-size 16 --gradient_accumulation_steps 4 --load_best_model_at_end true --log-to-tb
```
- Fine-tune with Transformers: [model_finetuning.ipynb](notebooks/model_finetuning.ipynb)

The fine-tuned model is available on HuggingFace: https://huggingface.co/baptiste-pasquier/distilcamembert-allocine

#### 2.1.2. Model evaluation
- Evaluate the fine-tuned model with TextAttack:
```{bash}
textattack eval --model-from-huggingface baptiste-pasquier/distilcamembert-allocine --dataset-from-huggingface allocine --num-examples 1000 --dataset-split test
```
- Evaluate with Transformers: [model_evaluation.ipynb](notebooks/model_evaluation.ipynb)

The model offers an accuracy score of 97%.


### 2.2. TCAB Dataset Generation

This section provides
a database of attacks with a fine-tuned DistilCamemBERT model on the task of Allociné reviews classification.

:globe_with_meridians: Reference: https://github.com/react-nlp/tcab_generation

#### 2.2.1. Download the Allociné dataset

Run
```{bash}
python scripts/download_data.py
```
This generates a `train.csv`, `val.csv`, and `test.csv`.

#### 2.2.2. Run some attacks with TextAttack (optional)

Run
```{bash}
textattack attack --model-from-huggingface baptiste-pasquier/distilcamembert-allocine --dataset-from-huggingface allocine --recipe deepwordbug --num-examples 50
```

#### 2.2.3. Run attacks for TCAB dataset

Run
```{bash}
python scripts/attack.py
```
:memo: Usage
```
usage: attack.py [-h] [--dir_dataset DIR_DATASET] [--dir_out DIR_OUT]
                 [--task_name TASK_NAME] [--model_name MODEL_NAME]
                 [--pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH]
                 [--model_max_seq_len MODEL_MAX_SEQ_LEN]
                 [--model_batch_size MODEL_BATCH_SIZE] [--dataset_name DATASET_NAME]
                 [--target_model_train_dataset TARGET_MODEL_TRAIN_DATASET]
                 [--attack_toolchain ATTACK_TOOLCHAIN] [--attack_name ATTACK_NAME]
                 [--attack_query_budget ATTACK_QUERY_BUDGET]
                 [--attack_n_samples ATTACK_N_SAMPLES] [--random_seed RANDOM_SEED]

options:
  -h, --help            show this help message and exit
  --dir_dataset DIR_DATASET
                        Central directory for storing datasets. (default: data/)
  --dir_out DIR_OUT     Central directory for storing attacks. (default: attacks/)
  --task_name TASK_NAME
                        e.g., abuse, sentiment or fake_news. (default: sentiment)
  --model_name MODEL_NAME
                        Model type. (default: distilcamembert)
  --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                        Fine-tuned model configuration to load from cache or download
                        (HuggingFace). (default: baptiste-pasquier/distilcamembert-
                        allocine)
  --model_max_seq_len MODEL_MAX_SEQ_LEN
                        Max. no. tokens per string. (default: 512)
  --model_batch_size MODEL_BATCH_SIZE
                        No. instances per mini-batch. (default: 32)
  --dataset_name DATASET_NAME
                        Dataset to attack. (default: allocine)
  --target_model_train_dataset TARGET_MODEL_TRAIN_DATASET
                        Dataset used to train the target model. (default: allocine)
  --attack_toolchain ATTACK_TOOLCHAIN
                        e.g., textattack or none. (default: textattack)
  --attack_name ATTACK_NAME
                        Name of the attack; clean = no attack. (default: deepwordbug)
  --attack_query_budget ATTACK_QUERY_BUDGET
                        Max. no. of model queries per attack; 0 = infinite budget.
                        (default: 0)
  --attack_n_samples ATTACK_N_SAMPLES
                        No. samples to attack; 0 = attack all samples. (default: 10)
  --random_seed RANDOM_SEED
                        Random seed value to use for reproducibility. (default: 0)
```

:bulb: Notebook step-by-step: [run_attack.ipynb](/notebooks/run_attack.ipynb)

:bulb: Notebook attack statistics: [attack_statistics.ipynb](/notebooks/attack_statistics.ipynb)


### 2.3. TCAB Benchmark

:globe_with_meridians: Reference: https://github.com/react-nlp/tcab_benchmark

#### 2.3.1. Generate the whole dataset

Run
```{bash}
python scripts/generate_catted_dataset.py
```

#### 2.3.2. Encode the dataset with feature extraction

- TP: text properties
- TM: target model properties
- LM: language model properties

Run
```{bash}
python scripts/encode_main.py
```
:memo: Usage
```
usage: encode_main.py [-h] [--target_model TARGET_MODEL]
                      [--target_model_dataset TARGET_MODEL_DATASET]
                      [--target_model_train_dataset TARGET_MODEL_TRAIN_DATASET]
                      [--attack_name ATTACK_NAME]
                      [--max_clean_instance MAX_CLEAN_INSTANCE] [--tp_model TP_MODEL]
                      [--lm_perplexity_model LM_PERPLEXITY_MODEL]
                      [--lm_proba_model LM_PROBA_MODEL]
                      [--target_model_name_or_path TARGET_MODEL_NAME_OR_PATH]
                      [--test TEST] [--disable_tqdm DISABLE_TQDM]

options:
  -h, --help            show this help message and exit
  --target_model TARGET_MODEL
                        Target model type. (default: distilcamembert)
  --target_model_dataset TARGET_MODEL_DATASET
                        Dataset attacked. (default: allocine)
  --target_model_train_dataset TARGET_MODEL_TRAIN_DATASET
                        Dataset used to train the target model. (default: allocine)
  --attack_name ATTACK_NAME
                        Name of the attack or ALL or ALLBUTCLEAN. (default: ALL)
  --max_clean_instance MAX_CLEAN_INSTANCE
                        Only consider certain number of clean instances; 0 = consider
                        all. (default: 0)
  --tp_model TP_MODEL   Sentence embeddings model for text properties features.
                        (default: sentence-transformers/bert-base-nli-mean-tokens)
  --lm_perplexity_model LM_PERPLEXITY_MODEL
                        GPT2 model for lm perplexity features. (e.g. gpt2,
                        gpt2-medium, gpt2-large, gpt2-xl, distilgpt2) (default: gpt2)
  --lm_proba_model LM_PROBA_MODEL
                        Roberta model for lm proba features. (e.g. roberta-base,
                        roberta-large, distilroberta-base) (default: roberta-base)
  --target_model_name_or_path TARGET_MODEL_NAME_OR_PATH
                        Fine-tuned target model to load from cache or download
                        (HuggingFace). (default: baptiste-pasquier/distilcamembert-
                        allocine)
  --test TEST           If True only computes first 10 instance. (default: False)
  --disable_tqdm DISABLE_TQDM
                        If True silent tqdm progress bar. (default: False)
```

For instance, to use the `distilcamembert` model trained on the `allocine` dataset as the target model, a version of distilcamembert for the tp and lm probabilities, and a French implementation of GPT2 for the perplexity (those are the default values), run

```{bash}
python scripts/encode_main.py \
    --tp_model cmarkea/distilcamembert-base-nli \
    --lm_perplexity_model asi/gpt-fr-cased-small \
    --lm_proba_model cmarkea/distilcamembert-base \
    --target_model_name_or_path baptiste-pasquier/distilcamembert-allocine \
    --prefix_file_name fr+small \
    --tasks ALL
```

This will create the file `data_tcab/reprs/samplewise/fr+small_distilcamembert_allocine_ALL_ALL.joblib`. This command is quite long, around 7 hours for the allocine dataset.

To generate only the tp features, with another model, run

```{bash}
python scripts/encode_main.py \
    --tp_model google/canine-c \
    --prefix_file_name fr+canine \
    --tasks TP
```

This will create the file `data_tcab/reprs/samplewise/fr+canine_distilcamembert_allocine_ALL_TP.joblib`. This is quite fast compared to the previous command (around 5 minutes).

:bulb: Notebook step-by-step: [run_encode_main.ipynb](/notebooks/run_encode_main.ipynb)

:bulb: Notebook step-by-step for encode_samplewise_features: [encode_samplewise_features.ipynb](/notebooks/encode_samplewise_features.ipynb)

:bulb: Notebook for feature extraction: [feature_extraction.ipynb](/notebooks/feature_extraction.ipynb)


#### 2.3.3. Compile the joblib features in a dataframe

Run

```{bash}
python scripts/generate_feature_dataset.py
```

:memo: Usage

```{bash}
Usage: generate_feature_dataset.py [OPTIONS]

  Generate the dataframe from the joblib file

Options:
  --joblib_path TEXT  Path to the joblib file
                      (default: data_tcab/reprs/samplewise/fr+small_distilcamembert_allocine_ALL.joblib)
  --use_canine        Add CANINE features to the dataframe
                      (default: False)
  --canine_path TEXT  Path to the canine joblib file
                      (default: data_tcab/reprs/samplewise/fr+canine_distilcamembert_allocine_ALL_TP.joblib)
  --disable_tqdm      Disable tqdm progress bar
                      (default: False)
  --help              Show this message and exit.
```

:bulb: Notebook step-by-step: [create_feature_dataframe.ipynb](/notebooks/create_feature_dataframe.ipynb)
