# Berry-Seg Dataset

The Blueberry Semantic Segmentation Dataset (Berry-Seg) supports two blueberry semantic segmentation tasks:

- postharvest blueberry bruise segmentation;
- harvest-stage blueberry fruit segmentation.

The dataset contains four subsets. `Bruise Seg.` and `Fruit Seg.` are used for model training, validation, and testing, while `Ind. Bruise Seg.` and `Ind. Fruit Seg.` are used only for independent external validation.

## Download

The Berry-Seg dataset is publicly available on Kaggle:

**[Download Berry-Seg from Kaggle](KAGGLE_DATASET_LINK)**

## Dataset Samples

![Representative samples of the Berry-Seg dataset](dataset_sample.jpg)

## Dataset Statistics

| Dataset | Images | Instances | Training | Validation | Test |
|---|---:|---:|---:|---:|---:|
| Bruise Seg. | 1,001 | 1,642 | 799 | 101 | 101 |
| Fruit Seg. | 2,772 | 129,080 | 1,940 | 554 | 278 |
| Ind. Bruise Seg. | 864 | 1,164 | 606 | 168 | 90 |
| Ind. Fruit Seg. | 724 | 9,277 | 524 | 145 | 55 |
| **Total** | **5,361** | **141,163** | **3,869** | **968** | **524** |

`Ind.` denotes an independent external-validation dataset.

## Expected Dataset Structure

```text
dataset_root/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Set the dataset path in `src/config.py` before training or evaluation.
