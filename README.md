# SAViT - A general framework to enhance both data and parameter efficiency in medical image analysis.

This is the repo for the paper [A General Framework for Efficient Medical Image Analysis via Shared Attention Vision Transformer]:

- SAViT is data-efficient when trained from scratch on limited medical images.
- SAViT is parameter-efficient, requiring fewer trainable parameters in PEFT.
- SAViT can be efficiently to improve pre-trained models through full fine-tuning.

## Get Started

**Training from scratch**
```bash
python train_SAViT.py \
--data_path ./your_datasets/ \
--nb_classes 5
```

**Training PEFT**
```bash
python train_PEFT.py  \
--data_path ./your_datasets/ \
--nb_classes 5 \
--finetune ./vit_base_patch16_224_in21k.pth
```

**Training Full Fine-Tuning**
```bash
python train_FT.py \
--data_path ./your_datasets/ \
--nb_classes 5 \
--finetune ./vit_base_patch16_224_in21k.pth
```

## 🙏 Acknowledgement

The code is based on [RETFound](https://github.com/rmaphoh/RETFound_MAE#fine-tuning-with-retfound-weights) . We greatly thank the authors for releasing the code.

