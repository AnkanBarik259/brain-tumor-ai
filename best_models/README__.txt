# Best Models

This directory contains the best-performing model checkpoints obtained during training and validation.

Due to the use of cloud-based training environments with session time limitations, model checkpoints and training logs were saved periodically to ensure training continuity, reproducibility, and recovery from interrupted sessions.

The saved checkpoints allow training to resume from the latest completed epoch while preserving the best-performing model discovered during previous training stages. For example, if training resumes from epoch 20 but the highest validation performance was achieved at epoch 15, the epoch 15 checkpoint remains preserved as the best model until a better-performing checkpoint is produced.

### 📂 Included Models

* **fast_seg_best.pth** — Best tumour segmentation model
* **cls_best.pth** — Best tumour classification model
* **growth_best.pth** — Best tumour growth prediction model
* **survival_best.pth** — Best survival prediction model

### 📈 Training Logs

Associated training logs are included to track training progress, validation performance, checkpoint selection, and experiment reproducibility.

These checkpoints represent the highest-performing models obtained throughout the training process and serve as the primary models for inference, evaluation, and further fine-tuning.

**Developed and trained by Ankan Barik.**
