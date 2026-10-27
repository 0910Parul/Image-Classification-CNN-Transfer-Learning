# Classifying Pet Photos with Deep Learning: From a Baseline CNN to 99% with Transfer Learning

This project trains computer-vision models to tell **dogs from cats** in photographs, using **25,000 labeled training images** and **12,500 test images** from Kaggle's *Dogs vs. Cats Redux* competition. A CNN built from scratch reached **77% validation accuracy**. Transfer learning with a fine-tuned **EfficientNetB4** reached **99.1%** and a Kaggle **log loss of 0.04553**, the best of three submissions. Built for **MSBA 6421: Predictive Analytics** (Carlson, University of Minnesota) and trained on Kaggle GPUs.

## 🔍 Project Goals

- Build an image classifier that reliably labels pet photos as dog or cat.
- Measure how much **transfer learning** from ImageNet-pretrained networks improves on a CNN trained from scratch.
- Get well-calibrated probabilities that score well under Kaggle's **log-loss** metric, while controlling overfitting.

## 📄 Files Included

- **`Pet_Image_Classifier.ipynb`**: the full pipeline, covering data organization, augmentation, the baseline CNN, EfficientNetB4 transfer learning and fine-tuning, and the Kaggle submission. It also has a ResNet50 comparison setup in an appendix.
- **`report/Pet_Image_Classifier_Report.pdf`**: write-up of preprocessing, model exploration, overfitting prevention, the validation strategy and the Kaggle leaderboard scores.
- **`outputs/`**: sample training images, CNN training curves and the EfficientNetB4 Kaggle submission file.
- **`requirements.txt`**: Python dependencies.

> The images (~850 MB) aren't stored in this repo. You can download them from the [Kaggle competition page](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data).
>
> **About the notebook:** two cells, marked 🔧, were rebuilt from the architecture documented in the report because they were missing from the saved notebook. All training outputs come from the original Kaggle run.

## 🧪 Methods Used

- **Preprocessing:** images sorted into class folders, resized to 224×224, model-specific ImageNet normalization, and an **80/20 train/validation split** (20,000 / 5,000 images).
- **Data augmentation:** horizontal and vertical flips, rotation (±20°), zoom, width/height shift, shear and brightness changes.
- **Model 1, CNN from scratch:** three convolution and max-pooling blocks (32/64/128 filters), then Dense(128), Dropout(0.5) and a sigmoid output.
- **Models 2 and 3, transfer learning:** ImageNet-pretrained **EfficientNetB4** and **ResNet50**, each with a custom head (GlobalAveragePooling2D, BatchNorm, Dense(512), Dropout(0.5), Dense(256), Dropout(0.3), sigmoid).
  - The pretrained layers were first frozen, with a learning rate of 1e-3.
  - Then the **last 30 layers were unfrozen** and fine-tuned at a learning rate of 1e-5.
- **Overfitting control and validation:** early stopping on validation loss (patience 3, best weights restored), dropout, batch normalization and augmentation.
- **Probability clipping** to [0.01, 0.99], which limits how heavily log loss penalizes confident wrong predictions.

## 📈 Key Findings

| Model | Best validation accuracy | Kaggle log loss ↓ |
|---|---|---|
| CNN from scratch (baseline) | 77.1% | — |
| ResNet50 + fine-tuning (clipped) | ~98% | 0.05453 |
| EfficientNetB4 + fine-tuning (no clipping) | 99.1% | 0.06584 |
| **EfficientNetB4 + fine-tuning (clipped)** | **99.1%** | **0.04553** |

- **Transfer learning made the biggest difference:** EfficientNetB4 reached **98.9% validation accuracy after a single epoch**, while the from-scratch CNN plateaued at about 77% after six.
- **Clipping made the biggest difference to the log-loss score:** clipping the same EfficientNetB4 model's probabilities cut log loss by **31%** (0.06584 → 0.04553), because log loss heavily penalizes rare confident mistakes.
- **Fine-tuning the top pretrained layers beat keeping them frozen.** Batch normalization and stronger augmentation also improved generalization, keeping validation loss close to training loss.

## 🛠 Technologies

- Python, **TensorFlow / Keras**
- CNNs and transfer learning (EfficientNetB4, ResNet50)
- ImageDataGenerator (data augmentation)
- NumPy, pandas, Matplotlib, Pillow
- Kaggle Notebooks (2× NVIDIA Tesla T4 GPUs)

## ✅ Outcome

A pet-photo classifier with about **99% validation accuracy** and a **0.04553 Kaggle log loss**. The project also shows a repeatable approach for image classification with limited compute: start from a pretrained backbone, fine-tune only the top layers, regularize, and calibrate the output probabilities for the evaluation metric.

---

By **Parul Chaudhary** · [LinkedIn](https://www.linkedin.com/in/parulchaud) · [Email](mailto:parul.jaswant@gmail.com)
