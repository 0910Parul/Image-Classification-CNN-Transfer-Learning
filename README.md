# Image Classification Using CNNs and Transfer Learning

## Project Overview

This project classifies dog and cat photographs and compares a CNN trained from scratch with ImageNet-pretrained EfficientNetB4 and ResNet50 models. The objective is to evaluate how model choice, fine-tuning, and probability handling affect validation accuracy and Kaggle log loss.

The dataset contains **25,000 labeled images** and **12,500 competition test images** from [Dogs vs. Cats Redux](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data). Training was performed in Kaggle notebooks.

## Key Highlights

- **Baseline comparison:** the saved CNN run reached **77.12% validation accuracy**; the EfficientNetB4 fine-tuning run reached **99.14%**.
- **Competition result:** the report records a best Kaggle log loss of **0.04553**, with a submission screenshot.
- **Evaluation design:** an 80/20 split produced **20,000 training images** and **5,000 validation images**.
- **Practical workflow:** image organization, augmentation, transfer learning, fine-tuning, and a 12,500-row submission artifact.

Explore the [analysis notebook](Pet_Image_Classifier.ipynb), [report](report/Pet_Image_Classifier_Report.pdf), and [results evidence](docs/RESULTS_EVIDENCE.md).

## Technical Implementation

1. **Prepare images:** extract the competition archives, organize class folders, and create training/validation generators. The main saved runs use 224×224 images.
2. **Establish a baseline:** train a CNN with three convolution/pooling blocks (32, 64, and 128 filters), Dense(128), dropout, and a sigmoid output.
3. **Apply transfer learning:** use an ImageNet-pretrained EfficientNetB4 backbone with global average pooling, batch normalization, dense layers, and dropout. Train a frozen base at learning rate 1e-3.
4. **Fine-tune:** unfreeze the last 30 backbone layers and recompile at learning rate 1e-5.
5. **Evaluate and submit:** monitor validation loss with early stopping, generate test probabilities, clip them to [0.01, 0.99], and export predictions by image ID.
6. **Compare architectures:** retain a ResNet50 setup in the appendix. Its reported performance comes from a separate run documented in the report.

Augmentation includes flips, rotation, zoom, and, for the EfficientNet generator, shifts and shear. The saved EfficientNet validation generator also applies augmentation, so validation scores describe that evaluation setup.

## Methods & Technologies

- **Python:** TensorFlow/Keras, NumPy, pandas, Matplotlib, and Pillow.
- **Computer vision:** CNNs, EfficientNetB4, ResNet50, ImageNet preprocessing, and ImageDataGenerator.
- **Training:** binary cross-entropy, Adam, dropout, batch normalization, early stopping, and transfer-learning fine-tuning.
- **Environment:** Kaggle notebooks with GPU training; the notebook records TensorFlow 2.19.0.

## Results & Insights

| Model / submission | Best validation accuracy | Kaggle log loss ↓ | Evidence |
|---|---:|---:|---|
| CNN from scratch | 77.12% | — | Saved training output |
| ResNet50, fine-tuned and clipped | Approximately 98% | 0.05453 | Report and separate-run description |
| EfficientNetB4, without clipping | Approximately 99% | 0.06584 | Report |
| EfficientNetB4, clipped submission | 99.14% best recorded fine-tuning validation accuracy | **0.04553** | Notebook training output; report for submission score |

- **Transfer learning provided the largest recorded accuracy improvement:** EfficientNetB4 reached 98.98% validation accuracy after its first frozen-base epoch, compared with the CNN's best 77.12%.
- **Probability handling matters for log loss:** the report attributes the lower EfficientNet score to clipping. The reported scores differ by about **31%** (0.06584 to 0.04553); this comparison is not a controlled calibration study.
- **Different metrics answer different questions:** validation accuracy measures class decisions on the validation split, while Kaggle log loss evaluates submitted probabilities on competition data.

### Result scope

The notebook marks reconstructed EfficientNet setup and edited prediction cells; the ResNet training history is not included. The report and notebook also differ on epoch counts and ResNet input size. The saved submission CSV contains probabilities outside the edited code's clipping range, so it cannot be assumed to be the clipped submission. Best epoch accuracy may differ from the checkpoint restored by early stopping on validation loss. These results are preserved as historical evidence, rather than presented as an independently reproduced benchmark. Details are in [the evidence notes](docs/RESULTS_EVIDENCE.md).

## Repository Structure

| Path | Purpose |
|---|---|
| [Pet_Image_Classifier.ipynb](Pet_Image_Classifier.ipynb) | Baseline CNN, EfficientNetB4 training, submission generation, and ResNet50 appendix |
| [report/Pet_Image_Classifier_Report.pdf](report/Pet_Image_Classifier_Report.pdf) | Methodology, comparisons, and Kaggle submission screenshot |
| [outputs/cnn_training_curves.png](outputs/cnn_training_curves.png) | Saved baseline training curves |
| [outputs/sample_training_images.png](outputs/sample_training_images.png) | Sample input images |
| [outputs/kaggle_submission_efficientnet.csv](outputs/kaggle_submission_efficientnet.csv) | Saved test-image probabilities |
| [requirements.txt](requirements.txt) | Python dependencies |
| [docs/RESULTS_EVIDENCE.md](docs/RESULTS_EVIDENCE.md) | Metric provenance and reproduction notes |

## Getting Started

1. Download `train.zip` and `test.zip` from the competition page; image files are not stored in this repository.
2. Open the notebook in Kaggle, attach the competition dataset, and enable a GPU. The saved paths use `/kaggle/input/competitions/dogs-vs-cats-redux-kernels-edition/` and `/kaggle/working/`.
3. Review the marked reconstruction notes before running. The baseline training cell refers to `train_generator` and `val_generator`, while the preceding setup defines `train_gen` and `val_gen`; these names need alignment for a full rerun.
4. Execute preparation, training, and prediction cells in order. The appendix is a separate comparison setup.
5. Inspect validation behavior and submission probabilities before uploading a newly generated CSV to Kaggle.

For another Python environment, install [requirements.txt](requirements.txt), provide a notebook interface, and adapt the Kaggle-specific paths. Dependencies are unpinned, so the current environment may differ from the saved run. No training was rerun for this documentation change.

## Applications & Takeaways

The project demonstrates an evaluation workflow for visual classification: establish a baseline, compare pretrained backbones, examine overfitting, and choose outputs that match the scoring objective. Similar methods can support image tagging or visual triage after evaluation on the intended data.

For decision makers, the key lesson is to assess both classification quality and probability quality before selecting a model. Production accuracy, latency, and business savings were not measured.

## Project Context

Developed for **MSBA 6421 — Predictive Analytics**, Carlson School of Management, University of Minnesota. Existing notebook names, reconstruction disclosures, reports, and saved artifacts are retained.

---

By **Parul Chaudhary** · [LinkedIn](https://www.linkedin.com/in/parulchaud) · [Email](mailto:parul.jaswant@gmail.com)
