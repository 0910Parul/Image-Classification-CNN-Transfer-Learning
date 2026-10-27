# Results Evidence and Reproduction Notes

Cell numbers are **1-based positions**, including markdown cells, in [Pet_Image_Classifier.ipynb](../Pet_Image_Classifier.ipynb). Saved outputs and the report were inspected; training and Kaggle submissions were not rerun.

| Claim | Evidence |
|---|---|
| 25,000 training / 12,500 test images | Notebook cell 3 output |
| 20,000 training / 5,000 validation images | Notebook cell 6 output |
| CNN best validation accuracy 0.7712 | Notebook cells 8 and 9 output |
| EfficientNet first-epoch validation accuracy 0.9898; frozen-base best 0.9910 | Notebook cell 12 output |
| EfficientNet fine-tuning best validation accuracy 0.9914 | Notebook cell 13 output |
| Kaggle scores 0.06584, 0.05453, and 0.04553 | [Report](../report/Pet_Image_Classifier_Report.pdf), page 2 text and page 3 submission screenshot |
| Submission probabilities | [Saved CSV](../outputs/kaggle_submission_efficientnet.csv); the image IDs and probability range can be inspected without the original images |

## Interpretation boundaries

- The notebook explicitly marks reconstructed EfficientNet setup and edited prediction code. The retained training outputs do not prove that the reconstructed notebook executes identically to the original run.
- The report attributes 0.06584 to unclipped EfficientNetB4, 0.05453 to ResNet50, and 0.04553 to clipped EfficientNetB4. The screenshot confirms three scores, but does not by itself establish identical model weights or evaluation conditions across submissions.
- The saved CSV has 12,500 unique image IDs (1–12,500), but probabilities range from approximately 9.6e-11 to 1.0, outside the edited code's [0.01, 0.99] range. It cannot independently verify the clipped submission or its score.
- The CNN training cell uses `train_generator`/`val_generator` while the preceding cell creates `train_gen`/`val_gen`. The original code is preserved; the mismatch must be resolved before rerunning.
- The EfficientNet training generator, including random augmentation, is also used to create its validation subset. Treat the validation metric as specific to that setup.
- The main notebook uses 224×224 inputs; the report describes ResNet50 at 256×256, while the appendix setup uses 224×224. The report also lists longer EfficientNet training schedules than the saved outputs.
- The ResNet50 appendix contains a model setup, not the complete saved training history for its reported score.
- Early stopping monitors validation loss and restores its best weights. The maximum validation accuracy across epochs need not describe the restored checkpoint used for submission.
- The approximately 31% log-loss difference is calculated as `(0.06584 - 0.04553) / 0.06584`; it is not evidence of universal gains from clipping or calibrated probabilities.

The documentation update changes no code, reports, images, predictions, or notebook outputs.
