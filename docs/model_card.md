# RobustLungAI — Model Card

## 1. Model Overview

**Model name:** RobustLungAI AST
**Project:** Robust Respiratory Sound Classification under Noise and Domain Shift using Audio Spectrogram Transformers
**Developer:** Pravesh Subba
**Model architecture:** Audio Spectrogram Transformer (AST)
**Framework:** PyTorch and Hugging Face Transformers
**Task:** Four-class respiratory sound classification
**Status:** Research prototype; not clinically validated

RobustLungAI investigates the classification of respiratory audio into four categories: normal, crackle, wheeze, and both crackle and wheeze. The project studies classification performance, training objectives, data augmentation, robustness to recording conditions, cross-dataset transfer, and interpretability.

## 2. Model and Resources

* **Fine-tuned model repository:** [praveshsubba/robustlungai-ast](https://huggingface.co/praveshsubba/robustlungai-ast)
* **Base checkpoint:** [`MIT/ast-finetuned-audioset-10-10-0.4593`](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
* **Research code and notebooks:** [RobustLungAI GitHub repository](https://github.com/)
* **Demo notebook:** [`notebooks/05b_gradio_demo.ipynb`](../notebooks/05b_gradio_demo.ipynb)
* **Reproducibility guide:** [`reproducibility.md`](reproducibility.md)

The application runs from the Google Colab demo notebook. The model checkpoint is hosted separately on Hugging Face; a permanent hosted application is not provided.

The fine-tuned checkpoint used by the demo is named `ast_lung_fp16.pt`. Consult the model repository and demo notebook for the actual loading procedure and compatibility requirements.

## 3. Intended Use

### Intended uses

* Academic research and education in respiratory audio classification.
* Exploration of AST fine-tuning and alternative training objectives.
* Investigation of model behaviour under selected noise and recording-device conditions.
* Demonstration of audio classification and exploratory interpretability visualizations.

### Out-of-scope uses

* Diagnosing respiratory diseases.
* Recommending or withholding treatment.
* Replacing a clinician, diagnostic test, or medical device.
* Making emergency or other patient-care decisions.
* Claiming clinical validity based solely on internal test results or visualization outputs.

**Important:** This model is a research prototype and is not a clinically validated diagnostic system.

## 4. Input and Output

### Input

The demo accepts respiratory audio recordings. The expected audio format, duration handling, resampling, filtering, and feature preparation are defined by the preprocessing implementation in the demo notebook.

The documented configuration includes:

* Sampling rate: 16,000 Hz.
* Target audio-cycle duration: 8 seconds.
* Mel frequency bins: 128.
* Bandpass filtering: nominally 50–8,000 Hz, subject to the implementation and Nyquist frequency.

These values describe the documented project configuration. Users should verify them against the actual checkpoint and notebook before reproducing results.

### Output

The model predicts one of four classes:

| Class     | Meaning                        |
| --------- | ------------------------------ |
| `normal`  | No crackle or wheeze label     |
| `crackle` | Crackle label                  |
| `wheeze`  | Wheeze label                   |
| `both`    | Both crackle and wheeze labels |

The demo may display model scores and visualizations alongside predictions. These outputs are model estimates, not confirmed clinical findings.

## 5. Training Data

The primary dataset is the [ICBHI 2017 Respiratory Sound Database](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge).

The project also investigates cross-dataset transfer using a selected SPRSound subset containing 355 WAV recordings and 355 corresponding JSON annotation files.

* [SPRSound dataset repository](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound)
* [SPRSound publication](https://doi.org/10.1109/TBCAS.2022.3204910)

The ICBHI experiments use respiratory-cycle annotations mapped to four classes. Dataset partitions, annotation mapping, preprocessing, and inclusion criteria must be checked in the corresponding notebooks.

The original audio datasets are not included in this repository by default. Users must obtain them from their respective sources and follow their applicable licenses and terms.

## 6. Training Procedure

The project investigates the following experiment groups:

1. Traditional machine-learning and CNN baselines.
2. AST fine-tuning with cross-entropy loss.
3. AST fine-tuning with focal loss.
4. Focal loss combined with supervised contrastive learning.
5. SpecAugment and Mixup augmentation ablations.
6. Cross-device generalization.
7. Added-noise robustness.
8. Cross-dataset transfer.
9. Noise-augmented training as a possible mitigation.
10. Attention rollout and Grad-CAM-related interpretability.

The pretrained AST checkpoint is `MIT/ast-finetuned-audioset-10-10-0.4593`.

The project documents a 16 kHz audio configuration, 8-second target cycles, and 128 Mel bins. Exact training hyperparameters, optimizer settings, epoch counts, seeds, and checkpoint-selection criteria must be taken from the relevant experiment notebook. They should not be inferred from the model architecture or the final metrics alone.

## 7. Evaluation Protocol

The results below use an internal patient-level 70/15/15 split. They are not results from the official ICBHI challenge split.

The project reports:

* **Sensitivity:** Abnormal-class sensitivity under the project's normal-versus-abnormal metric definition.
* **Specificity:** Normal-class specificity.
* **ICBHI score:** The average of sensitivity and specificity, expressed as a percentage.

Confirm the metric implementation and averaging procedure against the original evaluation code before using the figures in a publication.

### Baseline and loss-function experiments

| Model                | Split      | Sensitivity (%) | Specificity (%) | Reported ICBHI score (%) |
| -------------------- | ---------- | --------------: | --------------: | -----------------------: |
| Random Forest + MFCC | Validation |           23.83 |           81.45 |                    52.64 |
| Simple CNN           | Validation |           78.94 |           28.51 |                    53.72 |
| Random Forest + MFCC | Test       |           18.79 |           91.45 |                   55.05* |
| Simple CNN           | Test       |           74.58 |           41.89 |                    58.23 |
| AST + Cross-Entropy  | Validation |           67.45 |           74.51 |                    70.98 |
| AST + Cross-Entropy  | Test       |           56.51 |           76.87 |                    66.69 |
| AST + Focal Loss     | Validation |           61.49 |           79.49 |                    70.49 |
| AST + Focal Loss     | Test       |           48.32 |           83.22 |                    65.77 |
| AST + Focal + SupCon | Validation |           66.81 |           71.49 |                    69.15 |
| AST + Focal + SupCon | Test       |           43.49 |           88.86 |                    66.17 |

* The Random Forest test sensitivity and specificity average to approximately 55.12%, which differs from the recorded score of 55.05%. This discrepancy remains unresolved and requires verification against the original experiment output.

### Augmentation ablation

| Configuration       | Split      | Sensitivity (%) | Specificity (%) | ICBHI score (%) |
| ------------------- | ---------- | --------------: | --------------: | --------------: |
| Baseline            | Validation |           54.68 |           85.97 |           70.33 |
| Baseline            | Test       |           48.53 |           79.69 |           64.11 |
| SpecAugment         | Validation |           80.21 |           62.29 |           71.25 |
| SpecAugment         | Test       |           61.55 |           66.71 |           64.13 |
| Mixup               | Validation |           66.38 |           75.11 |           70.75 |
| Mixup               | Test       |           50.42 |           81.66 |           66.04 |
| SpecAugment + Mixup | Validation |           74.47 |           63.95 |           69.21 |
| SpecAugment + Mixup | Test       |           66.66 |           67.84 |           67.22 |

These are recorded project results, not independently verified benchmarks. The two experiment groups should be interpreted separately because their training configurations and experimental procedures may differ.

## 8. Robustness and Interpretability

The project includes experiments involving noise, recording-device differences, cross-dataset transfer, and noise-augmented training. Findings apply only to the tested conditions and data.

The interpretability notebook investigates attention rollout and Grad-CAM-related visualizations. The results include examples of correct classifications, errors, noise conditions, frequency profiles, and randomization analysis.

These visualizations are exploratory. They do not prove causal reasoning, identify disease with clinical certainty, or establish that model decisions are medically valid.

## 9. Limitations and Risks

* The internal patient-level split is not the official ICBHI challenge evaluation protocol.
* Results depend on annotation quality, patient distribution, preprocessing, and training configuration.
* Recording equipment, environmental noise, and domain differences may affect predictions.
* The selected SPRSound subset does not represent every respiratory sound population or recording condition.
* The reported class scores and visualizations have not established clinical validity.
* Model outputs should not be treated as calibrated clinical probabilities without suitable calibration and validation.
* No claim of generalization to all hospitals, devices, age groups, or respiratory conditions is made.
* The model is not a substitute for professional medical assessment.

## 10. Reproducibility and Citation

See [`docs/reproducibility.md`](reproducibility.md) for the notebook workflow, preprocessing details, environment requirements, evaluation protocol, and known reproduction constraints.

Relevant resources:

* [ICBHI 2017 dataset](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)
* [SPRSound dataset](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound)
* [SPRSound publication](https://doi.org/10.1109/TBCAS.2022.3204910)
* [Fine-tuned RobustLungAI checkpoint](https://huggingface.co/praveshsubba/robustlungai-ast)
* [AST base checkpoint](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

**Responsible-use statement:** RobustLungAI is an MSc research and educational project. It has not been clinically validated and must not be used as the sole basis for medical decisions.
