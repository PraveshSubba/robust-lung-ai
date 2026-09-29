# RobustLungAI

**Robust Respiratory Sound Classification under Noise and Domain Shift using Audio Spectrogram Transformers**

RobustLungAI is a personal MSc research project investigating respiratory sound classification using Audio Spectrogram Transformers (AST). It explores four-class classification, training objectives, data augmentation, recording-condition robustness, cross-dataset evaluation, and model interpretability.

> **Research and educational use only.** RobustLungAI is not a clinically validated diagnostic system and must not be used as the sole basis for medical decisions.

## Project at a Glance

* **Task:** Four-class respiratory sound classification
* **Classes:** Normal, crackle, wheeze, and both
* **Architecture:** Audio Spectrogram Transformer (AST)
* **Framework:** PyTorch and Hugging Face Transformers
* **Primary dataset:** ICBHI 2017 Respiratory Sound Database
* **Additional evaluation:** Selected SPRSound subset
* **Interface:** Gradio, run from a Google Colab notebook
* **Model weights:** [RobustLungAI on Hugging Face](https://huggingface.co/praveshsubba/robustlungai-ast)

## Running the Interactive Demo

The repository does not host a permanent live demo. You can run the Gradio application yourself using the demo notebook in Google Colab.

### Steps

1. Open [`notebooks/05b_gradio_demo.ipynb`](notebooks/05b_gradio_demo.ipynb).
2. Open the notebook in Google Colab.
3. Review the notebook and install any required dependencies.
4. Run the cells in order, including the cell that downloads and loads the fine-tuned checkpoint from Hugging Face.
5. Run the final cell to launch the Gradio interface.
6. Open the local or temporary public URL provided by Gradio, depending on the notebook's launch configuration.

A temporary public Gradio URL is not a permanent deployment. It may stop working when the Colab runtime ends.

The notebook downloads the model checkpoint from the [RobustLungAI Hugging Face repository](https://huggingface.co/praveshsubba/robustlungai-ast). Access to the checkpoint and compatibility with the notebook's dependencies are required to run the demo.

## Model and Preprocessing

The project uses the pretrained AST checkpoint:

`MIT/ast-finetuned-audioset-10-10-0.4593`

The documented preprocessing configuration includes:

* Sampling rate: 16,000 Hz
* Target audio-cycle duration: 8 seconds
* Mel frequency bins: 128
* Audio filtering: bandpass filtering, with implementation-specific settings

The exact preprocessing, normalization, tensor dimensions, and checkpoint-loading procedure must match the configuration used by the relevant experiment and trained model.

The fine-tuned checkpoint is hosted separately on [Hugging Face](https://huggingface.co/praveshsubba/robustlungai-ast), rather than being duplicated in this GitHub repository.

## Research Objectives

1. Establish traditional machine-learning and CNN baselines.
2. Fine-tune AST using cross-entropy and focal loss.
3. Investigate supervised contrastive learning.
4. Compare SpecAugment and Mixup configurations.
5. Evaluate robustness to added noise and recording-device differences.
6. Investigate cross-dataset transfer using a selected SPRSound subset.
7. Explore attention rollout and Grad-CAM visualizations.

## Experimental Results

The following results were recorded using an internal patient-level 70/15/15 split. Scores are percentages. This split is not the official ICBHI challenge evaluation protocol.

The ICBHI score is reported as the average of abnormal-class sensitivity and normal-class specificity. Metric definitions and aggregation should be checked against the corresponding experiment outputs.

### Baseline and loss-function experiments

| Model                | Split      | Sensitivity (%) | Specificity (%) | ICBHI score (%) |
| -------------------- | ---------- | --------------: | --------------: | --------------: |
| Random Forest + MFCC | Validation |           23.83 |           81.45 |           52.64 |
| Simple CNN           | Validation |           78.94 |           28.51 |           53.72 |
| Random Forest + MFCC | Test       |           18.79 |           91.45 |          55.05* |
| Simple CNN           | Test       |           74.58 |           41.89 |           58.23 |
| AST + Cross-Entropy  | Validation |           67.45 |           74.51 |           70.98 |
| AST + Cross-Entropy  | Test       |           56.51 |           76.87 |           66.69 |
| AST + Focal Loss     | Validation |           61.49 |           79.49 |           70.49 |
| AST + Focal Loss     | Test       |           48.32 |           83.22 |           65.77 |
| AST + Focal + SupCon | Validation |           66.81 |           71.49 |           69.15 |
| AST + Focal + SupCon | Test       |           43.49 |           88.86 |           66.17 |

* **Metric verification required:** The reported Random Forest test sensitivity and specificity average to approximately 55.12%, rather than the recorded score of 55.05%. This discrepancy has not yet been resolved.

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

The loss-function and augmentation experiments are separate experiment groups. Their results should be interpreted in the context of their individual training configurations, checkpoints, and evaluation procedures. Values in this README are recorded experiment outputs and should be verified against the original result files before formal publication.

See [`results/tables/results_phase3_combined.csv`](results/tables/results_phase3_combined.csv) for the consolidated Phase 3 results.

## Robustness and Cross-Dataset Evaluation

The project includes notebooks investigating:

* **Cross-device generalization:** Changes in performance across recording devices or acquisition conditions.
* **Noise robustness:** Evaluation under different added-noise signal-to-noise ratios.
* **Cross-dataset transfer:** Evaluation using a selected subset of SPRSound.
* **Noise-augmented mitigation:** Experiments investigating whether noise augmentation changes performance under noisy conditions.

The selected SPRSound subset contains 355 WAV recordings and 355 corresponding JSON annotation files. Cross-dataset findings are specific to the selected subset, annotation mapping, preprocessing, and evaluation procedure. They do not establish general clinical validity.

Dataset: [SPRSound official repository](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound)

Publication: Zhang et al., *SPRSound: Open-Source SJTU Paediatric Respiratory Sound Database* (2022). [DOI: 10.1109/TBCAS.2022.3204910](https://doi.org/10.1109/TBCAS.2022.3204910)

## Interpretability

The [`05a-interpretability.ipynb`](notebooks/05a-interpretability.ipynb) notebook investigates model interpretability using attention rollout and Grad-CAM-related visualizations.

The `results/figures/` directory contains figures associated with augmentation ablation, device generalization, frequency profiles, interpretability examples, noise robustness, mitigation comparisons, and randomization analysis.

These visualizations are exploratory. Attention maps and attribution methods do not prove that the model uses causal reasoning, identify disease with clinical certainty, or independently validate its predictions.

## Repository Structure

```text
RobustLungAI/
├── docs/
│   ├── model_card.md
│   └── reproducibility.md
├── notebooks/
│   ├── 01-data-preparation.ipynb
│   ├── 02-baseline-models.ipynb
│   ├── 03a-ast-finetuning-ce.ipynb
│   ├── 03b-ast-focal-loss.ipynb
│   ├── 03c-ast-subcon.ipynb
│   ├── 03d-ast-augmentation-ablation.ipynb
│   ├── 04a-cross-device-generalization.ipynb
│   ├── 04b-noise-robustness-sweep.ipynb
│   ├── 04c-cross-dataset-transfer.ipynb
│   ├── 04d-noise-augmented-mitigation.ipynb
│   ├── 04-final-results.ipynb
│   ├── 05a-interpretability.ipynb
│   └── 05b_gradio_demo.ipynb
├── results/
│   ├── figures/
│   └── tables/
│       └── results_phase3_combined.csv
├── README.md
└── requirements.txt
```

This tree reflects the current repository organization. Empty directories that are not tracked by Git are omitted.

## Documentation

* [Model card](docs/model_card.md)
* [Reproducibility guide](docs/reproducibility.md)
* [Experiment results](results/tables/)
* [Gradio demo notebook](notebooks/05b_gradio_demo.ipynb)
* [Interpretability notebook](notebooks/05a-interpretability.ipynb)

## Reproducing Experiments

Experiments were developed primarily in Kaggle notebooks, while the interactive demo is designed to run in Google Colab.

Reproducing an experiment may require access to the original dataset, prepared manifests, waveform caches, compatible dependencies, model checkpoints, and experiment-specific configurations.

Recommended workflow:

1. Review the data-preparation notebook.
2. Follow the notebook for the experiment you want to reproduce.
3. Check its recorded split, preprocessing, training configuration, and evaluation metric.
4. Review the final-results notebook and corresponding result files.
5. To run the interactive application, open `05b_gradio_demo.ipynb` in Google Colab and execute its cells in order.

The final-results notebook aggregates recorded experiment outputs; it does not retrain every model. Dataset files, waveform caches, and large model checkpoints are not included in the GitHub repository by default.

See [`docs/reproducibility.md`](docs/reproducibility.md) for additional details.

## Limitations and Responsible Use

* Results depend on dataset splits, annotations, preprocessing, and training configurations.
* The internal patient-level split is not the official ICBHI challenge split.
* Robustness findings apply only to the conditions actually evaluated.
* Model outputs should not be interpreted as calibrated clinical probabilities without appropriate validation.
* Interpretability visualizations are not proof of causal explanations.
* Cross-dataset results are limited by the evaluated subset and annotation compatibility.
* This project is intended for research and education, not standalone clinical diagnosis.

## References

1. [ICBHI 2017 Respiratory Sound Database](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)
2. [SPRSound official repository](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound)
3. Zhang et al., *SPRSound: Open-Source SJTU Paediatric Respiratory Sound Database* (2022). [DOI](https://doi.org/10.1109/TBCAS.2022.3204910)

## Acknowledgements

This project builds on publicly available respiratory sound datasets, pretrained Audio Spectrogram Transformer weights, and open-source machine-learning libraries. Please consult the respective dataset licenses, model terms, and library licenses before redistributing data or derived artifacts.
