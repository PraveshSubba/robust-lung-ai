# RobustLungAI — Reproducibility Guide

## 1. Purpose

This guide describes the project structure, experiment workflow, data requirements, model configuration, evaluation protocol, and limitations that affect reproduction of RobustLungAI.

The project investigates respiratory sound classification using Audio Spectrogram Transformers (AST), with experiments covering baselines, loss functions, augmentation, noise robustness, recording-device generalization, cross-dataset transfer, and interpretability.

The experiments were developed primarily in Kaggle notebooks. The interactive Gradio demo is designed to run from a Google Colab notebook.

Reproduction requires access to the relevant datasets and, depending on the experiment, prepared manifests, model checkpoints, and cached audio features. These resources are not all included in the GitHub repository.

## 2. Repository Structure

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

This reflects the intended current repository layout. Empty directories and local-only files may not appear in Git.

## 3. Dataset Requirements

### 3.1 ICBHI 2017 Respiratory Sound Database

The primary dataset is the [ICBHI 2017 Respiratory Sound Database](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge).

The data-preparation notebook handles the relevant audio preparation and respiratory-cycle annotations. The exact dataset paths are environment-specific and must be updated for the machine or notebook environment used.

The project uses four target classes:

* `normal`
* `crackle`
* `wheeze`
* `both`

The four-class mapping is derived from the crackle and wheeze annotations. Verify that the annotation-to-class mapping in the experiment matches the mapping used for evaluation.

### 3.2 SPRSound subset

Cross-dataset evaluation uses a selected SPRSound subset containing 355 WAV recordings and 355 corresponding JSON annotation files.

Source: [SPRSound official repository](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound)

Publication: Zhang et al., *SPRSound: Open-Source SJTU Paediatric Respiratory Sound Database* (2022), [DOI](https://doi.org/10.1109/TBCAS.2022.3204910).

Reproduction requires the same subset selection, annotation interpretation, class mapping, and preprocessing used by the corresponding experiment. Different subset selection or annotation mapping may change the results.

Respect the source dataset's licensing, attribution, and redistribution terms. The GitHub repository does not include the original recordings by default.

## 4. Environment and Dependencies

The experiments use Python and a combination of:

* PyTorch
* Hugging Face Transformers
* Hugging Face Hub
* NumPy
* SciPy
* Librosa
* Matplotlib
* Pandas
* Gradio

The root `requirements.txt` records project dependencies. Check the installed versions against that file and the relevant notebook before running an experiment.

For the Colab demo, run the dependency-installation cell in `05b_gradio_demo.ipynb` before importing the application modules.

For notebook-based training, use a compatible Python environment with the required PyTorch and Transformers versions. GPU availability, memory, dependency versions, and library defaults can affect execution and numerical results.

The exact versions used for each original training run should be recorded from the corresponding notebook if they are available. Do not assume that the current Colab environment is identical to the original Kaggle environment.

## 5. Model Configuration

The project uses the pretrained checkpoint:

`MIT/ast-finetuned-audioset-10-10-0.4593`

The documented preprocessing configuration includes:

| Parameter                | Documented configuration                                    |
| ------------------------ | ----------------------------------------------------------- |
| Audio sampling rate      | 16,000 Hz                                                   |
| Target cycle duration    | 8 seconds                                                   |
| Mel frequency bins       | 128                                                         |
| Bandpass filter          | Nominally 50–8,000 Hz, constrained by the Nyquist frequency |
| Number of output classes | 4                                                           |
| Output labels            | Normal, crackle, wheeze, both                               |

These are project-level documented settings. Confirm the exact feature extractor, normalization, tensor dimensions, padding or truncation policy, and model-head configuration in the notebook and checkpoint-loading code.

Do not assume that every experiment uses identical augmentation, loss functions, or training parameters.

## 6. Notebook Workflow

Run the notebooks in the following conceptual order.

| Notebook                                | Purpose                                                      |
| --------------------------------------- | ------------------------------------------------------------ |
| `01-data-preparation.ipynb`             | Prepare audio segments, annotations, manifests, and features |
| `02-baseline-models.ipynb`              | Establish traditional machine-learning and CNN baselines     |
| `03a-ast-finetuning-ce.ipynb`           | Fine-tune AST using cross-entropy loss                       |
| `03b-ast-focal-loss.ipynb`              | Investigate focal loss                                       |
| `03c-ast-subcon.ipynb`                  | Investigate supervised contrastive learning                  |
| `03d-ast-augmentation-ablation.ipynb`   | Compare augmentation configurations                          |
| `04a-cross-device-generalization.ipynb` | Evaluate recording-device differences                        |
| `04b-noise-robustness-sweep.ipynb`      | Evaluate performance under added noise                       |
| `04c-cross-dataset-transfer.ipynb`      | Evaluate transfer to the selected SPRSound subset            |
| `04d-noise-augmented-mitigation.ipynb`  | Investigate noise-augmented training                         |
| `04-final-results.ipynb`                | Aggregate recorded experimental results                      |
| `05a-interpretability.ipynb`            | Generate and inspect interpretability visualizations         |
| `05b_gradio_demo.ipynb`                 | Launch the interactive application in Google Colab           |

Some experiments depend on artifacts generated by earlier notebooks. If a notebook expects a manifest, cached waveform, feature array, or checkpoint, generate or obtain that artifact using the relevant preparation or training workflow first.

The final-results notebook aggregates recorded outputs; it does not necessarily retrain every model.

## 7. Audio Preparation and Caching

The documented preprocessing pipeline includes audio loading, resampling to 16 kHz, bandpass filtering, cycle extraction, and preparation for AST input.

The target cycle duration is documented as 8 seconds. Shorter or longer inputs must be handled according to the actual notebook's padding, truncation, and segmentation logic.

The original workflow uses prepared manifests and cached waveform or feature artifacts to avoid repeating expensive preprocessing.

Cached artifacts are environment-specific. Their filenames and paths must match the manifest and the preprocessing code. A cached file should not be assumed valid solely because its filename exists.

When reproducing experiments, record:

* Source dataset version and location.
* Patient-level train, validation, and test assignments.
* Audio preprocessing parameters.
* Annotation-to-class mapping.
* Cache-generation configuration.
* Checkpoint path and model configuration.

If you intend to evaluate an existing cache without regenerating it, first verify that all expected cache files exist and correspond to the correct recording and cycle. Do not silently substitute missing files or mix artifacts from different splits.

## 8. Training Experiments

The notebooks investigate different training strategies, including cross-entropy, focal loss, supervised contrastive learning, SpecAugment, and Mixup.

For a reproducible training run, record the following from the notebook:

* Model and checkpoint initialization.
* Patient-level split assignments.
* Random seed, where configured.
* Optimizer and learning rate.
* Batch size and number of epochs.
* Loss-function settings.
* Augmentation settings.
* Checkpoint-selection criterion.
* Validation and test procedures.
* Library versions and compute environment.

The exact settings may differ between experiment groups. Consult the original notebook instead of inferring missing hyperparameters from the final scores.

## 9. Evaluation Protocol and Metrics

The reported internal experiments use a patient-level 70/15/15 train/validation/test split.

This is an internal project split and is not the official ICBHI challenge split. Direct comparisons with published challenge results require matching the evaluation protocol and metric implementation.

The project reports:

* **Sensitivity:** Abnormal-class sensitivity for the binary normal-versus-abnormal metric.
* **Specificity:** Normal-class specificity.
* **ICBHI score:** The mean of sensitivity and specificity.

For a consistent percentage-based calculation:

$$
\text{ICBHI score} =
\frac{\text{Sensitivity}+\text{Specificity}}{2}
$$

Confirm the actual implementation in the evaluation notebook before treating this expression as the source of every reported value.

### Reported baseline and loss-function results

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

* The Random Forest test sensitivity and specificity average to approximately 55.12%, rather than the reported 55.05%. Verify the original experiment output and metric implementation before resolving this discrepancy.

### Reported augmentation ablation

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

These tables represent recorded project results and are not independently verified in this guide. The loss-function and augmentation experiments are separate groups; compare them only after checking their respective protocols and configurations.

The consolidated Phase 3 table is located at [`results/tables/results_phase3_combined.csv`](../results/tables/results_phase3_combined.csv).

## 10. Robustness Experiments

### Added-noise robustness

The noise robustness notebook evaluates the model under selected signal-to-noise ratios. Reproduction requires matching the original noise-generation procedure, SNR definition, randomization, audio normalization, and evaluation data.

The notebook's recorded clean score and noisy scores should be treated as experiment-specific results. Noise results should not be generalized to all real-world recording conditions.

### Cross-device generalization

The cross-device notebook investigates differences across recording-device or acquisition-condition groups. Reproduction requires the same device labels, group definitions, test recordings, and preprocessing.

### Cross-dataset transfer

The cross-dataset notebook uses the selected SPRSound subset. Results depend on annotation mapping, the exact files selected, preprocessing, and the evaluation protocol. They should not be interpreted as broad external clinical validation.

### Noise-augmented mitigation

The mitigation notebook investigates whether training with noise augmentation changes performance under noisy evaluation conditions. Any comparison should use consistent evaluation data and clearly documented training and test configurations.

## 11. Interpretability

The interpretability notebook investigates attention rollout and Grad-CAM-related visualizations. Generated figures include examples of correct predictions, prediction errors, noise conditions, frequency profiles, and randomization analysis.

To reproduce these figures, use the relevant trained checkpoint, input audio, preprocessing pipeline, and visualization settings. The same audio and checkpoint are important when comparing visualizations across runs.

Interpretability visualizations are exploratory and do not prove causal reasoning or clinical validity.

## 12. Running the Gradio Demo

The demo is run from:

[`notebooks/05b_gradio_demo.ipynb`](../notebooks/05b_gradio_demo.ipynb)

### Procedure

1. Open the notebook in Google Colab.
2. Run the dependency-installation cell.
3. Run the imports and model-initialization cells.
4. Allow the notebook to download the checkpoint from the [RobustLungAI Hugging Face repository](https://huggingface.co/praveshsubba/robustlungai-ast).
5. Run the remaining cells in order.
6. Execute the final Gradio launch cell and open the URL provided by the notebook.

The notebook uses the fine-tuned checkpoint `ast_lung_fp16.pt`, subject to the actual repository contents and the loading code.

No permanent hosted demo is provided. A temporary share URL may stop working when the Colab runtime ends.

The demo is for research and educational exploration, not clinical diagnosis.

## 13. Known Reproducibility Limitations

* The original Kaggle and current Colab environments may differ.
* Dataset paths and cache locations are environment-specific.
* Original audio recordings and large checkpoints are not duplicated in the GitHub repository by default.
* Missing manifests, caches, or checkpoints can prevent a notebook from running directly.
* Random seeds, hardware, library versions, and training nondeterminism may affect results.
* The Random Forest test metric discrepancy remains unresolved.
* The internal patient-level split is not the official ICBHI challenge split.
* Cross-dataset findings apply to the selected SPRSound subset and its annotation mapping.
* Interpretability figures are not evidence of causal explanations or clinical validity.

## 14. Recommended Reproduction Record

For each new run, record:

| Item          | Record                                                    |
| ------------- | --------------------------------------------------------- |
| Notebook      | Exact notebook filename                                   |
| Dataset       | Source, version, and subset                               |
| Split         | Patient-level assignments and split protocol              |
| Preprocessing | Sampling rate, filtering, segmentation, and normalization |
| Model         | Architecture and initialization checkpoint                |
| Training      | Loss, optimizer, learning rate, batch size, and epochs    |
| Environment   | Python and library versions, hardware                     |
| Evaluation    | Test data, metrics, and metric implementation             |
| Artifacts     | Checkpoint filename and results file                      |
| Deviations    | Any changes from the original notebook                    |

## 15. Responsible Use

RobustLungAI is a research and educational project. It has not been clinically validated and must not be used as the sole basis for diagnosis, treatment, or other medical decisions.

### References

* [ICBHI 2017 Respiratory Sound Database](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)
* [SPRSound official repository](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound)
* [SPRSound publication](https://doi.org/10.1109/TBCAS.2022.3204910)
* [RobustLungAI fine-tuned checkpoint](https://huggingface.co/praveshsubba/robustlungai-ast)
* [AST base checkpoint](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
