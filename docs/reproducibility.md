# RobustLungAI — Reproducibility Guide

## 1. Purpose

This guide describes the project structure, datasets, experiment workflow, model checkpoints, evaluation metrics, and practical steps for investigating the recorded RobustLungAI experiments.

The experiments were developed primarily in Kaggle notebooks. The Gradio application was developed and tested separately.

Reproducing a result may require the original datasets, prepared manifests, waveform caches, model checkpoints, software versions, random seeds, and experiment-specific configuration.

## 2. Repository Layout

| Path                                              | Purpose                                      |
| ------------------------------------------------- | -------------------------------------------- |
| `notebooks/01-data-preparation.ipynb`             | Prepare respiratory-cycle data and manifests |
| `notebooks/02-baseline-models.ipynb`              | Evaluate baseline models                     |
| `notebooks/03a-ast-finetuning-ce.ipynb`           | AST with cross-entropy                       |
| `notebooks/03b-ast-focal-loss.ipynb`              | AST with focal loss                          |
| `notebooks/03c-ast-subcon.ipynb`                  | Supervised contrastive learning experiment   |
| `notebooks/03d-ast-augmentation-ablation.ipynb`   | SpecAugment and Mixup experiments            |
| `notebooks/04a-cross-device-generalization.ipynb` | Recording-device evaluation                  |
| `notebooks/04b-noise-robustness-sweep.ipynb`      | Added-noise evaluation                       |
| `notebooks/04c-cross-dataset-transfer.ipynb`      | SPRSound evaluation                          |
| `notebooks/04d-noise-augmented-mitigation.ipynb`  | Noise-augmented training evaluation          |
| `notebooks/04-final-results.ipynb`                | Aggregate recorded experiment results        |
| `notebooks/05a-interpretability.ipynb`            | Attention and Grad-CAM visualizations        |
| `notebooks/05b_gradio_demo.ipynb`                 | Demo notebook                                |
| `app/gradio_app.py`                               | Gradio application source                    |
| `docs/model_card.md`                              | Model overview and limitations               |
| `results/tables/results_phase3_combined.csv`      | Consolidated experiment results              |

Notebook filenames and references should match the actual repository. Some experiments may depend on outputs created by earlier notebooks.

## 3. Datasets

### 3.1 ICBHI 2017

The primary dataset is the [ICBHI 2017 Respiratory Sound Database](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge).

The project classifies annotated respiratory cycles into:

* Normal
* Crackle
* Wheeze
* Both crackle and wheeze

The internal experiments use a patient-level 70/15/15 training, validation, and test split. This split is not the official ICBHI challenge split.

Obtain the dataset through its official distribution channel and follow the applicable access and use terms. The dataset is not included in this repository.

### 3.2 SPRSound

The cross-dataset experiment uses a selected subset of the [SPRSound repository](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound).

The selected subset contains 355 WAV recordings and 355 corresponding JSON annotation files from the inter-subject test set.

Reference: Zhang et al., *SPRSound: Open-Source SJTU Paediatric Respiratory Sound Database*, 2022. [DOI](https://doi.org/10.1109/TBCAS.2022.3204910)

Verify audio-to-annotation pairing, label mapping, preprocessing, and the subset definition before evaluating. The SPRSound annotation categories must not be assumed to map one-to-one to the ICBHI classes.

Follow the current dataset repository's licensing and attribution requirements.

## 4. Environment and Dependencies

The research notebooks use Python and libraries such as:

* PyTorch
* Hugging Face Transformers
* NumPy and pandas
* librosa
* scikit-learn
* Matplotlib
* Jupyter

The Gradio application additionally uses Gradio and the Hugging Face Hub client.

Install dependencies using the repository's `requirements.txt` after checking that its versions are compatible with the application and the notebook you intend to run.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate the environment using `.venv\Scripts\activate` instead.

The original research environment was primarily Kaggle. These commands create a local environment but do not guarantee that all Kaggle-specific notebooks will run unchanged.

For a more exact reproduction, record the Python version, package versions, accelerator, random seeds, and experiment configuration from the original run.

## 5. Preprocessing

The documented AST configuration includes:

* Sampling rate: 16 kHz.
* Target cycle duration: 8 seconds.
* Mel representation: 128 Mel bins.
* Feature extraction: AST-compatible log-Mel preprocessing.

The precise preprocessing settings can vary between experiments and must match the corresponding checkpoint.

Reproduction requires consistent audio segmentation, annotation parsing, patient-level splits, label mapping, filtering, and feature extraction.

Some notebooks reuse existing waveform caches. Verify the configured paths and expected filenames before running them. A missing cache should not silently be replaced by a newly generated cache if exact reuse is required.

## 6. Training Workflow

Follow the notebooks in the order required by the experiment:

1. Prepare the data and annotated respiratory-cycle manifest.
2. Establish baseline performance.
3. Fine-tune AST with cross-entropy.
4. Evaluate focal loss and supervised contrastive learning.
5. Run the augmentation ablation experiments.
6. Evaluate device generalization and added-noise robustness.
7. Evaluate cross-dataset transfer and noise-augmented mitigation.
8. Aggregate the recorded results.
9. Generate interpretability visualizations.
10. Run the Gradio application.

Not every experiment requires all preceding notebooks. Check its input paths and dependencies before execution.

For faithful reproduction, record the relevant random seeds, optimizer, learning rate, batch size, epoch count, loss parameters, augmentation settings, checkpoint-selection criteria, and evaluation protocol.

## 7. Evaluation Metrics

The project reports sensitivity, specificity, and ICBHI score, along with other classification metrics where available.

For the binary ICBHI score, crackle, wheeze, and both are grouped into the abnormal category:

**ICBHI score = (abnormal sensitivity + normal specificity) / 2**

The reported score is expressed as a percentage. Validation and test metrics must remain separately identified.

The internal patient-level split must not be confused with the official ICBHI challenge evaluation protocol.

## 8. Consolidated Results

The consolidated results file is:

`results/tables/results_phase3_combined.csv`

It aggregates recorded outputs from the baseline, loss-function, and augmentation experiments. It does not retrain the models.

To regenerate it:

1. Make the source experiment CSV files available.
2. Inspect the paths expected by `04-final-results.ipynb`.
3. Run the cells that load and combine the results.
4. Confirm that all expected experiment groups are present.
5. Compare the exported values with their source CSV files.
6. Verify units, split labels, metric definitions, and experiment identifiers.

The consolidated results include a recorded Random Forest test-score discrepancy. Check the original sensitivity, specificity, and metric calculation before changing the value.

Phase 4 robustness and cross-dataset results should be reported separately unless the relevant experiment outputs have explicitly been added to the consolidated table.

## 9. Running the Gradio Application

Application source: `app/gradio_app.py`

Demo notebook: `notebooks/05b_gradio_demo.ipynb`

The application uses the fine-tuned checkpoint configured in its source code. When using Hugging Face Hub retrieval, verify the repository ID, checkpoint filename, and checkpoint-loading procedure.

To launch the app from a compatible local environment:

```bash
python app/gradio_app.py
```

This command assumes the script supports direct execution from the repository root and that all required dependencies are installed. If your current script is designed to run in a notebook, follow the notebook's launch instructions instead.

A public Gradio share link may be temporary and can stop working when the hosting session ends. The public link, if available, is listed in the root README.

## 10. Checkpoints and Large Files

The repository does not include datasets, waveform caches, or large model checkpoints by default.

The fine-tuned checkpoint is hosted at:

[RobustLungAI on Hugging Face](https://huggingface.co/praveshsubba/robustlungai-ast)

Check the repository for the exact file available for download. The checkpoint used for an experiment or the live application should be identified explicitly rather than inferred from its filename.

## 11. Limitations

* Kaggle-specific paths may need adjustment in other environments.
* Results depend on data splits, preprocessing, annotations, and checkpoint selection.
* Reproducing results may require unavailable intermediate artifacts.
* The internal split is not the official ICBHI challenge split.
* Noise and device experiments apply only to tested conditions.
* Cross-dataset evaluation uses a selected SPRSound subset and a specific label mapping.
* These research results do not establish clinical validity.

## 12. References

1. [ICBHI 2017 Respiratory Sound Database](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)
2. [SPRSound official repository](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound)
3. Zhang et al., *SPRSound: Open-Source SJTU Paediatric Respiratory Sound Database*. [DOI](https://doi.org/10.1109/TBCAS.2022.3204910)
