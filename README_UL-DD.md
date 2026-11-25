# UL-DD Driver Drowsiness Detection

An offline machine-learning and deep-learning study for classifying driver drowsiness into **Low, Medium, and High** levels using extracted facial, action-unit, and pose features from the UL-DD dataset.

> **Project scope:** This notebook works with pre-extracted feature streams (FL, FAU, and PL). The image-model experiments use colour-encoded feature sequences—not raw camera frames or infrared face images. The current project is an experimental pipeline, not a deployed real-time driver-monitoring system.

## Overview

Driver drowsiness develops through changes in eye activity, facial behaviour, and posture. This project combines several extracted feature modalities and compares conventional machine-learning models with temporal deep-learning architectures.

The pipeline includes:

- Loading and aligning UL-DD feature files and KSS labels
- Deriving facial and posture-related signals
- Handling short missing-data gaps and subject-wise robust normalisation
- Creating labelled, overlapping temporal windows
- Exploratory data analysis and feature inspection
- Training and comparing ML, sequence-DL, and image-based models
- Evaluating models with subject-grouped cross-validation

## Dataset

The experiments use the **UL-DD extracted-feature files and accompanying `Labels.csv`**.

### Feature modalities

| Modality | Representation | Role |
|---|---|---|
| FL | 68 facial landmarks with x/y coordinates (136 values) | Eye, mouth, and facial geometry |
| FAU | 30 facial action-unit values | Facial expression-related activity |
| PL | 33 pose landmarks with x/y/z coordinates (99 values) | Head and posture-related information |

The notebook's executed dataset contains **35 sessions from 19 participants**. The feature streams are sampled at **60 FPS**.

### Labels

KSS ratings are converted into three classes:

- **Low:** KSS < 5
- **Medium:** 5 ≤ KSS < 7
- **High:** KSS ≥ 7

These are interval-level self-reported labels, not frame-by-frame clinical measurements.

## Methodology

### 1. Signal preparation and feature engineering

The FL, FAU, and PL streams are aligned using frame information where available. Derived signals include:

- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Mouth width and brow-gap signals
- Head-orientation proxies
- Shoulder position, width, and tilt
- Relative facial and pose measurements
- Blink count and duration
- Long eye closures and yawn events

Short gaps of up to **0.5 seconds** are interpolated. A validity mask is retained before repair, and subject-wise robust scaling is applied to the continuous signals.

### 2. Windowing

| Setting | Value |
|---|---:|
| Window duration | 10 seconds |
| Window stride | 5 seconds |
| Overlap | 50% |
| Sequence sampling rate | 10 Hz |
| Sequence length | 100 time steps |
| Minimum valid-frame fraction | 80% |

Windows that cross a KSS interval boundary are excluded. Windows with insufficient valid observations are also removed.

### 3. Model inputs

Two feature representations are prepared:

- **Tabular features:** window statistics such as mean, standard deviation, minimum, maximum, and slope, along with binary-flag fractions and event features. Shape: **(16,013, 252)**.
- **Temporal sequences:** ordered signals sampled at 10 Hz. Shape: **(16,013, 100, 51)**.

The processed experiment retained **16,013 windows**:

| Class | Number of windows |
|---|---:|
| Low | 6,233 |
| Medium | 3,964 |
| High | 5,816 |

## Models evaluated

### Classical machine learning

- Majority-class baseline
- Logistic Regression
- RBF Support Vector Machine
- Random Forest
- XGBoost (or HistGradientBoosting fallback, depending on availability)

### Deep learning on temporal features

- ANN / MLP
- 1D-CNN
- Simple RNN
- LSTM
- GRU
- Transformer
- CNN-BiLSTM

### Image-based experiments

- 2D CNN trained from scratch
- VGG16 with frozen ImageNet backbone and a trainable classifier head
- InceptionV3 with frozen ImageNet backbone and a trainable classifier head

For this branch, each numerical sequence is converted into a synthetic colour representation. These are **feature visualisations**, not raw video frames.

## Evaluation

The notebook uses **5-fold StratifiedGroupKFold**, with participant identity as the grouping variable. A participant's windows are kept together in the outer train/test split to reduce subject leakage.

Neural-network validation splits are also group-aware. Training uses class weighting and early stopping.

Reported metrics include:

- Macro-F1 (main comparison metric)
- Accuracy
- Balanced accuracy
- High-class recall
- AUC, where computable

The executed notebook run uses **one seed (0)**. Results are therefore a five-fold comparison for this run, not a multi-seed stability study.

## Results

Mean Macro-F1 across the subject-grouped folds:

| Model | Input | Macro-F1 |
|---|---|---:|
| XGBoost | Tabular | **0.482 ± 0.067** |
| Random Forest | Tabular | 0.480 ± 0.077 |
| RF (paper configuration) | Tabular | 0.461 ± 0.056 |
| GRU | Temporal sequence | **0.420 ± 0.049** |
| Transformer | Temporal sequence | 0.409 ± 0.080 |
| LSTM | Temporal sequence | 0.406 ± 0.043 |
| 1D-CNN | Temporal sequence | 0.398 ± 0.056 |
| Simple RNN | Temporal sequence | 0.397 ± 0.023 |
| CNN-BiLSTM | Temporal sequence | 0.381 ± 0.051 |
| ANN / MLP | Tabular | 0.355 ± 0.044 |
| VGG16 | Synthetic feature-image | 0.378 ± 0.046 |
| InceptionV3 | Synthetic feature-image | 0.368 ± 0.027 |
| 2D CNN | Synthetic feature-image | 0.232 ± 0.108 |

**Observation:** XGBoost and Random Forest performed best overall and were very close. GRU was the strongest temporal deep-learning model. VGG16 performed best among the image-based experiments, but those inputs were synthetic signal-images.

Fold standard deviations overlap for the top classical models; this experiment does not establish a statistically significant difference between XGBoost and Random Forest.

## Running the notebook

The notebook is designed for **Google Colab** and mounts Google Drive.

### 1. Prepare the data

Place the dataset files in Google Drive in the structure expected by the notebook:

```text
MyDrive/
├── Extracted_Features/
│   ├── A_FL_A.csv
│   ├── A_FAU_A.csv
│   ├── A_PL_A.csv
│   ├── A_FL_D.csv
│   ├── A_FAU_D.csv
│   ├── A_PL_D.csv
│   └── ... 
├── Labels.csv
└── Video_Data/                 # referenced in configuration; not used as raw frames
```

The notebook discovers feature files using the naming pattern:

```text
<Subject>_<Modality>_<A-or-D>.csv
```

Here, `A` represents an Alert/Awake session and `D` represents a Drowsy session. Each usable session must have FL, FAU, and PL files and a corresponding label entry.

**Dataset access:** Obtain UL-DD from its official source and follow its terms of use. Do not commit restricted dataset files, private labels, or large data archives to a public GitHub repository.

### 2. Open and run

1. Upload the notebook to Google Colab or open it from your repository.
2. Make sure the UL-DD files and `Labels.csv` are in the expected Google Drive locations.
3. Run the notebook cells in order, beginning with the configuration and data-loading sections.
4. Review the generated EDA, model metrics, and saved outputs.

### 3. Configuration

The main settings are defined near the beginning of the notebook:

```python
QUICK_MODE = False
RUN_IMAGE_MODELS = True
RUN_ABLATION = True

WIN_SEC = 10
STRIDE_SEC = 5
SEQ_HZ = 10
N_FOLDS = 5
SEEDS = [0]
```

The full run includes model training and may take considerable time. Image-model training can be especially resource-intensive; a Colab GPU is recommended for neural-network experiments. `QUICK_MODE` reduces the run size for a faster check, but quick-mode results should not be treated as the final experiment.

## Outputs

The notebook creates cache and result directories under the configured Google Drive root:

```text
MyDrive/
├── cache/
└── results/
```

Generated tables, plots, and model-comparison outputs are saved or displayed by the notebook's respective cells.

## Limitations

- The dataset used in this run contains 19 participants and 35 sessions.
- KSS labels are self-reported and assigned at interval level.
- The reported run uses one random seed.
- No external-dataset validation is reported.
- Subject-wise normalisation may require a suitable calibration approach for a new driver.
- Overlapping windows are correlated, even though participant-level grouping is used for train/test separation.
- The image experiments use synthetic feature-images, not raw video or IR face crops.
- The notebook does not implement live camera inference, real-time alerts, or vehicle integration.

## Repository structure

```text
UL-DD-Driver-Drowsiness/
├── UL_DD_Drowsiness_Pipeline.ipynb
├── README.md
└── results/                    # optional: selected plots/tables only
```

Keep the dataset itself outside the public repository unless its licence explicitly permits redistribution.

## Acknowledgement

This project uses the UL-DD dataset and its extracted FL, FAU, PL features and KSS labels. Please cite the dataset's official publication and follow its attribution and usage requirements when sharing or publishing results.

---

**Project status:** Research / academic experimentation. The reported metrics describe this notebook's offline evaluation and should not be interpreted as a safety-certified driver-monitoring system.
