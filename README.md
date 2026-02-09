# git

A machine learning and deep learning project for diagnosing compressor faults using simulated refrigerator operational data.

## Project Overview

This project implements multiple machine learning and deep learning models to predict compressor faults based on sensor data from a refrigeration system. The dataset contains various operational parameters such as temperature, pressure, and power consumption measurements.

## Dataset

- **Source**: Simulated Refrigerator Fault Diagnosis Dataset from Kaggle
- **Raw Dataset Location (local after download)**: `Data/datasets/samoilovmikhail/`
- **Processed Data**: `Data/processed/`
  - `train_data.csv` - Training dataset
  - `val_data.csv` - Validation dataset

Note: The raw Kaggle dataset is large and should NOT be committed to this repository. See the Installation section for instructions to download the dataset locally using the included script or the Kaggle CLI.

### Features Used

- T_amb (Ambient Temperature)
- T_evap_sat (Evaporator Saturation Temperature)
- P_dis_bar (Discharge Pressure)
- P_suc_bar (Suction Pressure)
- P_comp_W (Compressor Power)
- N_comp_Hz (Compressor Speed)
- delta_cond_evap (Condenser-Evaporator Delta)
- cooling_efficiency (Cooling Efficiency)
- P_dis_bar_rolling_mean (Rolling Mean of Discharge Pressure)
- P_suc_bar_rolling_std (Rolling Std of Suction Pressure)
- COP_diff (COP Difference)
- N_comp_Hz_diff (Compressor Speed Difference)
- door_open (Door Open Flag)
- frost_level (Frost Level)

## Project Structure

```
Predictive_maintaince/
├── Data/
│   ├── datasets/          # Raw datasets
│   ├── processed/         # Processed training and validation data
│   └── mlflow.db          # MLflow tracking database
├── NoteBooks/             # Jupyter notebooks
│   ├── EDA..ipynb         # Exploratory Data Analysis
│   ├── FeatureEngineering.ipynb  # Feature engineering
│   ├── ML-model.ipynb     # Machine learning models (CatBoost, Random Forest)
│   ├── DL-model.ipynb     # Deep learning models (MLP, LSTM)
│   └── model-kfold.ipynb  # K-Fold cross-validation approach
├── Model/                 # Trained models
│   ├── best_lstm.pt       # Best LSTM model (PyTorch)
│   └── best_mlp.pt        # Best MLP model (PyTorch)
├── Plots/                 # Generated visualizations
│   ├── EDA-Plots/
│   ├── DL/
│   └── Base-Models-Benchmarking/
├── Script/                # Python scripts
│   └── kaggle_import.py   # Script to download dataset from Kaggle
├── mlartifacts/           # MLflow artifacts and model storage
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## Models Implemented

### Machine Learning Models
- **CatBoost Classifier** - Gradient boosting with K-Fold cross-validation
- **Random Forest Classifier** - Ensemble learning approach

### Deep Learning Models
- **MLP (Multi-Layer Perceptron)** - Feed-forward neural network
- **LSTM (Long Short-Term Memory)** - Recurrent neural network for sequential data

## Methodology

### Data Preprocessing
1. Window-based feature extraction (window_size=60, stride=15)
2. Standardization using StandardScaler
3. Train-validation split

### Model Training
- **MLflow Tracking** for experiment management and artifact storage
- **GPU Acceleration** for CatBoost training

### Evaluation Metrics
- Accuracy
- Precision (macro-averaged)
- Recall (macro-averaged)
- F1-Score (macro-averaged)
- AUC-ROC (One-vs-Rest)
- Confusion Matrix

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for faster training)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Predictive_maintaince
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset (do not commit raw data to GitHub):

Option A — use the provided script (recommended):
```bash
# set an optional cache location and run the helper script
python Script/kaggle_import.py
```

Option B — using the Kaggle CLI (alternative):
1. Install and configure the Kaggle CLI (set `KAGGLE_USERNAME` and `KAGGLE_KEY` in your environment, or place `kaggle.json` under `~/.kaggle/`).
2. Download the dataset:
```bash
kaggle datasets download -d samoilovmikhail/simulated-refrigerator-fault-diagnosis-dataset -p Data/datasets/samoilovmikhail --unzip
```

After downloading, run any preprocessing or the notebooks to generate `Data/processed/` files.

## Usage

### Running Notebooks

1. Start Jupyter:
```bash
jupyter notebook
```

2. Navigate to the `NoteBooks/` directory and open the desired notebook:
   - Start with `EDA..ipynb` for data exploration
   - Continue with `FeatureEngineering.ipynb` for feature creation
   - Run `ML-model.ipynb` for traditional ML approaches
   - Run `DL-model.ipynb` for deep learning approaches
   - Run `model-kfold.ipynb` for K-Fold validation results

### Training Models

Models can be trained directly through the Jupyter notebooks. All training runs are logged to MLflow for tracking and comparison.

### Viewing MLflow Results

```bash
mlflow ui --backend-store-uri sqlite:///Data/mlflow.db
```

Then navigate to `http://localhost:5000` in your browser.

## Key Results

### CatBoost K-Fold Performance
- Mean Accuracy: ~0.95
- Mean F1-Score: ~0.94
- Mean AUC: ~0.98

### Deep Learning Performance
- MLP and LSTM models trained on windowed time-series data
- Results tracked and compared in MLflow

## Experiment Tracking

This project uses MLflow for:
- **Experiment Management**: Version control for different model configurations
- **Parameter Logging**: Hyperparameters for each run
- **Metric Tracking**: Performance metrics across runs
- **Artifact Storage**: Model checkpoints and visualizations

Experiments are stored in:
- Tracking URI: `sqlite:///Data/mlflow.db`
- Artifact Location: `file:///mlartifacts`

## Large Files & Artifacts

- MLflow artifacts, model checkpoints (`Model/*.pt`) and raw datasets are intentionally not tracked in this Git repository because they are large.
- To reproduce artifacts and trained models: run the notebooks (`NoteBooks/`) that perform training (for example `model-kfold.ipynb`, `DL-model.ipynb`). These notebooks will log results and artifacts to the MLflow artifact store configured in the notebook (see `TRACKING_DB` and `ARTIFACT_ROOT` variables at the top of each notebook).
- To view runs locally, start the MLflow UI against the tracking DB used in the notebooks (example above).

If you need to share models or artifacts, use a dedicated artifact store (S3, Azure Blob, or share the `mlartifacts/` directory separately) rather than committing to GitHub.

## Recommended .gitignore entries

Add the following to `.gitignore` to avoid committing large files and local artifacts:

```
# Raw datasets and caches
Data/datasets/

# MLflow artifacts and runs
mlartifacts/
mlruns/

# Trained model weights
Model/*.pt

# Jupyter checkpoints
.ipynb_checkpoints/

# Python virtualenv
venv/
```

## Dependencies

See `requirements.txt` for a complete list. Key libraries include:
- **MLflow**: Experiment tracking and model management
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Traditional machine learning algorithms
- **PyTorch**: Deep learning framework
- **CatBoost**: Gradient boosting library
- **Matplotlib & Seaborn**: Data visualization
- **Plotly**: Interactive visualizations

## Future Improvements

- [ ] Hyperparameter optimization using Optuna or Hyperopt
- [ ] Ensemble methods combining multiple models
- [ ] Real-time fault detection pipeline
- [ ] Model deployment as REST API
- [ ] Anomaly detection for detecting unknown fault types
- [ ] Attention mechanisms for improved LSTM performance

## License

MIT

## Contact

For questions or contributions, please contact `rajasbhingarde18@gmail.com`

## Acknowledgments

- Dataset source: [Samoilov Mikhail](https://www.kaggle.com/samoilovmikhail) on Kaggle
- Built with MLflow for comprehensive experiment tracking
