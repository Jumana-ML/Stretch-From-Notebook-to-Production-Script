# Model Comparison CLI Tool

A production-quality command-line tool to compare multiple machine learning models for churn prediction using stratified cross-validation.

## Installation
1. Ensure you have Python 3.8+ installed.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib joblib
   ```

## Usage
You can run the script from the terminal using `python compare_models.py --data-path data/telecom_churn.csv` followed by the necessary arguments.

### Arguments Explained:
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--data-path` | String | (Required) | The path to your input CSV dataset. |
| `--output-dir` | String | `./output` | The directory where plots, logs, and the best model will be saved. |
| `--n-folds` | Integer | `5` | Number of folds for Stratified Cross-Validation. |
| `--random-seed` | Integer | `42` | Seed used for reproducibility in data splitting and model training. |
| `--dry-run` | Flag | N/A | If included, the script validates the data and config then exits without training. |

## Example Commands

### 1. Normal Run (Full Training)
This command will process the data, perform cross-validation, save visualizations, and export the best model to the default output folder:
```bash
python compare_models.py --data-path data/telecom_churn.csv
```

### 2. Dry Run (Validation Mode)
Use this to quickly check if your dataset path is correct and if the columns match the requirements without waiting for the models to train:
```bash
python compare_models.py --data-path data/telecom_churn.csv --dry-run
```

### 3. Custom Experiment
Run an experiment with 10 folds and a specific random seed, saving results to a custom folder:
```bash
python compare_models.py --data-path data/telecom_churn.csv --n-folds 10 --random-seed 123 --output-dir ./my_experiment_results
```

## Outputs
After a successful run, the `--output-dir` will contain:
* `comparison_table.csv`: Mean metrics for all models.
* `pr_curves.png`: Precision-Recall curves for the top 3 models.
* `calibration.png`: Calibration plots for the top 3 models.
* `best_model.joblib`: The serialized best-performing model.
* `experiment_log.csv`: A timestamped log of the performance results.
```

