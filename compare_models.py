import os
import sys
import argparse
import logging
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibrationDisplay
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (PrecisionRecallDisplay, average_precision_score,
                             precision_score, recall_score,
                             f1_score, accuracy_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# set up logging locally
logging.basicConfig(level=logging.DEBUG, # tipe of massages i select all level (debug, info, warning, error, critical).
                                         # when i use debug i can see all massages if i use info i can see all massages except debug.
                     format='%(asctime)s - %(levelname)s - %(message)s', #format of massages the time and level and massages
                     handlers=[ #tell us where to send the messages , hear we send them to terminal
                         logging.StreamHandler(sys.stdout)
                     ])
logger = logging.getLogger(__name__) 

NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents", "contract_months"]
def load_and_preprocess(filepath, random_state=42):
    """Load and validate data from a CSV file."""
    if not os.path.exists(filepath):
        logger.error(f"Data file not found: {filepath}")
        sys.exit(1) # exit the program because the file is not found we can't run the program without data file.
        
    df = pd.read_csv(filepath)
    logger.info(f"Available columns: {df.columns.tolist()}")    
    
    # why we use normalize=True??
    #          because we want to see the distribution as a percentage to balance the data before training
    #          it is important to (dry run) balance the data before training.
    churn_counts = df["churned"].value_counts(normalize=True)
    logger.info(f"Viewing Class Distribution Churn:\n{churn_counts}")

    num_df = df[NUMERIC_FEATURES]
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        num_df, y, test_size=0.2, random_state=random_state, stratify=y # stratify=y to balance the data
    )
    return X_train, X_test, y_train, y_test

def define_models(random_seed=42):
    """Define a dictionary of machine learning models."""
    logger.info(f"define models with random_seed={random_seed}") # to check the random_seed in the terminal
    pipelines = {
        "Dummy": Pipeline(steps=[("scaler", "passthrough"), ("estimator", DummyClassifier(random_state=random_seed))]),
        "LR_default": Pipeline(steps=[("scaler", StandardScaler()), ("estimator", LogisticRegression(max_iter=1000, random_state=random_seed))]),
        "LR_balanced": Pipeline(steps=[("scaler", StandardScaler()), ("estimator", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_seed))]),
        "DT_depth5": Pipeline(steps=[("scaler", "passthrough"), ("estimator", DecisionTreeClassifier(max_depth=5, random_state=random_seed))]),
        "RF_default": Pipeline(steps=[("scaler", "passthrough"), ("estimator", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_seed))]),
        "RF_balanced": Pipeline(steps=[("scaler", "passthrough"), ("estimator", RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=random_seed))])
    }
    return pipelines

def run_cv_comparison(models, X, y, n_splits=5, random_state=42):
    """Run 5-fold stratified cross-validation on all models"""
    logger.info(f" start Cross-validation (Folds={n_splits})") #check the number of folds
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    cv_results = [] # to store the results in a list
    
    for name, model in models.items():
        logger.debug(f"Running cross-validation for model name is: {name}")# to check the name of the model
        metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "pr_auc": []}# to store the metrics

        for train_index, val_index in skf.split(X, y):
            X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index] 
            y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]

            model.fit(X_train_cv, y_train_cv)
            y_pred = model.predict(X_val_cv)
            y_proba = model.predict_proba(X_val_cv)[:, 1]

            metrics["accuracy"].append(accuracy_score(y_val_cv, y_pred))
            metrics["precision"].append(precision_score(y_val_cv, y_pred, zero_division=0))
            metrics["recall"].append(recall_score(y_val_cv, y_pred, zero_division=0))
            metrics["f1"].append(f1_score(y_val_cv, y_pred, zero_division=0))
            metrics["pr_auc"].append(average_precision_score(y_val_cv, y_proba))

        cv_results.append({
            "model": name,
            "accuracy_mean": np.mean(metrics["accuracy"]),
            "f1_mean": np.mean(metrics["f1"]),
            "pr_auc_mean": np.mean(metrics["pr_auc"]),
            "pr_auc_std": np.std(metrics["pr_auc"])
        })
    return pd.DataFrame(cv_results)

def save_comparison_table(results_df, output_path):
    """Save the comparison table to a CSV file."""
    results_df.to_csv(output_path, index=False)
    logger.info(f"saved comparison table to: {output_path}")# to check if the table is saved and where

def plot_pr_curves_top3(models, X_test, y_test, output_path):
    """draw PR Curves for the top 3 models """
    logger.info("drawing PR Curves for the best 3 models")# to check if the PR Curves are drawn
    test_scores = []
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        score = average_precision_score(y_test, y_proba)
        test_scores.append((name, score))
    
    test_scores.sort(key=lambda x: x[1], reverse=True)
    top_3_names = [x[0] for x in test_scores[:3]]

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in top_3_names:
        PrecisionRecallDisplay.from_estimator(models[name], X_test, y_test, ax=ax, name=name)
    ax.set_title("PR Curves for Top 3 Models")
    plt.savefig(output_path)
    plt.close()
    logger.info(f"saved PR Curves to: {output_path}")# to check if the PR Curves are saved and where

def plot_calibration_top3(models, X_test, y_test, output_path):
    """draw Calibration Plots for the top 3 models ans save it to a PNG file."""
    # check if the Calibration Plots are drawn
    logger.info("the calibration plots are drawn")
    plt.savefig(output_path)
    plt.close()
    logger.info(f"save calibration plots to: {output_path}")

def save_best_model(best_model, output_path):
    """Save the best model to a pickle file."""
    dump(best_model, output_path)
    logger.info(f"saved best model to: {output_path}")

def log_experiment(results_df, output_path):
    """log the experiment results to a CSV file with a timestamp."""
    log_df = results_df.copy()
    log_df["timestamp"] = datetime.now().isoformat()
    log_df.to_csv(output_path, index=False)
    logger.info(f"saved experiment results to: {output_path}")

def find_tree_vs_linear_disagreement(rf_model, lr_model, X_test, y_test, feature_names):
    """Find the sample with the highest disagreement between RF and LR predictions."""
    logger.info("start find_tree_vs_linear_disagreement")#check if the function is running
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    prob_diff = np.abs(rf_proba - lr_proba)
    max_idx = np.argmax(prob_diff)
    
    return {
        "sample_idx": max_idx,
        "feature_values": X_test.iloc[max_idx].to_dict(),
        "rf_proba": float(rf_proba[max_idx]),
        "lr_proba": float(lr_proba[max_idx]),
        "prob_diff": prob_diff[max_idx],
        "true_label": int(y_test.iloc[max_idx])
    }

# --- Main Function using CLI , Logger and Argparse---

def main():
    parser = argparse.ArgumentParser(description="classification model comparison(Production CLI)")
    
    # identify the Arguments
    parser.add_argument('--data-path', required=True, help='Path to the input CSV dataset')
    parser.add_argument('--output-dir', default='./output', help='Directory to save results')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dry-run', action='store_true', help='Validate data and config without training')

    args = parser.parse_args()

    # 1. Task1: create output directory to save results
    if not args.dry_run: #check if we are in dry run mode
        os.makedirs(args.output_dir, exist_ok=True) # create the output directory and if it already exists, ignore it

    # 2. loading data and preprocessing
    X_train, X_test, y_train, y_test = load_and_preprocess(args.data_path, args.random_seed)

    # 3. Dry Run
    if args.dry_run:
        # The "Test Run" option; if you do this,
        # the code will stop here after loading the data just to make sure
        # that everything works without going through the long training process becuase it is expensive and takes a long time and space
        logger.info("--- DRY RUN MODE ---")
        logger.info(f"Configuration: Folds={args.n_folds}, Seed={args.random_seed}, Output={args.output_dir}")
        logger.info("Validation successful. Exiting without training.")
        sys.exit(0) # exit the program because we are in dry run mode

    # 4.Define Models(Task 2)
    models = define_models(args.random_seed)

    # 5. Comparison the models using CV (Task 3 & 4)
    results_df = run_cv_comparison(models, X_train, y_train, n_splits=args.n_folds, random_state=args.random_seed)
    save_comparison_table(results_df, os.path.join(args.output_dir, "comparison_table.csv"))

    # 6. training the models for drewing
    fitted_models = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline

    # 7. Visualize the results(Task 5 & 6)
    plot_pr_curves_top3(fitted_models, X_test, y_test, os.path.join(args.output_dir, "pr_curves.png"))
    plot_calibration_top3(fitted_models, X_test, y_test, os.path.join(args.output_dir, "calibration.png"))

    # 8. save the best model and log the experiment(Task 7 & 8)
    best_name = results_df.sort_values("pr_auc_mean", ascending=False).iloc[0]["model"]
    save_best_model(fitted_models[best_name], os.path.join(args.output_dir, "best_model.joblib"))
    log_experiment(results_df, os.path.join(args.output_dir, "experiment_log.csv"))

    # 9. disagreement(Task 9)
    disagreement = find_tree_vs_linear_disagreement(fitted_models["RF_default"], fitted_models["LR_default"], X_test, y_test, NUMERIC_FEATURES)
    logger.info(f"most disagreeing sample is {disagreement['sample_idx']} with difference {disagreement['prob_diff']:.2f}")
    
    logger.info("--- End of Experiment all tasks ---")#check if we finished all the tasks

if __name__ == "__main__":
    main()