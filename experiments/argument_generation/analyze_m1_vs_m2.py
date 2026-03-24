"""
M1 vs M2 Summary Analysis for Large-Scale Explainability Experiment

Compares M1 (presence-based, topic model baseline) vs M2 (full LASSO with sequential features)
across 3 synthesis types (strict, faithful, restructured) with ~5000 arguments each (15,000 total).

Expected Data Structure:
    experiments/argument_generation/argument_data/
    ├── synthesis_strict/
    │   └── pairwise_comparisons_bt_scores.csv  (5000 rows)
    ├── synthesis_faithful/
    │   └── pairwise_comparisons_bt_scores.csv  (5000 rows)
    └── synthesis_restructured/
        └── pairwise_comparisons_bt_scores.csv  (5000 rows)

Key Insight:
- M1 = Topic Model Baseline: Presence-based features represent what a bag-of-words topic model
  could capture - it knows *which* structures/contents were used but not *when* or *in what order*.
- M2 = Sequential Model: Position + interactions + chains capture the temporal/sequential aspects
  that topic models miss.

Usage:
    python experiments/argument_generation/analyze_m1_vs_m2.py

Output:
    All figures are saved to experiments/argument_generation/figures/.
"""

import argparse
import itertools
import json
import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Section 1: Setup & Constants
# =============================================================================

# =============================================================================
# RESPONSE VARIABLE SELECTION
# Set to "bt_score" to use Bradley-Terry scores (original)
# Set to "rank_score" to use normalized rank (rank / n_arguments)
# =============================================================================
RESPONSE_VARIABLE = "rank_score"  # Options: "bt_score" or "rank_score"

RANDOM_STATE = 42
TEST_SIZE = 0.4  # With N=5000, gives 2000 test samples
N_FOLDS = 10
ALPHA_GRID = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
N_BOOTSTRAP = 1000  # Number of bootstrap iterations for test set CIs
BOOTSTRAP_CI = 0.95  # 95% confidence interval
USE_BOOTSTRAP_COEF_CI = False  # Set to True to show bootstrap CIs on coefficient plots

# Set display and plot options
pd.set_option("display.max_colwidth", 100)

# Pure matplotlib styling (matching plotting_utils.py)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.color"] = "#333333"
plt.rcParams["ytick.color"] = "#333333"
plt.rcParams["text.color"] = "#333333"

# Load vocabularies (use paths relative to this script's location)
SCRIPT_DIR = Path(__file__).parent.resolve()
action_space_dir = SCRIPT_DIR / "action_space"

with open(action_space_dir / "structures.json") as f:
	STRUCTURES = list(json.load(f)["choices"].keys())

TOPIC_SUBTOPICS_FILES = {
	"single_use_plastic_specific_subtopics": "subtopics_specific_single_use_plastic.json",
	"social_media_age_restriction": "subtopics_specific_social_media_age_restriction.json",
	"universal_basic_income": "subtopics_specific_universal_basic_income.json",
	"standardized_testing": "subtopics_specific_standardized_testing.json",
	"meat_tax": "subtopics_specific_meat_tax.json",
}
TOPIC_DISPLAY_NAMES = {
	"single_use_plastic_specific_subtopics": "Plastic Pollution",
	"social_media_age_restriction": "Social Media Restriction",
	"universal_basic_income": "Universal Basic Income",
	"standardized_testing": "Standardized Testing",
	"meat_tax": "Meat Tax",
}
TOPIC_JUDGES = {
	"single_use_plastic_specific_subtopics": "GPT-5-mini",
	"social_media_age_restriction": "GPT-5-mini",
	"universal_basic_income": "GPT-5-mini",
	"standardized_testing": "Gemini-3.1-Flash-Lite",
	"meat_tax": "Claude Haiku 4.5",
}
SUBTOPICS = []

logger.info(f"Structures ({len(STRUCTURES)}): {STRUCTURES}")

# Dataset configuration (ordered by flexibility: least to most)
# File paths are relative to SCRIPT_DIR (argument_generation/ directory)
ARGUMENT_DATA_DIR = SCRIPT_DIR / "argument_data"
TOPIC = "single_use_plastic_specific_subtopics"
FIGURES_DIR = SCRIPT_DIR / "figures"
DATASETS = {
	"strict": {
		"file": ARGUMENT_DATA_DIR / TOPIC / "synthesis_strict" / "pairwise_comparisons_bt_scores.csv",
		"color": "#3498db",  # Blue
		"label": "Strict",
	},
	"faithful": {
		"file": ARGUMENT_DATA_DIR / TOPIC / "synthesis_faithful" / "pairwise_comparisons_bt_scores.csv",
		"color": "#2ecc71",  # Green
		"label": "Faithful",
	},
	"restructured": {
		"file": ARGUMENT_DATA_DIR / TOPIC / "synthesis_restructured" / "pairwise_comparisons_bt_scores.csv",
		"color": "#e74c3c",  # Red
		"label": "Restructured",
	},
}

# Dataset color palettes with model-specific shades (light to dark: M0 -> M1a -> M2)
# - Strict: shades of blue (least flexible)
# - Faithful: shades of green (medium flexibility)
# - Restructured: shades of red (most flexible)
DATASET_PALETTES = {
	"strict": {
		"M0": "#b3d9ff",  # Light blue (length-only baseline)
		"M1a": "#80c1ff",  # Medium-light blue
		"M1b": "#4da6ff",  # Medium blue
		"M1c": "#1a8cff",  # Dark blue
		"M2": "#0059b3",  # Darkest blue
		"base": "#3498db",  # Base color for single-value plots
	},
	"faithful": {
		"M0": "#b3ffb3",  # Light green (length-only baseline)
		"M1a": "#80ff80",  # Medium-light green
		"M1b": "#4dcc4d",  # Medium green
		"M1c": "#33a333",  # Dark green
		"M2": "#1a751a",  # Darkest green
		"base": "#2ecc71",  # Base color for single-value plots
	},
	"restructured": {
		"M0": "#ffb3b3",  # Light red (length-only baseline)
		"M1a": "#ff8080",  # Medium-light red
		"M1b": "#ff4d4d",  # Medium red
		"M1c": "#e60000",  # Dark red
		"M2": "#990000",  # Darkest red
		"base": "#e74c3c",  # Base color for single-value plots
	},
}


# =============================================================================
# Section 2: Feature Engineering Functions
# =============================================================================


def parse_action(action_str):
	"""Parse combined action string into (structure, subtopic) tuple."""
	if pd.isna(action_str) or not action_str:
		return "", ""
	for struct in STRUCTURES:
		if action_str == struct:
			return struct, ""
		if action_str.startswith(struct + "_"):
			return struct, action_str[len(struct) + 1 :]
	if action_str == "finish":
		return "finish", ""
	return action_str, ""


def prepare_data(df):
	"""Parse action columns to extract structure and content for each step.

	Also computes rank_score if RESPONSE_VARIABLE is set to "rank_score".
	The response variable (bt_score or rank_score) is standardized.
	"""
	df = df.copy()

	# Compute rank_score: rank / n_arguments (higher rank = better)
	# rank() gives 1 to n, so we normalize by n to get values in (0, 1]
	n_arguments = len(df)
	df["rank_score"] = df["bt_score"].rank() / n_arguments

	# Standardize the selected response variable
	response_col = RESPONSE_VARIABLE
	df[response_col] = (df[response_col] - df[response_col].mean()) / df[response_col].std()

	# Parse structure and content for steps 1-3
	for step in [1, 2, 3]:
		col = f"step_{step}_structure"
		if col in df.columns:
			parsed = df[col].apply(lambda x: pd.Series(parse_action(x)))
			df[f"structure_{step}"] = parsed[0]
			df[f"content_{step}"] = parsed[1]
	return df


# --- Presence Features (M1 - Topic Model Baseline) ---


def create_structure_presence(df, structures, drop_first=True):
	"""Binary: did this structure appear anywhere in trajectory?

	This is what a topic model could capture - it knows which structures
	were used but not when or in what order.
	"""
	features = pd.DataFrame(index=df.index)
	structs_to_use = structures[1:] if drop_first else structures
	for struct in structs_to_use:
		features[f"has_{struct}"] = (
			(df["structure_1"] == struct)
			| (df["structure_2"] == struct)
			| (df["structure_3"] == struct)
		).astype(int)
	return features


def create_length_feature(df):
	"""Create standardized argument length feature (character count).

	This is a control variable to account for the effect of argument length
	on quality scores.
	"""
	features = pd.DataFrame(index=df.index)
	length = df["final_argument"].str.len()
	# Standardize: (x - mean) / std
	features["argument_length"] = (length - length.mean()) / length.std()
	return features


def create_content_presence(df, subtopics, drop_first=True):
	"""Binary: did this content appear anywhere in trajectory?

	This is what a topic model could capture - it knows which content
	topics were used but not when or in what order.
	"""
	features = pd.DataFrame(index=df.index)
	subtopics_to_use = subtopics[1:] if drop_first else subtopics
	for subtopic in subtopics_to_use:
		features[f"has_{subtopic}"] = (
			(df["content_1"] == subtopic)
			| (df["content_2"] == subtopic)
			| (df["content_3"] == subtopic)
		).astype(int)
	return features


# --- Position Features (M2 - Sequential Model) ---


def create_structure_position(df, structures, drop_first=True):
	"""Create binary indicators for structure at each position.

	With drop_first=True, omits the first structure as reference category.
	"""
	features = pd.DataFrame(index=df.index)
	structs_to_use = structures[1:] if drop_first else structures
	for step in [1, 2, 3]:
		for struct in structs_to_use:
			features[f"step{step}_{struct}"] = (
				df[f"structure_{step}"] == struct
			).astype(int)
	return features


def create_content_position(df, subtopics, drop_first=True):
	"""Create binary indicators for content at each position.

	With drop_first=True, omits the first subtopic as reference category.
	"""
	features = pd.DataFrame(index=df.index)
	subtopics_to_use = subtopics[1:] if drop_first else subtopics
	for step in [1, 2, 3]:
		for subtopic in subtopics_to_use:
			features[f"step{step}_{subtopic}"] = (
				df[f"content_{step}"] == subtopic
			).astype(int)
	return features


def create_position_interactions(df, structures, subtopics, drop_first=False):
	"""Create structure x content interactions at each step.

	At each step, exactly one (structure, content) pair is chosen.
	This captures: "Using structure A with content B at step N has an effect
	beyond the individual main effects of A and B."

	With drop_first=False (default), we include all categories and let LASSO
	handle collinearity, rather than assuming the reference category has no
	special interactions.
	"""
	structs_to_use = structures[1:] if drop_first else structures
	subtopics_to_use = subtopics[1:] if drop_first else subtopics

	feature_dict = {}
	for step in [1, 2, 3]:
		for struct in structs_to_use:
			struct_at_step = (df[f"structure_{step}"] == struct).astype(int)
			for subtopic in subtopics_to_use:
				content_at_step = (df[f"content_{step}"] == subtopic).astype(int)
				feature_dict[f"step{step}_{struct}_x_{subtopic}"] = (
					struct_at_step * content_at_step
				).values
	return pd.DataFrame(feature_dict, index=df.index)


def create_structure_chains(df, structures, drop_first=False):
	"""Create structure transition features (step N -> step N+1).

	These are ACROSS-STEP interactions capturing sequential patterns.

	With drop_first=False (default), we include all categories and let LASSO
	handle collinearity, rather than assuming the reference category has no
	special chain effects.
	"""
	structs_to_use = structures[1:] if drop_first else structures

	feature_dict = {}
	for step1, step2 in [(1, 2), (2, 3)]:
		for s1 in structs_to_use:
			for s2 in structs_to_use:
				s1_at = (df[f"structure_{step1}"] == s1).astype(int)
				s2_at = (df[f"structure_{step2}"] == s2).astype(int)
				feature_dict[f"struct_{s1}_then_{s2}_s{step1}s{step2}"] = (
					s1_at * s2_at
				).values
	return pd.DataFrame(feature_dict, index=df.index)


def create_content_chains(df, subtopics, drop_first=False):
	"""Create content transition features (step N -> step N+1).

	These are ACROSS-STEP interactions for content/subtopic sequences.

	With drop_first=False (default), we include all categories and let LASSO
	handle collinearity, rather than assuming the reference category has no
	special chain effects.
	"""
	subtopics_to_use = subtopics[1:] if drop_first else subtopics

	feature_dict = {}
	for step1, step2 in [(1, 2), (2, 3)]:
		for c1 in subtopics_to_use:
			for c2 in subtopics_to_use:
				c1_at = (df[f"content_{step1}"] == c1).astype(int)
				c2_at = (df[f"content_{step2}"] == c2).astype(int)
				feature_dict[f"content_{c1}_then_{c2}_s{step1}s{step2}"] = (
					c1_at * c2_at
				).values
	return pd.DataFrame(feature_dict, index=df.index)


# =============================================================================
# Section 3: Feature Categorization
# =============================================================================


def categorize_feature(name):
	"""Categorize a feature name into its type."""
	if name.startswith("has_"):
		# Presence features
		for struct in STRUCTURES:
			if struct in name:
				return "Structure Presence"
		return "Content Presence"
	if "_then_" in name:
		# Chain features (length-2)
		if name.startswith("struct_"):
			return "Structure Chains (len-2)"
		return "Content Chains (len-2)"
	if "_x_" in name:
		return "Position Interactions"
	if name.startswith("step"):
		# Position main effects
		for struct in STRUCTURES:
			if struct in name:
				return "Structure Position"
		return "Content Position"
	return "Other"


# =============================================================================
# Section 4: Model Builders
# =============================================================================


def create_m0_features(df):
	"""M0: Argument length only (baseline control).

	This model serves as a baseline to show how much variance is explained
	by argument length alone.
	"""
	return create_length_feature(df)


def create_m1a_features(df):
	"""M1a: Structure presence only (+ length control)."""
	features = [create_structure_presence(df, STRUCTURES)]
	features.append(create_length_feature(df))
	return pd.concat(features, axis=1)


def create_m1b_features(df):
	"""M1b: Content presence only (+ length control)."""
	features = [create_content_presence(df, SUBTOPICS)]
	features.append(create_length_feature(df))
	return pd.concat(features, axis=1)


def create_m1c_features(df):
	"""M1c: Structure + Content presence (full topic model baseline) (+ length control)."""
	features = [
		create_structure_presence(df, STRUCTURES),
		create_content_presence(df, SUBTOPICS),
	]
	features.append(create_length_feature(df))
	return pd.concat(features, axis=1)


def create_m2_features(df):
	"""M2: Full sequential feature set (+ length control).

	Includes:
	1. Position main effects (structure + content at each step)
	2. Position interactions (structure x content at same step)
	3. Structure chains (transitions between steps)
	4. Content chains (transitions between steps)
	5. Argument length (control variable)
	"""
	# Create binary features first
	binary_features = pd.concat(
		[
			create_structure_position(df, STRUCTURES),
			create_content_position(df, SUBTOPICS),
			create_position_interactions(df, STRUCTURES, SUBTOPICS),
			create_structure_chains(df, STRUCTURES),
			create_content_chains(df, SUBTOPICS),
		],
		axis=1,
	)
	# Remove duplicates and zero-variance columns (for binary features only)
	binary_features = binary_features.loc[:, ~binary_features.columns.duplicated()]
	binary_features = binary_features.loc[:, binary_features.sum() > 0]

	# Add continuous length feature (not filtered by sum > 0)
	features = pd.concat([binary_features, create_length_feature(df)], axis=1)
	return features


# =============================================================================
# Section 5: Cross-Validation Framework
# =============================================================================


def run_cv_ols(X, y, n_folds=N_FOLDS):
	"""Run k-fold cross-validation for OLS regression."""
	kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
	model = LinearRegression()
	scores = cross_val_score(model, X, y, cv=kfold, scoring="r2")
	return {"mean": scores.mean(), "std": scores.std(), "scores": scores}


def run_cv_lasso(X, y, alphas=ALPHA_GRID, n_folds=N_FOLDS):
	"""Run k-fold cross-validation across alpha grid for LASSO.

	Returns best alpha and performance metrics.
	"""
	kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
	results = []

	for alpha in alphas:
		lasso = Lasso(alpha=alpha, max_iter=10000)
		scores = cross_val_score(lasso, X, y, cv=kfold, scoring="r2")

		# Fit to get feature count
		lasso.fit(X, y)
		n_selected = np.sum(lasso.coef_ != 0)

		results.append(
			{
				"alpha": alpha,
				"mean": scores.mean(),
				"std": scores.std(),
				"n_selected": n_selected,
				"scores": scores,
			}
		)

	# Find best alpha
	results_df = pd.DataFrame(results)
	best_idx = results_df["mean"].idxmax()
	best_result = results_df.loc[best_idx]

	return {
		"best_alpha": best_result["alpha"],
		"mean": best_result["mean"],
		"std": best_result["std"],
		"n_selected": best_result["n_selected"],
		"scores": results[best_idx]["scores"],  # CV fold scores for interval plots
		"all_results": results_df,
	}


def bootstrap_r2(y_true, y_pred, n_bootstrap=N_BOOTSTRAP, ci=BOOTSTRAP_CI, random_state=RANDOM_STATE):
	"""Compute bootstrap confidence interval for R² score.

	Args:
		y_true: True target values
		y_pred: Predicted values
		n_bootstrap: Number of bootstrap iterations
		ci: Confidence interval level (default 95%)
		random_state: Random seed for reproducibility

	Returns:
		dict with 'point', 'ci_lower', 'ci_upper', 'std', 'bootstrap_scores'
	"""
	rng = np.random.RandomState(random_state)
	n_samples = len(y_true)
	bootstrap_scores = []

	for _ in range(n_bootstrap):
		# Sample with replacement
		indices = rng.choice(n_samples, size=n_samples, replace=True)
		y_true_boot = y_true.iloc[indices] if hasattr(y_true, "iloc") else y_true[indices]
		y_pred_boot = y_pred[indices]
		bootstrap_scores.append(r2_score(y_true_boot, y_pred_boot))

	bootstrap_scores = np.array(bootstrap_scores)
	alpha = 1 - ci
	ci_lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
	ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))

	return {
		"point": r2_score(y_true, y_pred),
		"ci_lower": ci_lower,
		"ci_upper": ci_upper,
		"std": bootstrap_scores.std(),
		"bootstrap_scores": bootstrap_scores,
	}


def bootstrap_coefficients(X, y, alpha, n_bootstrap=N_BOOTSTRAP, ci=BOOTSTRAP_CI, random_state=RANDOM_STATE):
	"""Compute bootstrap confidence intervals for LASSO coefficients.

	Args:
		X: Feature matrix (DataFrame)
		y: Target variable
		alpha: LASSO regularization parameter
		n_bootstrap: Number of bootstrap iterations
		ci: Confidence interval level (default 95%)
		random_state: Random seed for reproducibility

	Returns:
		DataFrame with columns: 'coef', 'ci_lower', 'ci_upper', 'significant'
		(significant = True if CI doesn't cross zero)
	"""
	rng = np.random.RandomState(random_state)
	n_samples = len(y)
	feature_names = X.columns

	# Storage for bootstrap coefficients
	boot_coefs = np.zeros((n_bootstrap, len(feature_names)))

	for b in range(n_bootstrap):
		# Resample with replacement
		indices = rng.choice(n_samples, size=n_samples, replace=True)
		X_boot = X.iloc[indices]
		y_boot = y.iloc[indices] if hasattr(y, "iloc") else y[indices]

		# Fit LASSO with same alpha
		lasso = Lasso(alpha=alpha, max_iter=10000)
		lasso.fit(X_boot, y_boot)
		boot_coefs[b, :] = lasso.coef_

	# Compute point estimates (from full data)
	lasso_full = Lasso(alpha=alpha, max_iter=10000)
	lasso_full.fit(X, y)
	point_coefs = lasso_full.coef_

	# Compute percentile CIs
	alpha_ci = 1 - ci
	ci_lower = np.percentile(boot_coefs, 100 * alpha_ci / 2, axis=0)
	ci_upper = np.percentile(boot_coefs, 100 * (1 - alpha_ci / 2), axis=0)

	# Determine significance (CI doesn't cross zero)
	significant = (ci_lower > 0) | (ci_upper < 0)

	result = pd.DataFrame({
		"coef": point_coefs,
		"ci_lower": ci_lower,
		"ci_upper": ci_upper,
		"significant": significant,
	}, index=feature_names)

	return result


# =============================================================================
# Section 6: Main Analysis Loop
# =============================================================================


def analyze_dataset(name, config):
	"""Analyze a single dataset and return results."""
	logger.info(f"\n{'=' * 60}")
	logger.info(f"Processing: {config['label']} ({config['file']})")
	logger.info(f"{'=' * 60}")

	# Load and prepare data
	df = pd.read_csv(config["file"])
	n_original = len(df)
	df = df.drop_duplicates(subset=["final_argument"])
	n_after_dedup = len(df)
	df = prepare_data(df)
	logger.info(f"Loaded {n_original} samples, {n_after_dedup} after removing duplicates")

	# Train/test split
	train_df, test_df = train_test_split(
		df, test_size=TEST_SIZE, random_state=RANDOM_STATE
	)
	y_train = train_df[RESPONSE_VARIABLE]
	y_test = test_df[RESPONSE_VARIABLE]
	logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
	logger.info(f"Response variable: {RESPONSE_VARIABLE}")
	# Create feature sets
	X_m1a_train = create_m1a_features(train_df)
	X_m1b_train = create_m1b_features(train_df)
	X_m1c_train = create_m1c_features(train_df)
	X_m2_train = create_m2_features(train_df)

	# Remove zero-variance binary columns (but keep continuous features like argument_length)
	def filter_zero_variance(X):
		binary_cols = [c for c in X.columns if c != "argument_length"]
		keep_binary = [c for c in binary_cols if X[c].sum() > 0]
		keep_cols = keep_binary + (["argument_length"] if "argument_length" in X.columns else [])
		return X[keep_cols]

	X_m1a_train = filter_zero_variance(X_m1a_train)
	X_m1b_train = filter_zero_variance(X_m1b_train)
	X_m1c_train = filter_zero_variance(X_m1c_train)

	# M0 features (length only)
	X_m0_train = create_m0_features(train_df)

	logger.info(f"\nFeature counts:")
	logger.info(f"  M0 (Length Only):         {X_m0_train.shape[1]}")
	logger.info(f"  M1a (Structure Presence): {X_m1a_train.shape[1]}")
	logger.info(f"  M1b (Content Presence):   {X_m1b_train.shape[1]}")
	logger.info(f"  M1c (Both Presence):      {X_m1c_train.shape[1]}")
	logger.info(f"  M2 (Full Sequential):     {X_m2_train.shape[1]}")

	# Run cross-validation
	logger.info("\nRunning 10-fold cross-validation...")

	# M0 model (OLS, length only)
	cv_m0 = run_cv_ols(X_m0_train, y_train)

	# M1 models (OLS)
	cv_m1a = run_cv_ols(X_m1a_train, y_train)
	cv_m1b = run_cv_ols(X_m1b_train, y_train)
	cv_m1c = run_cv_ols(X_m1c_train, y_train)

	# M2 model (LASSO with alpha selection)
	cv_m2 = run_cv_lasso(X_m2_train, y_train)

	logger.info(f"\nCV Results:")
	logger.info(f"  M0 (Length Only):         R² = {cv_m0['mean']:.4f} ± {cv_m0['std']:.4f}")
	logger.info(
		f"  M1a (Structure Presence): R² = {cv_m1a['mean']:.4f} ± {cv_m1a['std']:.4f}"
	)
	logger.info(f"  M1b (Content Presence):   R² = {cv_m1b['mean']:.4f} ± {cv_m1b['std']:.4f}")
	logger.info(f"  M1c (Both Presence):      R² = {cv_m1c['mean']:.4f} ± {cv_m1c['std']:.4f}")
	logger.info(
		f"  M2 (Full LASSO):          R² = {cv_m2['mean']:.4f} ± {cv_m2['std']:.4f} (α={cv_m2['best_alpha']}, n={cv_m2['n_selected']})"
	)

	# Fit final models on full training set
	model_m0 = LinearRegression().fit(X_m0_train, y_train)
	model_m1a = LinearRegression().fit(X_m1a_train, y_train)
	model_m1b = LinearRegression().fit(X_m1b_train, y_train)
	model_m1c = LinearRegression().fit(X_m1c_train, y_train)
	model_m2 = Lasso(alpha=cv_m2["best_alpha"], max_iter=10000).fit(X_m2_train, y_train)

	# Prepare test features (aligned with training columns)
	X_m0_test = create_m0_features(test_df).reindex(
		columns=X_m0_train.columns, fill_value=0
	)
	X_m1a_test = create_m1a_features(test_df).reindex(
		columns=X_m1a_train.columns, fill_value=0
	)
	X_m1b_test = create_m1b_features(test_df).reindex(
		columns=X_m1b_train.columns, fill_value=0
	)
	X_m1c_test = create_m1c_features(test_df).reindex(
		columns=X_m1c_train.columns, fill_value=0
	)
	X_m2_test = create_m2_features(test_df).reindex(
		columns=X_m2_train.columns, fill_value=0
	)

	# Test set evaluation with bootstrap confidence intervals
	test_m0 = bootstrap_r2(y_test, model_m0.predict(X_m0_test))
	test_m1a = bootstrap_r2(y_test, model_m1a.predict(X_m1a_test))
	test_m1b = bootstrap_r2(y_test, model_m1b.predict(X_m1b_test))
	test_m1c = bootstrap_r2(y_test, model_m1c.predict(X_m1c_test))
	test_m2 = bootstrap_r2(y_test, model_m2.predict(X_m2_test))

	logger.info(f"\nTest Set Results (with 95% CI):")
	logger.info(f"  M0 (Length Only):         R² = {test_m0['point']:.4f} [{test_m0['ci_lower']:.4f}, {test_m0['ci_upper']:.4f}]")
	logger.info(f"  M1a (Structure Presence): R² = {test_m1a['point']:.4f} [{test_m1a['ci_lower']:.4f}, {test_m1a['ci_upper']:.4f}]")
	logger.info(f"  M1b (Content Presence):   R² = {test_m1b['point']:.4f} [{test_m1b['ci_lower']:.4f}, {test_m1b['ci_upper']:.4f}]")
	logger.info(f"  M1c (Both Presence):      R² = {test_m1c['point']:.4f} [{test_m1c['ci_lower']:.4f}, {test_m1c['ci_upper']:.4f}]")
	logger.info(f"  M2 (Full LASSO):          R² = {test_m2['point']:.4f} [{test_m2['ci_lower']:.4f}, {test_m2['ci_upper']:.4f}]")
	logger.info(f"  Improvement (M2 - M1c):   {test_m2['point'] - test_m1c['point']:+.4f}")

	# Bootstrap CIs for coefficients (optional, controlled by USE_BOOTSTRAP_COEF_CI)
	coef_bootstrap = None
	if USE_BOOTSTRAP_COEF_CI:
		logger.info("  Computing bootstrap CIs for coefficients...")
		coef_bootstrap = bootstrap_coefficients(
			X_m2_train, y_train, cv_m2["best_alpha"]
		)

	# Extract selected features from M2 LASSO
	selected_features = pd.Series(model_m2.coef_, index=X_m2_train.columns)
	selected_features = selected_features[selected_features != 0].sort_values(
		key=abs, ascending=False
	)

	# Categorize selected features
	feature_categories = {}
	for feat in selected_features.index:
		cat = categorize_feature(feat)
		if cat not in feature_categories:
			feature_categories[cat] = []
		feature_categories[cat].append(feat)

	logger.info(f"\nM2 Selected Features by Category:")
	for cat, feats in sorted(feature_categories.items()):
		logger.info(f"  {cat}: {len(feats)}")

	# Build result dict
	n_features = {
		"M0": X_m0_train.shape[1],
		"M1a": X_m1a_train.shape[1],
		"M1b": X_m1b_train.shape[1],
		"M1c": X_m1c_train.shape[1],
		"M2": X_m2_train.shape[1],
		"M2_selected": int(cv_m2["n_selected"]),
	}
	cv_results = {
		"M0": cv_m0,
		"M1a": cv_m1a,
		"M1b": cv_m1b,
		"M1c": cv_m1c,
		"M2": cv_m2,
	}
	test_results = {
		"M0": test_m0,
		"M1a": test_m1a,
		"M1b": test_m1b,
		"M1c": test_m1c,
		"M2": test_m2,
	}

	return {
		"name": name,
		"label": config["label"],
		"color": config["color"],
		"n_train": len(train_df),
		"n_test": len(test_df),
		"n_features": n_features,
		"cv": cv_results,
		"test": test_results,
		"best_alpha": cv_m2["best_alpha"],
		"selected_features": selected_features,
		"feature_categories": feature_categories,
		"alpha_results": cv_m2["all_results"],
		"coef_bootstrap": coef_bootstrap,
		"df": df,
		"model_m2": model_m2,
		"m2_columns": X_m2_train.columns,
	}


# =============================================================================
# Section 7: Visualization Functions
# =============================================================================


def apply_clean_style(ax):
	"""Apply clean plotting style matching plotting_utils.py."""
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.grid(True, alpha=0.3, zorder=0, axis="y")


def plot_interval_with_points(ax, x, data, color, width=0.6, jitter=0.15, show_points=True):
	"""Plot box interval with jittered individual points.

	Args:
		ax: Matplotlib axes object
		x: X-position for the interval plot
		data: Array of data values (e.g., fold scores or bootstrap samples)
		color: Color for the box and points
		width: Width of the box (default 0.6)
		jitter: Amount of horizontal jitter for points (default 0.15)
		show_points: Whether to show individual data points (default True)
	"""
	# Compute statistics
	q1, median, q3 = np.percentile(data, [25, 50, 75])
	min_val, max_val = np.min(data), np.max(data)

	# Draw box (IQR)
	box = plt.Rectangle(
		(x - width / 2, q1), width, q3 - q1,
		facecolor=color, alpha=0.3, edgecolor=color, linewidth=1
	)
	ax.add_patch(box)

	# Draw median line
	ax.hlines(median, x - width / 2, x + width / 2, color=color, linewidth=2)

	# Draw whiskers
	ax.vlines(x, min_val, q1, color=color, linewidth=1)
	ax.vlines(x, q3, max_val, color=color, linewidth=1)
	ax.hlines([min_val, max_val], x - width / 4, x + width / 4, color=color, linewidth=1)

	# Add jittered points
	if show_points:
		jittered_x = x + np.random.uniform(-jitter, jitter, len(data))
		ax.scatter(
			jittered_x, data, color=color, alpha=0.6, s=30,
			edgecolors="white", linewidths=0.5, zorder=5
		)


def plot_length_histograms():
	"""Plot argument length histograms for all three datasets."""
	fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

	for ax, (ds_name, config) in zip(axes, DATASETS.items()):
		# Load data
		df = pd.read_csv(config["file"])
		df = df.drop_duplicates(subset=["final_argument"])
		lengths = df["final_argument"].str.len()

		# Plot histogram
		ax.hist(
			lengths,
			bins=50,
			color=DATASET_PALETTES[ds_name]["base"],
			edgecolor="white",
			linewidth=0.5,
			alpha=0.8,
			range=(350, 1300),
		)

		# Add statistics
		mean_len = lengths.mean()
		median_len = lengths.median()
		ax.axvline(mean_len, color="black", linestyle="--", linewidth=1.5, label=f"Mean: {mean_len:.0f}")
		ax.axvline(median_len, color="black", linestyle=":", linewidth=1.5, label=f"Median: {median_len:.0f}")

		ax.set_xlabel("Argument Length (characters)")
		ax.set_title(f"{config['label']} (N={len(df)})", fontweight="bold")
		ax.legend(fontsize=9)
		apply_clean_style(ax)

	axes[0].set_ylabel("Count")

	plt.suptitle("Argument Length Distributions by Synthesis Type", fontweight="bold", y=1.02)
	plt.tight_layout()
	filename = "argument_length_histograms_rank_arg_length.pdf"
	plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
	plt.close()
	logger.info(f"Saved: {filename}")


def plot_unified_length_histograms():
	"""Plot a 5x3 grid of argument length histograms (topics x synthesis types)."""
	topic_keys = list(TOPIC_SUBTOPICS_FILES.keys())
	synthesis_types = ["strict", "faithful", "restructured"]
	synthesis_labels = {"strict": "Strict", "faithful": "Faithful", "restructured": "Restructured"}
	n_topics = len(topic_keys)
	n_synth = len(synthesis_types)

	fig, axes = plt.subplots(
		n_topics, n_synth, figsize=(12, 2 * n_topics), sharex=True, sharey=True
	)

	for row, topic_key in enumerate(topic_keys):
		display_name = TOPIC_DISPLAY_NAMES.get(topic_key, topic_key)
		for col, synth in enumerate(synthesis_types):
			ax = axes[row, col]
			csv_path = (
				ARGUMENT_DATA_DIR / topic_key / f"synthesis_{synth}"
				/ "pairwise_comparisons_bt_scores.csv"
			)
			df = pd.read_csv(csv_path)
			df = df.drop_duplicates(subset=["final_argument"])
			lengths = df["final_argument"].str.len()

			ax.hist(
				lengths,
				bins=50,
				color=DATASET_PALETTES[synth]["base"],
				edgecolor="white",
				linewidth=0.5,
				alpha=0.8,
			)

			mean_len = lengths.mean()
			median_len = lengths.median()
			ax.axvline(
				mean_len, color="black", linestyle="--", linewidth=1.2,
				label=f"Mean: {mean_len:.0f}",
			)
			ax.axvline(
				median_len, color="black", linestyle=":", linewidth=1.2,
				label=f"Median: {median_len:.0f}",
			)

			ax.legend(fontsize=7, loc="upper right")
			apply_clean_style(ax)

			# Column titles on first row
			if row == 0:
				ax.set_title(synthesis_labels[synth], fontweight="bold", fontsize=12)

			# Row labels on first column
			if col == 0:
				ax.set_ylabel(display_name, fontsize=10)

			# X-axis label on last row
			if row == n_topics - 1:
				ax.set_xlabel("Argument Length (chars)", fontsize=9)

	plt.tight_layout()
	unified_dir = SCRIPT_DIR / "figures" / "predictability"
	unified_dir.mkdir(parents=True, exist_ok=True)
	filename = "unified_length_histograms.pdf"
	plt.savefig(unified_dir / filename, dpi=300, bbox_inches="tight")
	plt.close()
	logger.info(f"Saved unified length histograms: {filename}")

	# Copy to paper figures
	paper_fig_dir = SCRIPT_DIR / ".." / ".." / "paper" / "figures" / "results" / "argument_generation"
	paper_fig_dir.mkdir(parents=True, exist_ok=True)
	shutil.copy2(unified_dir / filename, paper_fig_dir / filename)
	logger.info(f"Copied {filename} to {paper_fig_dir}")


def plot_feature_categories(all_results):
	"""Figure 4: M2 Feature Category Breakdown as Stacked Bar Chart."""
	fig, ax = plt.subplots(figsize=(8, 6))

	category_colors = {
		"Structure Position": "#e74c3c",
		"Content Position": "#3498db",
		"Position Interactions": "#2ecc71",
		"Structure Chains (len-2)": "#f39c12",
		"Content Chains (len-2)": "#9b59b6",
	}

	categories = [
		"Structure Position",
		"Content Position",
		"Position Interactions",
		"Structure Chains (len-2)",
		"Content Chains (len-2)",
	]

	datasets_list = list(DATASETS.keys())
	x = np.arange(len(datasets_list))

	bottom = np.zeros(len(datasets_list))
	for cat in categories:
		counts = [
			len(all_results[ds]["feature_categories"].get(cat, []))
			for ds in datasets_list
		]
		ax.bar(x, counts, bottom=bottom, label=cat, color=category_colors[cat])

		for i, (count, b) in enumerate(zip(counts, bottom)):
			if count > 0:
				ax.text(
					i, b + count / 2, str(count),
					ha="center", va="center", fontsize=11, fontweight="bold"
				)

		bottom = bottom + np.array(counts)

	ax.set_xticks(x)
	ax.set_xticklabels([DATASETS[ds]["label"] for ds in datasets_list], fontsize=12)
	ax.set_xlabel("Synthesis Type", fontsize=13)
	ax.set_ylabel("Number of Selected Features", fontsize=13)
	ax.set_title("M2 Feature Categories", fontweight="bold", fontsize=14)
	ax.tick_params(axis="both", labelsize=11)
	apply_clean_style(ax)

	# Legend
	handles, labels = ax.get_legend_handles_labels()
	fig.legend(
		handles, labels,
		loc="lower center",
		bbox_to_anchor=(0.5, -0.02),
		ncol=len(categories),
		fontsize=11,
		frameon=False,
	)

	plt.tight_layout(rect=[0, 0.06, 1, 1])
	filename = "m2_feature_categories_rank.pdf"
	plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
	plt.close()
	logger.info(f"Saved: {filename}")


def plot_top_features(all_results):
	"""Figure 5: Top Sequential Features from M2 LASSO.

	Uses vertical 3x1 layout to give more horizontal space for feature labels.
	If USE_BOOTSTRAP_COEF_CI is True, displays 95% bootstrap confidence intervals.
	"""
	from matplotlib.patches import Patch

	fig, axes = plt.subplots(3, 1, figsize=(12, 16))

	category_colors = {
		"Structure Position": "#e74c3c",
		"Content Position": "#3498db",
		"Position Interactions": "#2ecc71",
		"Structure Chains (len-2)": "#f39c12",
		"Content Chains (len-2)": "#9b59b6",
	}

	for ax, ds in zip(axes, DATASETS):
		result = all_results[ds]

		# Check if we have bootstrap CIs available
		use_ci = USE_BOOTSTRAP_COEF_CI and result.get("coef_bootstrap") is not None

		if use_ci:
			# CI mode: use coef_bootstrap DataFrame
			coef_df = result["coef_bootstrap"]
			nonzero = coef_df[coef_df["coef"] != 0].copy()
			nonzero["abs_coef"] = nonzero["coef"].abs()
			top_feats = nonzero.nlargest(20, "abs_coef")
		else:
			# Original mode: use selected_features Series
			top_feats = result["selected_features"].head(20)

		if len(top_feats) == 0:
			ax.text(0.5, 0.5, "No features selected", ha="center", va="center")
			ax.set_title(result["label"])
			continue

		y_pos = np.arange(len(top_feats))

		if use_ci:
			# CI mode: color by significance, plot CI bars
			colors = []
			for _, row in top_feats.iterrows():
				if row["ci_lower"] > 0:  # Entirely positive
					colors.append("#90EE90")  # Light green
				elif row["ci_upper"] < 0:  # Entirely negative
					colors.append("#FFB6C1")  # Light pink/red
				else:  # Crosses zero
					colors.append("#D3D3D3")  # Light gray

			# Plot CI bars (horizontal rectangles)
			for i, (idx, row) in enumerate(top_feats.iterrows()):
				ax.barh(
					y_pos[i],
					row["ci_upper"] - row["ci_lower"],  # width
					left=row["ci_lower"],  # starting x
					height=0.6,
					color=colors[i],
					edgecolor="gray",
					linewidth=0.5,
				)

			# Plot point estimates as dots
			ax.scatter(
				top_feats["coef"],
				y_pos,
				color="black",
				s=30,
				zorder=5,
			)

			# Vertical line at zero
			ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.7)
			ax.set_xlabel("Estimated Effects")
			ax.set_title(
				f"{result['label']} - Top 20 Features with 95% CI",
				fontweight="bold"
			)
		else:
			# Original mode: color by category, plot solid bars
			colors = [category_colors.get(categorize_feature(f), "#95a5a6") for f in top_feats.index]
			ax.barh(y_pos, top_feats.values, color=colors)
			ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
			ax.set_xlabel("Coefficient")
			ax.set_title(
				f"{result['label']} - Top 20 Features (|coef|)", fontweight="bold"
			)

		# Labels (common to both modes)
		labels = [f.replace("_", " ") for f in top_feats.index]
		ax.set_yticks(y_pos)
		ax.set_yticklabels(labels, fontsize=9)
		ax.invert_yaxis()
		apply_clean_style(ax)

	# Legend depends on mode
	if USE_BOOTSTRAP_COEF_CI and all(r.get("coef_bootstrap") is not None for r in all_results.values()):
		legend_elements = [
			Patch(facecolor="#90EE90", edgecolor="gray", label="Positive (significant)"),
			Patch(facecolor="#FFB6C1", edgecolor="gray", label="Negative (significant)"),
			Patch(facecolor="#D3D3D3", edgecolor="gray", label="Not significant (crosses 0)"),
		]
	else:
		legend_elements = [Patch(facecolor=c, label=k) for k, c in category_colors.items()]

	fig.legend(
		handles=legend_elements,
		loc="lower center",
		ncol=3,
		bbox_to_anchor=(0.5, -0.02),
		fontsize=10,
		frameon=True,
	)

	plt.tight_layout(rect=[0, 0.03, 1, 1])
	filename = "m2_top_features_rank_arg_length.pdf"
	plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
	plt.close()
	logger.info(f"Saved: {filename}")


def plot_alpha_selection(all_results):
	"""Figure 6: Alpha Selection Curves."""
	fig, ax = plt.subplots(figsize=(8, 6))
	datasets_list = list(DATASETS.keys())

	for ds in datasets_list:
		result = all_results[ds]
		alpha_df = result["alpha_results"]
		base_color = DATASET_PALETTES[ds]["base"]

		ax.errorbar(
			alpha_df["alpha"],
			alpha_df["mean"],
			yerr=alpha_df["std"],
			marker="o",
			capsize=3,
			label=result["label"],
			color=base_color,
			linewidth=2,
		)

		best_idx = alpha_df["mean"].idxmax()
		best = alpha_df.loc[best_idx]
		ax.scatter(
			[best["alpha"]],
			[best["mean"]],
			marker="*",
			s=200,
			color=base_color,
			zorder=5,
		)

	ax.set_xscale("log")
	ax.set_xlabel("Alpha (log scale)", fontsize=13)
	ax.set_ylabel("CV R²", fontsize=13)
	ax.set_title("M2 Alpha Selection", fontweight="bold", fontsize=14)
	ax.legend(fontsize=11)
	ax.tick_params(axis="both", labelsize=11)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.grid(True, alpha=0.3, zorder=0, axis="y")

	plt.tight_layout()
	filename = "m2_alpha_selection_rank.pdf"
	plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
	plt.close()
	logger.info(f"Saved: {filename}")


def plot_main_figure(all_results):
	"""Main Figure: CV Performance (left) + Test Set Performance (right).

	This is the primary figure for the main body of the paper, combining
	cross-validation results with held-out test set evaluation.
	"""
	from matplotlib.patches import Patch

	fig, axes = plt.subplots(1, 2, figsize=(14, 5))
	datasets_list = list(DATASETS.keys())

	# -------------------------------------------------------------------------
	# Left panel: CV Performance Comparison
	# -------------------------------------------------------------------------
	ax = axes[0]
	models = ["M0", "M1a", "M1b", "M1c", "M2"]
	x = np.arange(len(DATASETS))
	width = 0.15

	for i, model in enumerate(models):
		means = [all_results[ds]["cv"][model]["mean"] for ds in DATASETS]
		stds = [all_results[ds]["cv"][model]["std"] for ds in DATASETS]

		colors = [DATASET_PALETTES[ds][model] for ds in datasets_list]
		ax.bar(
			x + i * width, means, width, yerr=stds, capsize=3,
			color=colors, edgecolor="white", linewidth=0.5
		)

	ax.set_xlabel("Synthesis Type")
	ax.set_ylabel("CV R²")
	ax.set_title("(A) Cross-Validation Performance\n(Intervals show observed range)", fontweight="bold")
	ax.set_xticks(x + (len(models) - 1) * width / 2)
	ax.set_xticklabels([DATASETS[ds]["label"] for ds in DATASETS])

	model_legend = [
		Patch(facecolor="#999999", edgecolor="black", label="M0 (Length Only)"),
		Patch(facecolor="#777777", edgecolor="black", label="M1a (Structure)"),
		Patch(facecolor="#555555", edgecolor="black", label="M1b (Content)"),
		Patch(facecolor="#333333", edgecolor="black", label="M1c (Both)"),
		Patch(facecolor="#111111", edgecolor="black", label="M2 (Sequential)"),
	]
	ax.legend(
		handles=model_legend,
		loc="lower right",
		fontsize=8,
		frameon=True,
	)
	apply_clean_style(ax)

	# -------------------------------------------------------------------------
	# Right panel: Test Set Performance with 95% CI
	# -------------------------------------------------------------------------
	ax = axes[1]
	models = ["M0", "M1a", "M1b", "M1c", "M2"]
	x = np.arange(len(DATASETS))
	width = 0.15

	for i, model in enumerate(models):
		scores = [all_results[ds]["test"][model]["point"] for ds in DATASETS]
		colors = [DATASET_PALETTES[ds][model] for ds in datasets_list]

		# Calculate asymmetric error bars
		lower_errs = [all_results[ds]["test"][model]["point"] - all_results[ds]["test"][model]["ci_lower"] for ds in DATASETS]
		upper_errs = [all_results[ds]["test"][model]["ci_upper"] - all_results[ds]["test"][model]["point"] for ds in DATASETS]

		for j, ds in enumerate(datasets_list):
			ax.bar(x[j] + i * width, scores[j], width, color=colors[j], edgecolor="white", linewidth=0.5,
				   yerr=[[lower_errs[j]], [upper_errs[j]]], capsize=2, error_kw={"ecolor": "black", "elinewidth": 0.8})

	ax.set_xlabel("Synthesis Type")
	ax.set_ylabel("Test R²")
	ax.set_title("(B) Test Set Performance\n(95% CI)", fontweight="bold")
	ax.set_xticks(x + (len(models) - 1) * width / 2)
	ax.set_xticklabels([DATASETS[ds]["label"] for ds in DATASETS])

	ax.legend(
		handles=model_legend,
		loc="lower right",
		fontsize=8,
		frameon=True,
	)
	apply_clean_style(ax)

	plt.tight_layout()
	filename = "m1_vs_m2_main_figure_rank_arg_length.pdf"
	plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
	plt.close()
	logger.info(f"Saved: {filename}")


def plot_unified_cross_topic(
	topic_results: dict[str, dict[str, dict]],
) -> None:
	"""Unified 1x3 panel figure showing Test R² across all topics.

	Args:
		topic_results: {topic_key: all_results_dict} where each all_results_dict
			is keyed by synthesis type with "test" sub-dicts containing
			"point", "ci_lower", "ci_upper" per model.
	"""
	from matplotlib.patches import Patch

	topic_keys = list(topic_results.keys())
	n_topics = len(topic_keys)
	fig_width = max(18, n_topics * 3.6)
	fig, axes = plt.subplots(1, n_topics, figsize=(fig_width, 5), sharey=True)

	if n_topics == 1:
		axes = [axes]

	datasets_list = ["strict", "faithful", "restructured"]
	models = ["M0", "M1a", "M1b", "M1c", "M2"]
	n_models = len(models)
	width = 0.15
	x = np.arange(len(datasets_list))
	panel_labels = [chr(ord("A") + i) for i in range(n_topics)]

	for panel_idx, topic_key in enumerate(topic_keys):
		ax = axes[panel_idx]
		all_results = topic_results[topic_key]
		display_name = TOPIC_DISPLAY_NAMES.get(topic_key, topic_key)

		for i, model in enumerate(models):
			for j, ds in enumerate(datasets_list):
				score = all_results[ds]["test"][model]["point"]
				lower_err = score - all_results[ds]["test"][model]["ci_lower"]
				upper_err = all_results[ds]["test"][model]["ci_upper"] - score
				color = DATASET_PALETTES[ds][model]

				ax.bar(
					x[j] + i * width,
					score,
					width,
					color=color,
					edgecolor="white",
					linewidth=0.5,
					yerr=[[lower_err], [upper_err]],
					capsize=2,
					error_kw={"ecolor": "black", "elinewidth": 0.8},
				)

		ax.set_xlabel("Synthesis Type")
		ax.set_title(
			f"({panel_labels[panel_idx]}) {display_name}", fontweight="bold"
		)
		ax.set_xticks(x + (n_models - 1) * width / 2)
		ax.set_xticklabels(
			[DATASETS[ds]["label"] for ds in datasets_list]
		)
		apply_clean_style(ax)

		if panel_idx == 0:
			ax.set_ylabel("Test R²")

	# Single legend on the right
	model_legend = [
		Patch(facecolor="#999999", edgecolor="black", label="M0 (Length Only)"),
		Patch(facecolor="#777777", edgecolor="black", label="M1a (Structure)"),
		Patch(facecolor="#555555", edgecolor="black", label="M1b (Content)"),
		Patch(facecolor="#333333", edgecolor="black", label="M1c (Both)"),
		Patch(facecolor="#111111", edgecolor="black", label="M2 (Sequential)"),
	]
	axes[-1].legend(
		handles=model_legend,
		loc="lower right",
		fontsize=8,
		frameon=True,
	)

	plt.tight_layout()
	unified_dir = SCRIPT_DIR / "figures" / "predictability"
	unified_dir.mkdir(parents=True, exist_ok=True)
	filename = "unified_cross_topic_test_r2_rank_arg_length.pdf"
	plt.savefig(unified_dir / filename, dpi=300, bbox_inches="tight")
	plt.close()
	logger.info(f"Saved unified cross-topic figure: {filename}")


def plot_strict_cross_topic(
	topic_results: dict[str, dict[str, dict]],
) -> None:
	"""Single-panel grouped bar chart of strict-only Test R2 across all topics.

	Args:
		topic_results: {topic_key: all_results_dict} where each all_results_dict
			is keyed by synthesis type with "test" sub-dicts containing
			"point", "ci_lower", "ci_upper" per model.
	"""
	from matplotlib.patches import Patch

	topic_keys = list(topic_results.keys())
	n_topics = len(topic_keys)

	models = ["M0", "M1a", "M1b", "M1c", "M2"]
	n_models = len(models)
	width = 0.15

	# Blue shades for strict-only figure (light to dark: M0 -> M2)
	model_colors = {
		"M0": "#b3d9ff",
		"M1a": "#80c1ff",
		"M1b": "#4da6ff",
		"M1c": "#1a8cff",
		"M2": "#0059b3",
	}
	fig, ax = plt.subplots(figsize=(10, 4))
	x = np.arange(n_topics)

	for i, model in enumerate(models):
		for j, topic_key in enumerate(topic_keys):
			results = topic_results[topic_key]
			if "strict" not in results:
				continue
			score = results["strict"]["test"][model]["point"]
			lower_err = score - results["strict"]["test"][model]["ci_lower"]
			upper_err = results["strict"]["test"][model]["ci_upper"] - score

			ax.bar(
				x[j] + i * width,
				score,
				width,
				color=model_colors[model],
				edgecolor="white",
				linewidth=0.5,
				yerr=[[lower_err], [upper_err]],
				capsize=2,
				error_kw={"ecolor": "black", "elinewidth": 0.8},
			)

	# Two-line x-axis labels: topic name + judge
	x_labels = []
	for topic_key in topic_keys:
		display_name = TOPIC_DISPLAY_NAMES.get(topic_key, topic_key)
		judge = TOPIC_JUDGES.get(topic_key, "")
		x_labels.append(f"{display_name}\n({judge})")

	ax.set_xticks(x + (n_models - 1) * width / 2)
	ax.set_xticklabels(x_labels, fontsize=9)
	ax.set_ylabel("Test R\u00b2")
	ax.set_title("Predictability Across Topics", fontweight="bold")
	apply_clean_style(ax)

	# Legend (blue shades matching bars)
	model_legend = [
		Patch(facecolor="#b3d9ff", edgecolor="black", label="M0 (Length Only)"),
		Patch(facecolor="#80c1ff", edgecolor="black", label="M1a (Structure)"),
		Patch(facecolor="#4da6ff", edgecolor="black", label="M1b (Content)"),
		Patch(facecolor="#1a8cff", edgecolor="black", label="M1c (Both)"),
		Patch(facecolor="#0059b3", edgecolor="black", label="M2 (Sequential)"),
	]
	ax.legend(
		handles=model_legend,
		loc="lower right",
		fontsize=8,
		frameon=True,
	)

	plt.tight_layout()
	unified_dir = SCRIPT_DIR / "figures" / "predictability"
	unified_dir.mkdir(parents=True, exist_ok=True)
	filename = "strict_cross_topic_test_r2_rank_arg_length.pdf"
	plt.savefig(unified_dir / filename, dpi=300, bbox_inches="tight")
	plt.close()
	logger.info(f"Saved strict cross-topic figure: {filename}")


def _fmt_r2(result_dict):
	"""Format R² as point ± half-width (e.g., 0.366\\,$\\pm$\\,0.037)."""
	pt = result_dict["point"]
	hw = (result_dict["ci_upper"] - result_dict["ci_lower"]) / 2
	return f"{pt:.3f}\\,$\\pm$\\,{hw:.3f}"



def plot_cross_topic_by_synthesis(
	topic_results: dict[str, dict[str, dict]],
) -> None:
	"""3-row cross-topic figure: one row per synthesis mode (strict, faithful, restructured).

	Each row is a grouped bar chart with topics on x-axis and models as grouped bars,
	matching the layout of plot_strict_cross_topic but stacked vertically for all
	three synthesis modes.

	Args:
		topic_results: {topic_key: all_results_dict} where each all_results_dict
			is keyed by synthesis type with "test" sub-dicts containing
			"point", "ci_lower", "ci_upper" per model.
	"""
	from matplotlib.patches import Patch

	topic_keys = list(topic_results.keys())
	n_topics = len(topic_keys)

	synthesis_modes = ["strict", "faithful", "restructured"]
	synthesis_labels = {
		"strict": "Predictability - Strict Synthesis",
		"faithful": "Predictability - Faithful Synthesis",
		"restructured": "Predictability - Restructured Synthesis",
	}

	models = ["M0", "M1a", "M1b", "M1c", "M2"]
	n_models = len(models)
	width = 0.15

	fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharey=True)

	for row, synth in enumerate(synthesis_modes):
		ax = axes[row]
		# Use synthesis-specific color shades
		model_colors = {m: DATASET_PALETTES[synth][m] for m in models}

		x = np.arange(n_topics)

		for i, model in enumerate(models):
			for j, topic_key in enumerate(topic_keys):
				results = topic_results[topic_key]
				if synth not in results:
					continue
				score = results[synth]["test"][model]["point"]
				lower_err = score - results[synth]["test"][model]["ci_lower"]
				upper_err = results[synth]["test"][model]["ci_upper"] - score

				ax.bar(
					x[j] + i * width,
					score,
					width,
					color=model_colors[model],
					edgecolor="white",
					linewidth=0.5,
					yerr=[[lower_err], [upper_err]],
					capsize=2,
					error_kw={"ecolor": "black", "elinewidth": 0.8},
				)

		# Two-line x-axis labels: topic name + judge
		x_labels = []
		for topic_key in topic_keys:
			display_name = TOPIC_DISPLAY_NAMES.get(topic_key, topic_key)
			judge = TOPIC_JUDGES.get(topic_key, "")
			x_labels.append(f"{display_name}\n({judge})")

		ax.set_xticks(x + (n_models - 1) * width / 2)
		ax.set_xticklabels(x_labels, fontsize=9)
		ax.set_ylabel("Test R\u00b2")
		ax.set_title(synthesis_labels[synth], fontweight="bold")
		apply_clean_style(ax)

	# Shared legend from first row
	model_legend = [
		Patch(facecolor="#999999", edgecolor="black", label="M0 (Length Only)"),
		Patch(facecolor="#777777", edgecolor="black", label="M1a (Structure)"),
		Patch(facecolor="#555555", edgecolor="black", label="M1b (Content)"),
		Patch(facecolor="#333333", edgecolor="black", label="M1c (Both)"),
		Patch(facecolor="#111111", edgecolor="black", label="M2 (Sequential)"),
	]
	fig.legend(
		handles=model_legend,
		loc="lower center",
		bbox_to_anchor=(0.5, -0.02),
		ncol=len(models),
		fontsize=8,
		frameon=True,
	)

	plt.tight_layout(rect=[0, 0.04, 1, 1])
	unified_dir = SCRIPT_DIR / "figures" / "predictability"
	unified_dir.mkdir(parents=True, exist_ok=True)
	filename = "cross_topic_by_synthesis_test_r2_rank_arg_length.pdf"
	plt.savefig(unified_dir / filename, dpi=300, bbox_inches="tight")
	plt.close()
	logger.info(f"Saved cross-topic by synthesis figure: {filename}")
	_copy_to_paper(unified_dir, filename)


def generate_latex_table(results_no_length, results_with_length, topic_key=None):
	"""Generate LaTeX table with all models, two sections (with/without length control).

	Args:
		results_no_length: Results dict from run without length control.
		results_with_length: Results dict from run with length control.
		topic_key: Optional topic key for per-topic table filename.

	Returns the LaTeX code as a string and saves to file.
	"""
	lines = []
	lines.append(r"\begin{table}[htbp]")
	lines.append(r"\centering")
	lines.append(r"\small")
	lines.append(r"\caption{Model comparison across synthesis types. "
				 r"R$^2$ values on held-out test set with 95\% bootstrap CI ($\pm$ half-width). ")
	lines.append(r"\label{tab:argument_generation_results}")
	lines.append(r"\begin{tabular}{lcccccc}")
	lines.append(r"\toprule")
	lines.append(
		r"Synthesis & N & M0 R$^2$ & M1a R$^2$ & M1b R$^2$ & M1c R$^2$ & M2 R$^2$ \\"
	)
	lines.append(r"\midrule")

	# --- Section 1: Without length control ---
	models_no_len = ["M1a", "M1b", "M1c", "M2"]
	for ds in DATASETS:
		r = results_no_length[ds]

		cols = [
			r["label"],
			f"{r['n_train']}/{r['n_test']}",
			"--",  # M0 not available without length control
		]
		for m in models_no_len:
			val = _fmt_r2(r["test"][m])
			cols.append(r"\textbf{" + val + "}" if m == "M2" else val)
		lines.append(" & ".join(cols) + " \\\\")

	# --- Separator for length control section ---
	lines.append(r"\midrule")
	lines.append(r"\multicolumn{7}{l}{\textit{+ Argument length control}} \\")
	lines.append(r"\midrule")

	# --- Section 2: With length control ---
	models_with_len = ["M0", "M1a", "M1b", "M1c", "M2"]
	for ds in DATASETS:
		r = results_with_length[ds]

		cols = [
			r["label"],
			f"{r['n_train']}/{r['n_test']}",
		]
		for m in models_with_len:
			val = _fmt_r2(r["test"][m])
			cols.append(r"\textbf{" + val + "}" if m == "M2" else val)
		lines.append(" & ".join(cols) + " \\\\")

	lines.append(r"\bottomrule")
	lines.append(r"\end{tabular}")
	lines.append(r"\end{table}")

	latex_code = "\n".join(lines)

	# Save to file
	suffix = f"_{topic_key}" if topic_key else ""
	out_path = SCRIPT_DIR / ".." / ".." / "paper" / "latex" / "tables" / f"argument_generation_results{suffix}.tex"
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(latex_code)
	logger.info(f"Saved LaTeX table: {out_path.resolve()}")

	logger.info("\n" + "=" * 70)
	logger.info("LaTeX Table Code")
	logger.info("=" * 70)
	logger.info(latex_code)

	return latex_code


def generate_unified_latex_table(
	all_topics_no_length: dict[str, dict[str, dict]],
	all_topics_with_length: dict[str, dict[str, dict]],
) -> str:
	"""Generate a unified LaTeX table with results for all topics.

	Args:
		all_topics_no_length: {topic_key: results_dict} without length control.
		all_topics_with_length: {topic_key: results_dict} with length control.

	Returns the LaTeX code as a string and saves to file.
	"""
	lines = []
	lines.append(r"\begin{table*}[htbp]")
	lines.append(r"\centering")
	lines.append(r"\small")
	lines.append(r"\begin{tabular}{llcccc}")
	lines.append(r"\toprule")
	lines.append(
		r"Topic & Synthesis & M1a R$^2$ & M1b R$^2$ & M1c R$^2$ & M2 R$^2$ \\"
	)
	lines.append(r"\midrule")

	datasets_order = ["strict", "faithful", "restructured"]

	for topic_idx, (topic_key, results) in enumerate(all_topics_no_length.items()):
		display_name = TOPIC_DISPLAY_NAMES.get(topic_key, topic_key)

		for ds_idx, ds in enumerate(datasets_order):
			r = results[ds]
			# Show topic name only on first row of each topic group
			topic_col = display_name if ds_idx == 0 else ""

			cols = [
				topic_col,
				r["label"],
				_fmt_r2(r["test"]["M1a"]),
				_fmt_r2(r["test"]["M1b"]),
				_fmt_r2(r["test"]["M1c"]),
				r"\textbf{" + _fmt_r2(r["test"]["M2"]) + "}",
			]
			lines.append(" & ".join(cols) + " \\\\")

		# Add midrule between topics (but not after the last one)
		if topic_idx < len(all_topics_no_length) - 1:
			lines.append(r"\midrule")

	lines.append(r"\bottomrule")
	lines.append(r"\end{tabular}")
	lines.append(
		r"\caption{Model comparison across synthesis types and debate topics. "
		r"R$^2$ values on held-out test set (40\%) with 95\% bootstrap CI ($\pm$ half-width). "
		r"M1a uses structure presence features, M1b content presence features, "
		r"and M1c both structure and content presence features. "
		r"M2 LASSO models select from sequential structure and content features "
		r"as detailed in Section~\ref{sec:experiments-argument-generation}.}"
	)
	lines.append(r"\label{tab:argument_generation_results}")
	lines.append(r"\end{table*}")

	latex_code = "\n".join(lines)

	# Save to paper/tables/ (new preprint location)
	out_path = SCRIPT_DIR / ".." / ".." / "paper" / "tables" / "argument_generation_results.tex"
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(latex_code)
	logger.info(f"Saved unified LaTeX table: {out_path.resolve()}")

	# Also save to paper/latex/tables/ (legacy location)
	out_path_legacy = SCRIPT_DIR / ".." / ".." / "paper" / "latex" / "tables" / "argument_generation_results.tex"
	out_path_legacy.parent.mkdir(parents=True, exist_ok=True)
	out_path_legacy.write_text(latex_code)
	logger.info(f"Saved unified LaTeX table (legacy): {out_path_legacy.resolve()}")

	logger.info("\n" + "=" * 70)
	logger.info("Unified LaTeX Table Code")
	logger.info("=" * 70)
	logger.info(latex_code)

	# Also generate the length-controlled version
	lines_len = []
	lines_len.append(r"\begin{table*}[htbp]")
	lines_len.append(r"\centering")
	lines_len.append(r"\small")
	lines_len.append(r"\begin{tabular}{llccccc}")
	lines_len.append(r"\toprule")
	lines_len.append(
		r"Topic & Synthesis & M0 R$^2$ & M1a R$^2$ & M1b R$^2$ & M1c R$^2$ & M2 R$^2$ \\"
	)
	lines_len.append(r"\midrule")

	for topic_idx, (topic_key, results) in enumerate(all_topics_with_length.items()):
		display_name = TOPIC_DISPLAY_NAMES.get(topic_key, topic_key)

		for ds_idx, ds in enumerate(datasets_order):
			r = results[ds]
			topic_col = display_name if ds_idx == 0 else ""

			cols = [
				topic_col,
				r["label"],
				_fmt_r2(r["test"]["M0"]),
				_fmt_r2(r["test"]["M1a"]),
				_fmt_r2(r["test"]["M1b"]),
				_fmt_r2(r["test"]["M1c"]),
				r"\textbf{" + _fmt_r2(r["test"]["M2"]) + "}",
			]
			lines_len.append(" & ".join(cols) + " \\\\")

		if topic_idx < len(all_topics_with_length) - 1:
			lines_len.append(r"\midrule")

	lines_len.append(r"\bottomrule")
	lines_len.append(r"\end{tabular}")
	lines_len.append(
		r"\caption{Model comparison with argument length control across topics. "
		r"R$^2$ values on held-out test set (40\%) with 95\% bootstrap CI ($\pm$ half-width). "
		r"M0 uses only argument length (characters). All other models additionally include length.}"
	)
	lines_len.append(r"\label{tab:argument_generation_results_length}")
	lines_len.append(r"\end{table*}")

	latex_len_code = "\n".join(lines_len)

	out_path_len = SCRIPT_DIR / ".." / ".." / "paper" / "tables" / "argument_generation_results_length.tex"
	out_path_len.write_text(latex_len_code)
	logger.info(f"Saved length-controlled LaTeX table: {out_path_len.resolve()}")

	return latex_code


# =============================================================================
# =============================================================================
# Section 8: Main Execution
# =============================================================================


def export_coefficients_csv(all_results):
	"""Export M2 LASSO coefficients to CSV for each dataset."""
	for ds_name, result in all_results.items():
		selected = result["selected_features"]
		rows = []
		for feat, coef in selected.items():
			rows.append({
				"feature": feat,
				"coefficient": coef,
				"abs_coefficient": abs(coef),
				"category": categorize_feature(feat),
			})
		coef_df = pd.DataFrame(rows)
		subdir = ARGUMENT_DATA_DIR / TOPIC / f"synthesis_{ds_name}"
		subdir.mkdir(parents=True, exist_ok=True)
		out_path = subdir / f"m2_coefficients_{ds_name}.csv"
		coef_df.to_csv(out_path, index=False)
		logger.info(f"Saved: {out_path} ({len(coef_df)} non-zero coefficients)")


def export_trajectory_predictions(all_results):
	"""Export full trajectory prediction table (all combos) for each dataset."""
	for ds_name, result in all_results.items():
		logger.info(f"\nBuilding trajectory predictions for {ds_name}...")
		model_m2 = result["model_m2"]
		m2_columns = result["m2_columns"]
		df = result["df"]

		# Generate all 1M combinations
		combos = list(itertools.product(
			STRUCTURES, SUBTOPICS,
			STRUCTURES, SUBTOPICS,
			STRUCTURES, SUBTOPICS,
		))
		combo_df = pd.DataFrame(combos, columns=[
			"structure_1", "content_1",
			"structure_2", "content_2",
			"structure_3", "content_3",
		])

		# Build M2 binary features from synthetic DataFrame
		# (no final_argument column, so skip length; reindex fills it with 0 = mean)
		synthetic_df = combo_df.copy()
		X_syn = pd.concat(
			[
				create_structure_position(synthetic_df, STRUCTURES),
				create_content_position(synthetic_df, SUBTOPICS),
				create_position_interactions(synthetic_df, STRUCTURES, SUBTOPICS),
				create_structure_chains(synthetic_df, STRUCTURES),
				create_content_chains(synthetic_df, SUBTOPICS),
			],
			axis=1,
		)
		X_syn = X_syn.loc[:, ~X_syn.columns.duplicated()]
		X_syn = X_syn.reindex(columns=m2_columns, fill_value=0)

		# Predict
		combo_df["predicted_score"] = model_m2.predict(X_syn)

		# Compute observed aggregates
		group_cols = [
			"structure_1", "content_1",
			"structure_2", "content_2",
			"structure_3", "content_3",
		]
		observed = (
			df.groupby(group_cols)[RESPONSE_VARIABLE]
			.agg(["mean", "count"])
			.rename(columns={"mean": "actual_score_mean", "count": "n_observed"})
			.reset_index()
		)

		# Left-join
		combo_df = combo_df.merge(observed, on=group_cols, how="left")

		# Sort by predicted score descending
		combo_df = combo_df.sort_values("predicted_score", ascending=False)

		subdir = ARGUMENT_DATA_DIR / TOPIC / f"synthesis_{ds_name}"
		subdir.mkdir(parents=True, exist_ok=True)
		out_path = subdir / f"m2_trajectory_rankings_{ds_name}.csv"
		combo_df.to_csv(out_path, index=False)
		logger.info(f"Saved: {out_path} ({len(combo_df)} rows)")


def plot_unified_alpha_selection(cross_topic_results):
	"""Unified alpha selection: 5 rows (topics) x 3 columns (synthesis)."""
	topic_keys = list(TOPIC_SUBTOPICS_FILES.keys())
	synthesis_modes = ["strict", "faithful", "restructured"]
	synthesis_labels = {"strict": "Strict", "faithful": "Faithful", "restructured": "Restructured"}
	n_topics = len(topic_keys)
	n_synth = len(synthesis_modes)

	fig, axes = plt.subplots(
		n_topics, n_synth, figsize=(12, 2 * n_topics), sharex=True, sharey=True
	)

	for row, topic_key in enumerate(topic_keys):
		display_name = TOPIC_DISPLAY_NAMES.get(topic_key, topic_key)
		for col, synth in enumerate(synthesis_modes):
			ax = axes[row, col]
			topic_results = cross_topic_results.get(topic_key, {})
			result = topic_results.get(synth)

			if result is None:
				ax.set_visible(False)
				continue

			alpha_df = result["alpha_results"]
			base_color = DATASET_PALETTES[synth]["base"]

			ax.errorbar(
				alpha_df["alpha"],
				alpha_df["mean"],
				yerr=alpha_df["std"],
				marker="o",
				capsize=3,
				color=base_color,
				linewidth=1.5,
				markersize=4,
			)

			best_idx = alpha_df["mean"].idxmax()
			best = alpha_df.loc[best_idx]
			ax.scatter(
				[best["alpha"]],
				[best["mean"]],
				marker="*",
				s=150,
				color=base_color,
				zorder=5,
			)

			ax.set_xscale("log")
			ax.spines["top"].set_visible(False)
			ax.spines["right"].set_visible(False)
			ax.grid(True, alpha=0.3, zorder=0, axis="y")
			ax.tick_params(axis="both", labelsize=9)

			# Column titles on first row
			if row == 0:
				ax.set_title(synthesis_labels[synth], fontweight="bold", fontsize=12)

			# Row labels on first column
			if col == 0:
				ax.set_ylabel(display_name, fontsize=10)

			# X-axis label on last row
			if row == n_topics - 1:
				ax.set_xlabel("Alpha (log scale)", fontsize=9)

	plt.tight_layout()
	filename = "unified_m2_alpha_selection_rank.pdf"
	unified_dir = SCRIPT_DIR / "figures" / "predictability"
	unified_dir.mkdir(parents=True, exist_ok=True)
	plt.savefig(unified_dir / filename, dpi=300, bbox_inches="tight")
	plt.close()
	logger.info(f"Saved: {filename}")
	_copy_to_paper(unified_dir, filename)


def plot_unified_feature_categories(cross_topic_results):
	"""Unified feature categories: 5 rows (topics) x 3 columns (synthesis)."""
	topic_keys = list(TOPIC_SUBTOPICS_FILES.keys())
	synthesis_modes = ["strict", "faithful", "restructured"]
	synthesis_labels = {"strict": "Strict", "faithful": "Faithful", "restructured": "Restructured"}
	n_topics = len(topic_keys)
	n_synth = len(synthesis_modes)

	category_colors = {
		"Structure Position": "#e74c3c",
		"Content Position": "#3498db",
		"Position Interactions": "#2ecc71",
		"Structure Chains (len-2)": "#f39c12",
		"Content Chains (len-2)": "#9b59b6",
	}
	categories = list(category_colors.keys())

	fig, axes = plt.subplots(
		n_topics, n_synth, figsize=(12, 2 * n_topics), sharex=True, sharey=True
	)

	for row, topic_key in enumerate(topic_keys):
		display_name = TOPIC_DISPLAY_NAMES.get(topic_key, topic_key)
		for col, synth in enumerate(synthesis_modes):
			ax = axes[row, col]
			topic_results = cross_topic_results.get(topic_key, {})
			result = topic_results.get(synth)

			if result is None:
				ax.set_visible(False)
				continue

			feat_cats = result["feature_categories"]
			bottom = 0
			for cat in categories:
				count = len(feat_cats.get(cat, []))
				ax.bar(0, count, bottom=bottom, color=category_colors[cat], width=0.6)
				if count > 0:
					ax.text(
						0, bottom + count / 2, str(count),
						ha="center", va="center", fontsize=9, fontweight="bold",
					)
				bottom += count

			ax.set_xticks([])
			ax.spines["top"].set_visible(False)
			ax.spines["right"].set_visible(False)
			ax.grid(True, alpha=0.3, zorder=0, axis="y")
			ax.tick_params(axis="both", labelsize=9)

			# Column titles on first row
			if row == 0:
				ax.set_title(synthesis_labels[synth], fontweight="bold", fontsize=12)

			# Row labels on first column
			if col == 0:
				ax.set_ylabel(display_name, fontsize=10)

	# Shared legend at the bottom
	from matplotlib.patches import Patch
	legend_handles = [Patch(facecolor=category_colors[c], label=c) for c in categories]
	fig.legend(
		handles=legend_handles,
		loc="lower center",
		bbox_to_anchor=(0.5, -0.02),
		ncol=len(categories),
		fontsize=10,
		frameon=False,
	)

	plt.tight_layout(rect=[0, 0.05, 1, 1])
	filename = "unified_m2_feature_categories_rank.pdf"
	unified_dir = SCRIPT_DIR / "figures" / "predictability"
	unified_dir.mkdir(parents=True, exist_ok=True)
	plt.savefig(unified_dir / filename, dpi=300, bbox_inches="tight")
	plt.close()
	logger.info(f"Saved: {filename}")
	_copy_to_paper(unified_dir, filename)


def _copy_to_paper(source_dir, filename):
	"""Copy a figure from source_dir to the paper figures directory."""
	paper_fig_dir = SCRIPT_DIR / ".." / ".." / "paper" / "figures" / "results" / "argument_generation"
	paper_fig_dir.mkdir(parents=True, exist_ok=True)
	src = source_dir / filename
	if src.exists():
		shutil.copy2(src, paper_fig_dir / filename)
		logger.info(f"Copied {filename} to {paper_fig_dir}")


def _run_all_topics():
	"""Run analysis for all topics and generate unified cross-topic figure."""
	global TOPIC, DATASETS, SUBTOPICS, FIGURES_DIR



	cross_topic_results = {}

	for topic_key, subtopics_file in TOPIC_SUBTOPICS_FILES.items():
		logger.info("\n" + "#" * 70)
		logger.info(f"TOPIC: {TOPIC_DISPLAY_NAMES.get(topic_key, topic_key)}")
		logger.info("#" * 70)

		# Set up globals for this topic
		TOPIC = topic_key
		with open(action_space_dir / subtopics_file) as f:
			SUBTOPICS = list(json.load(f)["choices"].keys())
		logger.info(f"Subtopics ({len(SUBTOPICS)}): {SUBTOPICS}")

		FIGURES_DIR = SCRIPT_DIR / "figures" / "predictability" / TOPIC
		FIGURES_DIR.mkdir(parents=True, exist_ok=True)

		DATASETS = {
			"strict": {
				"file": ARGUMENT_DATA_DIR / TOPIC / "synthesis_strict" / "pairwise_comparisons_bt_scores.csv",
				"color": "#3498db",
				"label": "Strict",
			},
			"faithful": {
				"file": ARGUMENT_DATA_DIR / TOPIC / "synthesis_faithful" / "pairwise_comparisons_bt_scores.csv",
				"color": "#2ecc71",
				"label": "Faithful",
			},
			"restructured": {
				"file": ARGUMENT_DATA_DIR / TOPIC / "synthesis_restructured" / "pairwise_comparisons_bt_scores.csv",
				"color": "#e74c3c",
				"label": "Restructured",
			},
		}

		# Run with length control
		all_results = {}
		for name, config in DATASETS.items():
			all_results[name] = analyze_dataset(name, config)

		cross_topic_results[topic_key] = all_results

		# Generate per-topic figures
		plot_main_figure(all_results)

		# Generate per-topic LaTeX table (pass same results for both args)
		generate_latex_table(all_results, all_results, topic_key=topic_key)

	# Generate unified cross-topic figures
	plot_unified_cross_topic(cross_topic_results)
	plot_strict_cross_topic(cross_topic_results)
	plot_cross_topic_by_synthesis(cross_topic_results)

	# Generate unified multi-topic LaTeX table (pass same results for both args)
	generate_unified_latex_table(cross_topic_results, cross_topic_results)

	# Generate unified 5x3 length histogram
	plot_unified_length_histograms()

	# Generate unified alpha selection and feature categories
	plot_unified_alpha_selection(cross_topic_results)
	plot_unified_feature_categories(cross_topic_results)

	# Copy unified figures to paper directory
	paper_fig_dir = SCRIPT_DIR / ".." / ".." / "paper" / "figures" / "results" / "argument_generation"
	paper_fig_dir.mkdir(parents=True, exist_ok=True)
	unified_fig_dir = SCRIPT_DIR / "figures" / "predictability"
	for fig_name in [
		"unified_cross_topic_test_r2_rank_arg_length.pdf",
		"strict_cross_topic_test_r2_rank_arg_length.pdf",
		"cross_topic_by_synthesis_test_r2_rank_arg_length.pdf",
	]:
		src = unified_fig_dir / fig_name
		if src.exists():
			shutil.copy2(src, paper_fig_dir / fig_name)
			logger.info(f"Copied {fig_name} to {paper_fig_dir}")


def main():
	global TOPIC, DATASETS, SUBTOPICS, FIGURES_DIR

	parser = argparse.ArgumentParser(description="M1 vs M2 Summary Analysis")
	parser.add_argument(
		"--topic",
		type=str,
		default="single_use_plastic_specific_subtopics",
		help="Topic subdirectory name, or 'all' to run all topics and generate unified figure.",
	)
	args = parser.parse_args()

	if args.topic == "all":
		_run_all_topics()
		return

	TOPIC = args.topic

	# Load subtopics dynamically based on topic
	subtopics_file = TOPIC_SUBTOPICS_FILES.get(TOPIC)
	if subtopics_file is None:
		raise ValueError(
			f"Unknown topic: {TOPIC}. Must be one of: {list(TOPIC_SUBTOPICS_FILES.keys())}"
		)
	with open(action_space_dir / subtopics_file) as f:
		SUBTOPICS = list(json.load(f)["choices"].keys())
	logger.info(f"Subtopics ({len(SUBTOPICS)}): {SUBTOPICS}")

	# Per-topic figures directory
	FIGURES_DIR = SCRIPT_DIR / "figures" / "predictability" / TOPIC
	FIGURES_DIR.mkdir(parents=True, exist_ok=True)

	DATASETS = {
		"strict": {
			"file": ARGUMENT_DATA_DIR / TOPIC / "synthesis_strict" / "pairwise_comparisons_bt_scores.csv",
			"color": "#3498db",
			"label": "Strict",
		},
		"faithful": {
			"file": ARGUMENT_DATA_DIR / TOPIC / "synthesis_faithful" / "pairwise_comparisons_bt_scores.csv",
			"color": "#2ecc71",
			"label": "Faithful",
		},
		"restructured": {
			"file": ARGUMENT_DATA_DIR / TOPIC / "synthesis_restructured" / "pairwise_comparisons_bt_scores.csv",
			"color": "#e74c3c",
			"label": "Restructured",
		},
	}

	logger.info("=" * 70)
	logger.info("M1 vs M2 Summary Analysis")
	logger.info("=" * 70)
	logger.info(f"\nResponse Variable: {RESPONSE_VARIABLE}")
	if RESPONSE_VARIABLE == "rank_score":
		logger.info("  (rank_score = rank of bt_score / n_arguments)")
	logger.info("\nM1 = Topic Model Baseline (presence features - what elements were used)")
	logger.info("M2 = Sequential Model (position + interactions + chains - when and how)")

	all_results = {}
	for name, config in DATASETS.items():
		all_results[name] = analyze_dataset(name, config)

	# Export CSVs
	logger.info("\n" + "=" * 70)
	logger.info("Exporting CSVs")
	logger.info("=" * 70)
	export_coefficients_csv(all_results)
	export_trajectory_predictions(all_results)

	# Generate visualizations
	logger.info("\n" + "=" * 70)
	logger.info("Generating Visualizations")
	logger.info("=" * 70)

	plot_length_histograms()  # Argument length distributions
	plot_main_figure(all_results)
	plot_top_features(all_results)
	plot_feature_categories(all_results)
	plot_alpha_selection(all_results)
	# Generate LaTeX table (pass same results for both args)
	generate_latex_table(all_results, all_results)

	# Print summary
	logger.info("\n" + "=" * 70)
	logger.info("SUMMARY: Topic Model (M1c) vs Sequential Model (M2)")
	logger.info("=" * 70)

	for ds in DATASETS:
		result = all_results[ds]
		m1c = result["test"]["M1c"]
		m2 = result["test"]["M2"]
		delta = m2["point"] - m1c["point"]
		pct = (delta / m1c["point"] * 100) if m1c["point"] > 0 else 0
		logger.info(f"\n{result['label']} (N={result['n_train']} train, {result['n_test']} test):")
		logger.info(f"  M1c (Topic Baseline): R² = {m1c['point']:.4f} [{m1c['ci_lower']:.4f}, {m1c['ci_upper']:.4f}]")
		logger.info(f"  M2 (Sequential):      R² = {m2['point']:.4f} [{m2['ci_lower']:.4f}, {m2['ci_upper']:.4f}]")
		logger.info(f"  Improvement:          ΔR² = {delta:+.4f} ({pct:+.1f}%)")
		logger.info(f"  M2 Best Alpha:        {result['best_alpha']}")
		logger.info(f"  M2 Selected Features: {result['n_features']['M2_selected']}/{result['n_features']['M2']}")

	# Key findings
	logger.info("\n" + "=" * 70)
	logger.info("KEY FINDINGS")
	logger.info("=" * 70)

	avg_m1c = np.mean([all_results[ds]["test"]["M1c"]["point"] for ds in DATASETS])
	avg_m2 = np.mean([all_results[ds]["test"]["M2"]["point"] for ds in DATASETS])
	avg_delta = avg_m2 - avg_m1c
	avg_pct = (avg_delta / avg_m1c * 100) if avg_m1c > 0 else 0

	logger.info(f"\nAverage across all datasets:")
	logger.info(f"  M1c (Topic Baseline): R² = {avg_m1c:.4f}")
	logger.info(f"  M2 (Sequential):      R² = {avg_m2:.4f}")
	logger.info(f"  Improvement:          ΔR² = {avg_delta:+.4f} ({avg_pct:+.1f}%)")

	# Which components matter?
	logger.info("\nStructure vs Content contribution (M1a vs M1b):")
	for ds in DATASETS:
		result = all_results[ds]
		m1a = result["test"]["M1a"]
		m1b = result["test"]["M1b"]
		logger.info(
			f"  {result['label']}: Structure R²={m1a['point']:.4f} [{m1a['ci_lower']:.4f}, {m1a['ci_upper']:.4f}], "
			f"Content R²={m1b['point']:.4f} [{m1b['ci_lower']:.4f}, {m1b['ci_upper']:.4f}]"
		)

	logger.info("\n" + "=" * 70)
	logger.info("Analysis complete. All figures saved.")
	logger.info("=" * 70)


if __name__ == "__main__":
	main()
