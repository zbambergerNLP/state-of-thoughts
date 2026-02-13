"""
M1 vs M2 Summary Analysis for Large-Scale Explainability Experiment

Compares M1 (presence-based, topic model baseline) vs M2 (full LASSO with sequential features)
across 3 synthesis types (strict, faithful, restructured) with ~5000 arguments each (15,000 total).

Expected Data Structure:
    experiments/argument_generation/explainability/
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
    python experiments/argument_generation/explainability/m1_vs_m2_analysis.py

Output:
    All figures are saved to the explainability/ directory.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

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

# Length control flag: when True, includes argument_length in all models and adds M0 baseline
INCLUDE_LENGTH_CONTROL = True  # Set to True to include argument_length and M0 baseline

# Suffix for figure filenames based on response variable
FIGURE_SUFFIX = "_rank" if RESPONSE_VARIABLE == "rank_score" else ""

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
action_space_dir = SCRIPT_DIR / ".." / "action_space"

with open(action_space_dir / "causal_structures.json") as f:
	STRUCTURES = list(json.load(f)["choices"].keys())

with open(action_space_dir / "causal_subtopics.json") as f:
	SUBTOPICS = list(json.load(f)["choices"].keys())

print(f"Structures ({len(STRUCTURES)}): {STRUCTURES}")
print(f"Subtopics ({len(SUBTOPICS)}): {SUBTOPICS}")

# Dataset configuration (ordered by flexibility: least to most)
# File paths are relative to SCRIPT_DIR (explainability/ directory)
DATASETS = {
	"strict": {
		"file": SCRIPT_DIR / "synthesis_strict" / "pairwise_comparisons_bt_scores.csv",
		"color": "#e74c3c",  # Red
		"label": "Strict",
	},
	"faithful": {
		"file": SCRIPT_DIR / "synthesis_faithful" / "pairwise_comparisons_bt_scores.csv",
		"color": "#2ecc71",  # Green
		"label": "Faithful",
	},
	"restructured": {
		"file": SCRIPT_DIR / "synthesis_restructured" / "pairwise_comparisons_bt_scores.csv",
		"color": "#3498db",  # Blue
		"label": "Restructured",
	},
}

# Dataset color palettes with model-specific shades (light to dark: M0 -> M1a -> M2)
# - Strict: shades of red (least flexible)
# - Faithful: shades of green (medium flexibility)
# - Restructured: shades of blue (most flexible)
DATASET_PALETTES = {
	"strict": {
		"M0": "#ffb3b3",  # Light red (length-only baseline) - more visible
		"M1a": "#ff8080",  # Medium-light red
		"M1b": "#ff4d4d",  # Medium red
		"M1c": "#e60000",  # Dark red
		"M2": "#990000",  # Darkest red
		"base": "#e74c3c",  # Base color for single-value plots
	},
	"faithful": {
		"M0": "#b3ffb3",  # Light green (length-only baseline) - more visible
		"M1a": "#80ff80",  # Medium-light green
		"M1b": "#4dcc4d",  # Medium green
		"M1c": "#33a333",  # Dark green
		"M2": "#1a751a",  # Darkest green
		"base": "#2ecc71",  # Base color for single-value plots
	},
	"restructured": {
		"M0": "#b3d9ff",  # Light blue (length-only baseline) - more visible
		"M1a": "#80c1ff",  # Medium-light blue
		"M1b": "#4da6ff",  # Medium blue
		"M1c": "#1a8cff",  # Dark blue
		"M2": "#0059b3",  # Darkest blue
		"base": "#3498db",  # Base color for single-value plots
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
	"""M1a: Structure presence only (+ length control if enabled)."""
	features = [create_structure_presence(df, STRUCTURES)]
	if INCLUDE_LENGTH_CONTROL:
		features.append(create_length_feature(df))
	return pd.concat(features, axis=1)


def create_m1b_features(df):
	"""M1b: Content presence only (+ length control if enabled)."""
	features = [create_content_presence(df, SUBTOPICS)]
	if INCLUDE_LENGTH_CONTROL:
		features.append(create_length_feature(df))
	return pd.concat(features, axis=1)


def create_m1c_features(df):
	"""M1c: Structure + Content presence (full topic model baseline) (+ length control if enabled)."""
	features = [
		create_structure_presence(df, STRUCTURES),
		create_content_presence(df, SUBTOPICS),
	]
	if INCLUDE_LENGTH_CONTROL:
		features.append(create_length_feature(df))
	return pd.concat(features, axis=1)


def create_m2_features(df):
	"""M2: Full sequential feature set (+ length control if enabled).

	Includes:
	1. Position main effects (structure + content at each step)
	2. Position interactions (structure x content at same step)
	3. Structure chains (transitions between steps)
	4. Content chains (transitions between steps)
	5. Argument length (control variable) - if INCLUDE_LENGTH_CONTROL is True
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

	# Add continuous features if enabled (not filtered by sum > 0)
	if INCLUDE_LENGTH_CONTROL:
		features = pd.concat([binary_features, create_length_feature(df)], axis=1)
	else:
		features = binary_features
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
	print(f"\n{'=' * 60}")
	print(f"Processing: {config['label']} ({config['file']})")
	print(f"{'=' * 60}")

	# Load and prepare data
	df = pd.read_csv(config["file"])
	n_original = len(df)
	df = df.drop_duplicates(subset=["final_argument"])
	n_after_dedup = len(df)
	df = prepare_data(df)
	print(f"Loaded {n_original} samples, {n_after_dedup} after removing duplicates")

	# Train/test split
	train_df, test_df = train_test_split(
		df, test_size=TEST_SIZE, random_state=RANDOM_STATE
	)
	y_train = train_df[RESPONSE_VARIABLE]
	y_test = test_df[RESPONSE_VARIABLE]
	print(f"Train: {len(train_df)}, Test: {len(test_df)}")
	print(f"Response variable: {RESPONSE_VARIABLE}")
	print(f"Length control: {'enabled' if INCLUDE_LENGTH_CONTROL else 'disabled'}")

	# Create feature sets
	X_m1a_train = create_m1a_features(train_df)
	X_m1b_train = create_m1b_features(train_df)
	X_m1c_train = create_m1c_features(train_df)
	X_m2_train = create_m2_features(train_df)

	# Remove zero-variance binary columns (but keep continuous features like argument_length)
	def filter_zero_variance(X):
		if INCLUDE_LENGTH_CONTROL:
			binary_cols = [c for c in X.columns if c != "argument_length"]
			keep_binary = [c for c in binary_cols if X[c].sum() > 0]
			keep_cols = keep_binary + (["argument_length"] if "argument_length" in X.columns else [])
			return X[keep_cols]
		else:
			return X.loc[:, X.sum() > 0]

	X_m1a_train = filter_zero_variance(X_m1a_train)
	X_m1b_train = filter_zero_variance(X_m1b_train)
	X_m1c_train = filter_zero_variance(X_m1c_train)

	# M0 features (length only) - only created when INCLUDE_LENGTH_CONTROL is True
	if INCLUDE_LENGTH_CONTROL:
		X_m0_train = create_m0_features(train_df)

	print(f"\nFeature counts:")
	if INCLUDE_LENGTH_CONTROL:
		print(f"  M0 (Length Only):         {X_m0_train.shape[1]}")
	print(f"  M1a (Structure Presence): {X_m1a_train.shape[1]}")
	print(f"  M1b (Content Presence):   {X_m1b_train.shape[1]}")
	print(f"  M1c (Both Presence):      {X_m1c_train.shape[1]}")
	print(f"  M2 (Full Sequential):     {X_m2_train.shape[1]}")

	# Run cross-validation
	print("\nRunning 10-fold cross-validation...")

	# M0 model (OLS, length only) - only when INCLUDE_LENGTH_CONTROL is True
	if INCLUDE_LENGTH_CONTROL:
		cv_m0 = run_cv_ols(X_m0_train, y_train)

	# M1 models (OLS)
	cv_m1a = run_cv_ols(X_m1a_train, y_train)
	cv_m1b = run_cv_ols(X_m1b_train, y_train)
	cv_m1c = run_cv_ols(X_m1c_train, y_train)

	# M2 model (LASSO with alpha selection)
	cv_m2 = run_cv_lasso(X_m2_train, y_train)

	print(f"\nCV Results:")
	if INCLUDE_LENGTH_CONTROL:
		print(f"  M0 (Length Only):         R² = {cv_m0['mean']:.4f} ± {cv_m0['std']:.4f}")
	print(
		f"  M1a (Structure Presence): R² = {cv_m1a['mean']:.4f} ± {cv_m1a['std']:.4f}"
	)
	print(f"  M1b (Content Presence):   R² = {cv_m1b['mean']:.4f} ± {cv_m1b['std']:.4f}")
	print(f"  M1c (Both Presence):      R² = {cv_m1c['mean']:.4f} ± {cv_m1c['std']:.4f}")
	print(
		f"  M2 (Full LASSO):          R² = {cv_m2['mean']:.4f} ± {cv_m2['std']:.4f} (α={cv_m2['best_alpha']}, n={cv_m2['n_selected']})"
	)

	# Fit final models on full training set
	if INCLUDE_LENGTH_CONTROL:
		model_m0 = LinearRegression().fit(X_m0_train, y_train)
	model_m1a = LinearRegression().fit(X_m1a_train, y_train)
	model_m1b = LinearRegression().fit(X_m1b_train, y_train)
	model_m1c = LinearRegression().fit(X_m1c_train, y_train)
	model_m2 = Lasso(alpha=cv_m2["best_alpha"], max_iter=10000).fit(X_m2_train, y_train)

	# Prepare test features (aligned with training columns)
	if INCLUDE_LENGTH_CONTROL:
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
	if INCLUDE_LENGTH_CONTROL:
		test_m0 = bootstrap_r2(y_test, model_m0.predict(X_m0_test))
	test_m1a = bootstrap_r2(y_test, model_m1a.predict(X_m1a_test))
	test_m1b = bootstrap_r2(y_test, model_m1b.predict(X_m1b_test))
	test_m1c = bootstrap_r2(y_test, model_m1c.predict(X_m1c_test))
	test_m2 = bootstrap_r2(y_test, model_m2.predict(X_m2_test))

	print(f"\nTest Set Results (with 95% CI):")
	if INCLUDE_LENGTH_CONTROL:
		print(f"  M0 (Length Only):         R² = {test_m0['point']:.4f} [{test_m0['ci_lower']:.4f}, {test_m0['ci_upper']:.4f}]")
	print(f"  M1a (Structure Presence): R² = {test_m1a['point']:.4f} [{test_m1a['ci_lower']:.4f}, {test_m1a['ci_upper']:.4f}]")
	print(f"  M1b (Content Presence):   R² = {test_m1b['point']:.4f} [{test_m1b['ci_lower']:.4f}, {test_m1b['ci_upper']:.4f}]")
	print(f"  M1c (Both Presence):      R² = {test_m1c['point']:.4f} [{test_m1c['ci_lower']:.4f}, {test_m1c['ci_upper']:.4f}]")
	print(f"  M2 (Full LASSO):          R² = {test_m2['point']:.4f} [{test_m2['ci_lower']:.4f}, {test_m2['ci_upper']:.4f}]")
	print(f"  Improvement (M2 - M1c):   {test_m2['point'] - test_m1c['point']:+.4f}")

	# Bootstrap CIs for coefficients (optional, controlled by USE_BOOTSTRAP_COEF_CI)
	coef_bootstrap = None
	if USE_BOOTSTRAP_COEF_CI:
		print("  Computing bootstrap CIs for coefficients...")
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

	print(f"\nM2 Selected Features by Category:")
	for cat, feats in sorted(feature_categories.items()):
		print(f"  {cat}: {len(feats)}")

	# Build result dict
	n_features = {
		"M1a": X_m1a_train.shape[1],
		"M1b": X_m1b_train.shape[1],
		"M1c": X_m1c_train.shape[1],
		"M2": X_m2_train.shape[1],
		"M2_selected": int(cv_m2["n_selected"]),
	}
	cv_results = {
		"M1a": cv_m1a,
		"M1b": cv_m1b,
		"M1c": cv_m1c,
		"M2": cv_m2,
	}
	test_results = {
		"M1a": test_m1a,
		"M1b": test_m1b,
		"M1c": test_m1c,
		"M2": test_m2,
	}

	# Add M0 results if length control is enabled
	if INCLUDE_LENGTH_CONTROL:
		n_features["M0"] = X_m0_train.shape[1]
		cv_results["M0"] = cv_m0
		test_results["M0"] = test_m0

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
	filename = f"argument_length_histograms{FIGURE_SUFFIX}.png"
	plt.savefig(SCRIPT_DIR / filename, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved: {filename}")


def plot_cv_comparison(all_results):
	"""Figure 1: CV Performance Comparison."""
	from matplotlib.patches import Patch

	fig, ax = plt.subplots(figsize=(12, 6))

	models = ["M0", "M1a", "M1b", "M1c", "M2"] if INCLUDE_LENGTH_CONTROL else ["M1a", "M1b", "M1c", "M2"]
	x = np.arange(len(DATASETS))
	width = 0.15 if INCLUDE_LENGTH_CONTROL else 0.2
	datasets_list = list(DATASETS.keys())

	for i, model in enumerate(models):
		means = [all_results[ds]["cv"][model]["mean"] for ds in DATASETS]
		stds = [all_results[ds]["cv"][model]["std"] for ds in DATASETS]

		# Use dataset-specific colors for each model
		colors = [DATASET_PALETTES[ds][model] for ds in datasets_list]
		ax.bar(
			x + i * width, means, width, yerr=stds, capsize=3,
			color=colors, edgecolor="white", linewidth=0.5
		)

	ax.set_xlabel("Dataset")
	ax.set_ylabel("CV R²")
	ax.set_title(
		"Topic Model Baseline (M1) vs Sequential Model (M2)\n10-fold Cross-Validation \n(Intervals of observed range)",
		fontweight="bold",
	)
	ax.set_xticks(x + (len(models) - 1) * width / 2)
	ax.set_xticklabels([DATASETS[ds]["label"] for ds in DATASETS])

	# Create custom legend showing models (using gray shades for model legend)
	if INCLUDE_LENGTH_CONTROL:
		model_legend = [
			Patch(facecolor="#999999", edgecolor="black", label="M0 (Length Only)"),
			Patch(facecolor="#777777", edgecolor="black", label="M1a (Structure)"),
			Patch(facecolor="#555555", edgecolor="black", label="M1b (Content)"),
			Patch(facecolor="#333333", edgecolor="black", label="M1c (Both)"),
			Patch(facecolor="#111111", edgecolor="black", label="M2 (Sequential)"),
		]
	else:
		model_legend = [
			Patch(facecolor="#777777", edgecolor="black", label="M1a (Structure)"),
			Patch(facecolor="#555555", edgecolor="black", label="M1b (Content)"),
			Patch(facecolor="#333333", edgecolor="black", label="M1c (Both)"),
			Patch(facecolor="#111111", edgecolor="black", label="M2 (Sequential)"),
		]
	ax.legend(
		handles=model_legend,
		title="Model (light→dark)",
		loc="center left",
		bbox_to_anchor=(1.0, 0.5),
		fontsize=10,
		frameon=False,
	)
	apply_clean_style(ax)

	plt.tight_layout(rect=[0, 0, 0.82, 0.95])
	filename = f"m1_vs_m2_cv_comparison{FIGURE_SUFFIX}.png"
	plt.savefig(SCRIPT_DIR / filename, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved: {filename}")


def plot_test_comparison(all_results):
	"""Figure 2: Test Set Performance with 95% CI error bars."""
	fig, axes = plt.subplots(1, 2, figsize=(14, 5))
	datasets_list = list(DATASETS.keys())

	# Left panel: M1c vs M2 test R² with error bars
	ax = axes[0]
	x = np.arange(len(DATASETS))
	width = 0.35

	# Extract point estimates and CI bounds
	m1c_scores = [all_results[ds]["test"]["M1c"]["point"] for ds in DATASETS]
	m2_scores = [all_results[ds]["test"]["M2"]["point"] for ds in DATASETS]

	# Calculate asymmetric error bars (lower_err, upper_err)
	m1c_lower_err = [all_results[ds]["test"]["M1c"]["point"] - all_results[ds]["test"]["M1c"]["ci_lower"] for ds in DATASETS]
	m1c_upper_err = [all_results[ds]["test"]["M1c"]["ci_upper"] - all_results[ds]["test"]["M1c"]["point"] for ds in DATASETS]
	m2_lower_err = [all_results[ds]["test"]["M2"]["point"] - all_results[ds]["test"]["M2"]["ci_lower"] for ds in DATASETS]
	m2_upper_err = [all_results[ds]["test"]["M2"]["ci_upper"] - all_results[ds]["test"]["M2"]["point"] for ds in DATASETS]

	# Use dataset-specific colors for M1c and M2
	m1c_colors = [DATASET_PALETTES[ds]["M1c"] for ds in datasets_list]
	m2_colors = [DATASET_PALETTES[ds]["M2"] for ds in datasets_list]

	# Plot bars with error bars
	for i, ds in enumerate(datasets_list):
		ax.bar(x[i] - width / 2, m1c_scores[i], width, color=m1c_colors[i], edgecolor="white", linewidth=0.5,
			   yerr=[[m1c_lower_err[i]], [m1c_upper_err[i]]], capsize=3, error_kw={"ecolor": "black", "elinewidth": 1})
		ax.bar(x[i] + width / 2, m2_scores[i], width, color=m2_colors[i], edgecolor="white", linewidth=0.5,
			   yerr=[[m2_lower_err[i]], [m2_upper_err[i]]], capsize=3, error_kw={"ecolor": "black", "elinewidth": 1})

	ax.set_xlabel("Dataset")
	ax.set_ylabel("Test R²")
	ax.set_title("Test Set Performance: M1c vs M2\n(95% CI)", fontweight="bold")
	ax.set_xticks(x)
	ax.set_xticklabels([DATASETS[ds]["label"] for ds in DATASETS])

	# Custom legend for left/right bar position
	from matplotlib.patches import Patch
	ax.legend(
		handles=[
			Patch(facecolor="#888888", label="M1c (lighter shade)"),
			Patch(facecolor="#444444", label="M2 (darker shade)"),
		]
	)
	apply_clean_style(ax)

	# Add test N annotation
	for i, ds in enumerate(DATASETS):
		ax.annotate(
			f"Test N={all_results[ds]['n_test']}",
			xy=(x[i], -0.02),
			ha="center",
			fontsize=9,
			color="gray",
		)

	# Right panel: Delta (M2 - M1c) using dataset base colors with propagated uncertainty
	ax = axes[1]
	deltas = [all_results[ds]["test"]["M2"]["point"] - all_results[ds]["test"]["M1c"]["point"] for ds in DATASETS]

	# Propagate uncertainty: approximate combined std from bootstrap samples
	# For difference, use quadrature: std_delta ≈ sqrt(std_m2² + std_m1c²)
	delta_stds = [
		np.sqrt(all_results[ds]["test"]["M2"]["std"]**2 + all_results[ds]["test"]["M1c"]["std"]**2)
		for ds in DATASETS
	]
	# Convert to 95% CI (±1.96 * std for approximate normal)
	delta_errs = [1.96 * std for std in delta_stds]

	colors = [DATASET_PALETTES[ds]["base"] for ds in datasets_list]

	ax.bar(x, deltas, color=colors, edgecolor="white", linewidth=0.5,
		   yerr=delta_errs, capsize=4, error_kw={"ecolor": "black", "elinewidth": 1})
	ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
	ax.set_xlabel("Dataset")
	ax.set_ylabel("ΔR² (M2 - M1c)")
	ax.set_title(
		"Sequential Improvement Over Topic Model Baseline (95% CI)", fontweight="bold"
	)
	ax.set_xticks(x)
	ax.set_xticklabels([DATASETS[ds]["label"] for ds in DATASETS])
	apply_clean_style(ax)

	# Add percentage improvement
	for i, (ds, delta) in enumerate(zip(DATASETS, deltas)):
		m1c_score = all_results[ds]["test"]["M1c"]["point"]
		if m1c_score > 0:
			pct = (delta / m1c_score) * 100
			ax.annotate(
				f"{delta:+.4f}\n({pct:+.1f}%)",
				xy=(i, delta + delta_errs[i] if delta > 0 else delta - delta_errs[i]),
				ha="center",
				va="bottom" if delta > 0 else "top",
				fontsize=9,
			)

	plt.tight_layout()
	filename = f"m1_vs_m2_test_comparison{FIGURE_SUFFIX}.png"
	plt.savefig(SCRIPT_DIR / filename, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved: {filename}")


def plot_m1_breakdown(all_results):
	"""Figure 3: M1 Breakdown (structure vs content) with 95% CI error bars."""
	fig, ax = plt.subplots(figsize=(10, 6))

	models = ["M1a", "M1b", "M1c"]
	model_labels = ["M1a (Structure)", "M1b (Content)", "M1c (Both)"]
	x = np.arange(len(DATASETS))
	width = 0.25
	datasets_list = list(DATASETS.keys())

	for i, (model, label) in enumerate(zip(models, model_labels)):
		scores = [all_results[ds]["test"][model]["point"] for ds in DATASETS]
		# Calculate asymmetric error bars
		lower_errs = [all_results[ds]["test"][model]["point"] - all_results[ds]["test"][model]["ci_lower"] for ds in DATASETS]
		upper_errs = [all_results[ds]["test"][model]["ci_upper"] - all_results[ds]["test"][model]["point"] for ds in DATASETS]
		# Use dataset-specific colors for each model
		colors = [DATASET_PALETTES[ds][model] for ds in datasets_list]
		# Plot each bar individually with its color and error bar
		for j, ds in enumerate(datasets_list):
			ax.bar(x[j] + i * width, scores[j], width, color=colors[j], edgecolor="white", linewidth=0.5,
				   yerr=[[lower_errs[j]], [upper_errs[j]]], capsize=2, error_kw={"ecolor": "black", "elinewidth": 1})

	ax.set_xlabel("Dataset")
	ax.set_ylabel("Test R²")
	ax.set_title(
		"M1 Breakdown: Structure vs Content Presence Features (95% CI)", fontweight="bold"
	)
	ax.set_xticks(x + width)
	ax.set_xticklabels([DATASETS[ds]["label"] for ds in DATASETS])

	# Custom legend showing models (using gray shades)
	from matplotlib.patches import Patch
	model_legend = [
		Patch(facecolor="#cccccc", edgecolor="black", label="M1a (Structure)"),
		Patch(facecolor="#999999", edgecolor="black", label="M1b (Content)"),
		Patch(facecolor="#666666", edgecolor="black", label="M1c (Both)"),
	]
	ax.legend(handles=model_legend, title="Model (light→dark)")
	apply_clean_style(ax)

	plt.tight_layout()
	filename = f"m1_breakdown{FIGURE_SUFFIX}.png"
	plt.savefig(SCRIPT_DIR / filename, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved: {filename}")


def plot_feature_categories(all_results):
	"""Figure 4: M2 Feature Category Breakdown as Stacked Bar Chart."""
	fig, ax = plt.subplots(figsize=(10, 6))

	category_colors = {
		"Structure Position": "#e74c3c",
		"Content Position": "#3498db",
		"Position Interactions": "#2ecc71",
		"Structure Chains (len-2)": "#f39c12",
		"Content Chains (len-2)": "#9b59b6",
	}

	# Define categories in stacking order
	categories = [
		"Structure Position",
		"Content Position",
		"Position Interactions",
		"Structure Chains (len-2)",
		"Content Chains (len-2)",
	]

	# Build data matrix: datasets x categories
	datasets_list = list(DATASETS.keys())
	x = np.arange(len(datasets_list))

	# Stack bars from bottom
	bottom = np.zeros(len(datasets_list))

	for cat in categories:
		counts = [
			len(all_results[ds]["feature_categories"].get(cat, []))
			for ds in datasets_list
		]
		bars = ax.bar(
			x, counts, bottom=bottom, label=cat, color=category_colors[cat]
		)

		# Add count labels on segments (if count > 0)
		for i, (count, b) in enumerate(zip(counts, bottom)):
			if count > 0:
				ax.text(
					i, b + count / 2, str(count),
					ha="center", va="center", fontsize=9, fontweight="bold"
				)

		bottom = bottom + np.array(counts)

	ax.set_xticks(x)
	ax.set_xticklabels([DATASETS[ds]["label"] for ds in datasets_list])
	ax.set_xlabel("Synthesis Type")
	ax.set_ylabel("Number of Selected Features")
	ax.legend(
		loc="center left",
		bbox_to_anchor=(1.0, 0.5),
		fontsize=10,
		frameon=False,
	)
	ax.set_title("LASSO: Selected Features by Category", fontweight="bold")
	apply_clean_style(ax)

	plt.tight_layout(rect=[0, 0, 0.82, 0.95])
	filename = f"m2_feature_categories{FIGURE_SUFFIX}.png"
	plt.savefig(SCRIPT_DIR / filename, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved: {filename}")


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
	filename = f"m2_top_features{FIGURE_SUFFIX}.png"
	plt.savefig(SCRIPT_DIR / filename, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved: {filename}")


def plot_alpha_selection(all_results):
	"""Figure 6: Alpha Selection Curves."""
	fig, ax = plt.subplots(figsize=(10, 6))
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

		# Mark best alpha
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
	ax.set_xlabel("Alpha (log scale)")
	ax.set_ylabel("CV R²")
	ax.set_title("M2 LASSO: Alpha Selection via Cross-Validation", fontweight="bold")
	ax.legend()
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.grid(True, alpha=0.3, zorder=0, axis="y")

	plt.tight_layout()
	filename = f"m2_alpha_selection{FIGURE_SUFFIX}.png"
	plt.savefig(SCRIPT_DIR / filename, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved: {filename}")


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
	models = ["M0", "M1a", "M1b", "M1c", "M2"] if INCLUDE_LENGTH_CONTROL else ["M1a", "M1b", "M1c", "M2"]
	x = np.arange(len(DATASETS))
	width = 0.15 if INCLUDE_LENGTH_CONTROL else 0.2

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

	if INCLUDE_LENGTH_CONTROL:
		model_legend = [
			Patch(facecolor="#999999", edgecolor="black", label="M0 (Length Only)"),
			Patch(facecolor="#777777", edgecolor="black", label="M1a (Structure)"),
			Patch(facecolor="#555555", edgecolor="black", label="M1b (Content)"),
			Patch(facecolor="#333333", edgecolor="black", label="M1c (Both)"),
			Patch(facecolor="#111111", edgecolor="black", label="M2 (Sequential)"),
		]
	else:
		model_legend = [
			Patch(facecolor="#777777", edgecolor="black", label="M1a (Structure)"),
			Patch(facecolor="#555555", edgecolor="black", label="M1b (Content)"),
			Patch(facecolor="#333333", edgecolor="black", label="M1c (Both)"),
			Patch(facecolor="#111111", edgecolor="black", label="M2 (Sequential)"),
		]
	ax.legend(
		handles=model_legend,
		loc="upper left",
		fontsize=8,
		frameon=True,
	)
	apply_clean_style(ax)

	# -------------------------------------------------------------------------
	# Right panel: Test Set Performance with 95% CI
	# -------------------------------------------------------------------------
	ax = axes[1]
	models = ["M0", "M1a", "M1b", "M1c", "M2"] if INCLUDE_LENGTH_CONTROL else ["M1a", "M1b", "M1c", "M2"]
	x = np.arange(len(DATASETS))
	width = 0.15 if INCLUDE_LENGTH_CONTROL else 0.2

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
		loc="upper left",
		fontsize=8,
		frameon=True,
	)
	apply_clean_style(ax)

	plt.tight_layout()
	filename = f"m1_vs_m2_main_figure{FIGURE_SUFFIX}.png"
	plt.savefig(SCRIPT_DIR / filename, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved: {filename}")


def generate_latex_table(all_results):
	"""Generate LaTeX table code for the summary statistics.

	Returns the LaTeX code as a string and also prints it.
	"""
	lines = []
	lines.append(r"\begin{table}[htbp]")
	lines.append(r"\centering")
	lines.append(r"\caption{Model comparison: Topic model baseline (M1c) vs.\ sequential model (M2). "
				 r"R² values are reported with 95\% bootstrap confidence intervals.}")
	lines.append(r"\label{tab:m1_vs_m2_results}")
	lines.append(r"\begin{tabular}{lcccccc}")
	lines.append(r"\toprule")
	lines.append(r"Synthesis & N (train/test) & M1c R² & M2 R² & $\Delta$R² & \% Gain \\")
	lines.append(r"\midrule")

	for ds in DATASETS:
		result = all_results[ds]
		m1c = result["test"]["M1c"]
		m2 = result["test"]["M2"]
		delta = m2["point"] - m1c["point"]
		pct = (delta / m1c["point"] * 100) if m1c["point"] > 0 else 0

		# Format with CI
		m1c_str = f"{m1c['point']:.3f} [{m1c['ci_lower']:.3f}, {m1c['ci_upper']:.3f}]"
		m2_str = f"{m2['point']:.3f} [{m2['ci_lower']:.3f}, {m2['ci_upper']:.3f}]"

		lines.append(
			f"{result['label']} & {result['n_train']}/{result['n_test']} & "
			f"{m1c_str} & {m2_str} & {delta:+.3f} & {pct:+.1f}\\% \\\\"
		)

	lines.append(r"\bottomrule")
	lines.append(r"\end{tabular}")
	lines.append(r"\end{table}")

	latex_code = "\n".join(lines)

	print("\n" + "=" * 70)
	print("LaTeX Table Code")
	print("=" * 70)
	print(latex_code)

	return latex_code


def plot_summary_dashboard(all_results, show_ci=True):
	"""Figure 7: Summary Dashboard with optional 95% CI error bars."""
	from matplotlib.patches import Patch

	fig, axes = plt.subplots(2, 2, figsize=(14, 11))
	datasets_list = list(DATASETS.keys())

	# (1,1) Model comparison: M0, M1a, M1b, M1c, M2 test R² across datasets
	ax = axes[0, 0]
	models = ["M0", "M1a", "M1b", "M1c", "M2"] if INCLUDE_LENGTH_CONTROL else ["M1a", "M1b", "M1c", "M2"]
	x = np.arange(len(DATASETS))
	width = 0.15 if INCLUDE_LENGTH_CONTROL else 0.2

	for i, model in enumerate(models):
		scores = [all_results[ds]["test"][model]["point"] for ds in DATASETS]
		colors = [DATASET_PALETTES[ds][model] for ds in datasets_list]

		if show_ci:
			# Calculate asymmetric error bars
			lower_errs = [all_results[ds]["test"][model]["point"] - all_results[ds]["test"][model]["ci_lower"] for ds in DATASETS]
			upper_errs = [all_results[ds]["test"][model]["ci_upper"] - all_results[ds]["test"][model]["point"] for ds in DATASETS]
			# Plot each bar individually with its color and error bar
			for j, ds in enumerate(datasets_list):
				ax.bar(x[j] + i * width, scores[j], width, color=colors[j], edgecolor="white", linewidth=0.5,
					   yerr=[[lower_errs[j]], [upper_errs[j]]], capsize=2, error_kw={"ecolor": "black", "elinewidth": 0.8})
		else:
			# Plot without error bars
			for j, ds in enumerate(datasets_list):
				ax.bar(x[j] + i * width, scores[j], width, color=colors[j], edgecolor="white", linewidth=0.5)

	ax.set_xlabel("Dataset")
	ax.set_ylabel("Test R²")
	ci_label = " (95% CI)" if show_ci else ""
	ax.set_title(f"Model Comparison (Test Set{ci_label})", fontweight="bold")
	ax.set_xticks(x + (len(models) - 1) * width / 2)
	ax.set_xticklabels([DATASETS[ds]["label"] for ds in DATASETS])
	apply_clean_style(ax)

	# Custom legend showing models (using gray shades) - placed below the plot
	if INCLUDE_LENGTH_CONTROL:
		model_legend = [
			Patch(facecolor="#999999", edgecolor="black", label="M0"),
			Patch(facecolor="#777777", edgecolor="black", label="M1a"),
			Patch(facecolor="#555555", edgecolor="black", label="M1b"),
			Patch(facecolor="#333333", edgecolor="black", label="M1c"),
			Patch(facecolor="#111111", edgecolor="black", label="M2"),
		]
		ncol = 5
	else:
		model_legend = [
			Patch(facecolor="#777777", edgecolor="black", label="M1a"),
			Patch(facecolor="#555555", edgecolor="black", label="M1b"),
			Patch(facecolor="#333333", edgecolor="black", label="M1c"),
			Patch(facecolor="#111111", edgecolor="black", label="M2"),
		]
		ncol = 4
	ax.legend(handles=model_legend, title="Model (light→dark)", loc="upper center",
			  bbox_to_anchor=(0.5, -0.12), ncol=ncol, frameon=True)

	# (1,2) Feature count vs Test R² scatter
	ax = axes[0, 1]
	for ds in datasets_list:
		result = all_results[ds]
		base_color = DATASET_PALETTES[ds]["base"]

		if show_ci:
			# M1c point with error bar
			m1c_err = [[result["test"]["M1c"]["point"] - result["test"]["M1c"]["ci_lower"]],
					   [result["test"]["M1c"]["ci_upper"] - result["test"]["M1c"]["point"]]]
			ax.errorbar(result["n_features"]["M1c"], result["test"]["M1c"]["point"],
						yerr=m1c_err, fmt="o", markersize=8, color=base_color, alpha=0.6, capsize=3)
			# M2 point with error bar
			m2_err = [[result["test"]["M2"]["point"] - result["test"]["M2"]["ci_lower"]],
					  [result["test"]["M2"]["ci_upper"] - result["test"]["M2"]["point"]]]
			ax.errorbar(result["n_features"]["M2_selected"], result["test"]["M2"]["point"],
						yerr=m2_err, fmt="^", markersize=10, color=base_color, label=result["label"], capsize=3)
		else:
			# Without error bars
			ax.scatter(result["n_features"]["M1c"], result["test"]["M1c"]["point"],
					   marker="o", s=100, color=base_color, alpha=0.6)
			ax.scatter(result["n_features"]["M2_selected"], result["test"]["M2"]["point"],
					   marker="^", s=150, color=base_color, label=result["label"])

		# Connect them
		ax.plot([result["n_features"]["M1c"], result["n_features"]["M2_selected"]],
				[result["test"]["M1c"]["point"], result["test"]["M2"]["point"]],
				"--", color=base_color, alpha=0.5)

	ax.set_xlabel("Number of Features")
	ax.set_ylabel("Test R²")
	ax.set_title(f"Features vs Performance\n(o=M1c, ^=M2{ci_label})", fontweight="bold")
	ax.legend()
	apply_clean_style(ax)

	# (2,1) Improvement (M2 - M1c) per dataset
	ax = axes[1, 0]
	x = np.arange(len(DATASETS))
	deltas = [all_results[ds]["test"]["M2"]["point"] - all_results[ds]["test"]["M1c"]["point"] for ds in DATASETS]
	colors = [DATASET_PALETTES[ds]["base"] for ds in datasets_list]

	if show_ci:
		# Propagate uncertainty: std_delta ≈ sqrt(std_m2² + std_m1c²)
		delta_stds = [
			np.sqrt(all_results[ds]["test"]["M2"]["std"]**2 + all_results[ds]["test"]["M1c"]["std"]**2)
			for ds in DATASETS
		]
		delta_errs = [1.96 * std for std in delta_stds]
		ax.bar(x, deltas, color=colors, edgecolor="white", linewidth=0.5,
			   yerr=delta_errs, capsize=4, error_kw={"ecolor": "black", "elinewidth": 1})
	else:
		delta_errs = [0] * len(deltas)
		ax.bar(x, deltas, color=colors, edgecolor="white", linewidth=0.5)

	ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
	ax.set_xlabel("Dataset")
	ax.set_ylabel("ΔR² (M2 - M1c)")
	ax.set_title(f"Sequential Improvement Over Topic Baseline{ci_label}", fontweight="bold")
	ax.set_xticks(x)
	ax.set_xticklabels([DATASETS[ds]["label"] for ds in DATASETS])
	apply_clean_style(ax)

	for i, (delta, err) in enumerate(zip(deltas, delta_errs)):
		y_offset = delta + err if delta > 0 else delta - err
		if not show_ci:
			y_offset = delta
		ax.annotate(f"{delta:+.4f}", xy=(i, y_offset), ha="center",
					va="bottom" if delta > 0 else "top", fontsize=10, fontweight="bold")

	# (2,2) Summary table
	ax = axes[1, 1]
	ax.axis("off")

	table_data = []
	for ds in DATASETS:
		result = all_results[ds]
		m1c_pt = result["test"]["M1c"]["point"]
		m2_pt = result["test"]["M2"]["point"]
		delta = m2_pt - m1c_pt
		pct = (delta / m1c_pt * 100) if m1c_pt > 0 else 0

		if show_ci:
			# Format with point estimate and CI on separate lines
			m1c_str = f"{m1c_pt:.2f}\n[{result['test']['M1c']['ci_lower']:.2f}, {result['test']['M1c']['ci_upper']:.2f}]"
			m2_str = f"{m2_pt:.2f}\n[{result['test']['M2']['ci_lower']:.2f}, {result['test']['M2']['ci_upper']:.2f}]"
		else:
			m1c_str = f"{m1c_pt:.4f}"
			m2_str = f"{m2_pt:.4f}"

		table_data.append([
			result["label"],
			f"{result['n_train']}",
			f"{result['n_test']}",
			m1c_str,
			m2_str,
			f"{delta:+.3f}",
			f"{pct:+.1f}%",
		])

	if show_ci:
		col_labels = ["Dataset", "N Train", "N Test", "M1c R²\n[95% CI]", "M2 R²\n[95% CI]", "Δ", "% Gain"]
		table_scale = (1.4, 2.0)
		title_y = 0.92
		title = f"Summary Statistics\n(95% CI via {N_BOOTSTRAP} bootstrap iterations)"
	else:
		col_labels = ["Dataset", "N Train", "N Test", "M1c R²", "M2 R²", "Δ", "% Gain"]
		table_scale = (1.3, 1.5)
		title_y = 0.85
		title = "Summary Statistics"

	table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
	table.auto_set_font_size(False)
	table.set_fontsize(9)
	table.scale(*table_scale)
	ax.set_title(title, fontweight="bold", y=title_y)

	plt.tight_layout()
	ci_suffix = "_ci" if show_ci else "_no_ci"
	filename = f"m1_vs_m2_summary_dashboard{ci_suffix}{FIGURE_SUFFIX}.png"
	plt.savefig(SCRIPT_DIR / filename, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved: {filename}")


# =============================================================================
# Section 8: Main Execution
# =============================================================================


def main():
	print("=" * 70)
	print("M1 vs M2 Summary Analysis")
	print("=" * 70)
	print(f"\nResponse Variable: {RESPONSE_VARIABLE}")
	if RESPONSE_VARIABLE == "rank_score":
		print("  (rank_score = rank of bt_score / n_arguments)")
	print("\nM1 = Topic Model Baseline (presence features - what elements were used)")
	print("M2 = Sequential Model (position + interactions + chains - when and how)")
	print()

	# Analyze all datasets
	all_results = {}
	for name, config in DATASETS.items():
		all_results[name] = analyze_dataset(name, config)

	# Generate visualizations
	print("\n" + "=" * 70)
	print("Generating Visualizations")
	print("=" * 70)

	plot_length_histograms()  # Argument length distributions
	plot_main_figure(all_results)  # Main figure for paper body
	plot_cv_comparison(all_results)
	plot_test_comparison(all_results)
	plot_m1_breakdown(all_results)
	plot_feature_categories(all_results)
	plot_top_features(all_results)
	plot_alpha_selection(all_results)
	plot_summary_dashboard(all_results, show_ci=True)

	# Generate LaTeX table
	generate_latex_table(all_results)

	# Print summary
	print("\n" + "=" * 70)
	print("SUMMARY: Topic Model (M1c) vs Sequential Model (M2)")
	print("=" * 70)

	for ds in DATASETS:
		result = all_results[ds]
		m1c = result["test"]["M1c"]
		m2 = result["test"]["M2"]
		delta = m2["point"] - m1c["point"]
		pct = (delta / m1c["point"] * 100) if m1c["point"] > 0 else 0
		print(f"\n{result['label']} (N={result['n_train']} train, {result['n_test']} test):")
		print(f"  M1c (Topic Baseline): R² = {m1c['point']:.4f} [{m1c['ci_lower']:.4f}, {m1c['ci_upper']:.4f}]")
		print(f"  M2 (Sequential):      R² = {m2['point']:.4f} [{m2['ci_lower']:.4f}, {m2['ci_upper']:.4f}]")
		print(f"  Improvement:          ΔR² = {delta:+.4f} ({pct:+.1f}%)")
		print(f"  M2 Best Alpha:        {result['best_alpha']}")
		print(f"  M2 Selected Features: {result['n_features']['M2_selected']}/{result['n_features']['M2']}")

	# Key findings
	print("\n" + "=" * 70)
	print("KEY FINDINGS")
	print("=" * 70)

	avg_m1c = np.mean([all_results[ds]["test"]["M1c"]["point"] for ds in DATASETS])
	avg_m2 = np.mean([all_results[ds]["test"]["M2"]["point"] for ds in DATASETS])
	avg_delta = avg_m2 - avg_m1c
	avg_pct = (avg_delta / avg_m1c * 100) if avg_m1c > 0 else 0

	print(f"\nAverage across all datasets:")
	print(f"  M1c (Topic Baseline): R² = {avg_m1c:.4f}")
	print(f"  M2 (Sequential):      R² = {avg_m2:.4f}")
	print(f"  Improvement:          ΔR² = {avg_delta:+.4f} ({avg_pct:+.1f}%)")

	# Which components matter?
	print("\nStructure vs Content contribution (M1a vs M1b):")
	for ds in DATASETS:
		result = all_results[ds]
		m1a = result["test"]["M1a"]
		m1b = result["test"]["M1b"]
		print(
			f"  {result['label']}: Structure R²={m1a['point']:.4f} [{m1a['ci_lower']:.4f}, {m1a['ci_upper']:.4f}], "
			f"Content R²={m1b['point']:.4f} [{m1b['ci_lower']:.4f}, {m1b['ci_upper']:.4f}]"
		)

	print("\n" + "=" * 70)
	print("Analysis complete. All figures saved.")
	print("=" * 70)


if __name__ == "__main__":
	main()
