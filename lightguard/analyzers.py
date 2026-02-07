import pandas as pd
import numpy as np
import warnings
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import json
from datetime import datetime
from collections import defaultdict, deque
import time
import psutil
import gc
import copy

from lightguard import (
    DriftAlert,
    LSTMClassifier,
    ModelConfig,
    NumpyEncoder,
    seed_everything,
    TransformerClassifier,
    WindowStatistics,
)
from lightguard.framework import LightGuardFramework
from lightguard.data import ensure_timestamp_columns, normalize_cic_flowmeter_columns
from lightguard.io import download_dataset
from lightguard.utils import to_serializable
from lightguard.plotting import apply_journal_style, plot_with_shadow, save_figure
from lightguard.pipeline import run_full_analysis_phase1
from lightguard.reporting_phase1 import (
    visualize_concept_drift as _visualize_concept_drift,
    visualize_performance_decay as _visualize_performance_decay,
    generate_class_specific_analysis as _generate_class_specific_analysis,
)
from lightguard.reporting_phase2 import (
    visualize_lightguard_performance as _visualize_lightguard_performance,
    visualize_tradeoff_analysis as _visualize_tradeoff_analysis,
)
from lightguard.experiments import (
    run_ablation_study as _run_ablation_study,
    run_drift_threshold_ablation as _run_drift_threshold_ablation,
    run_continual_learning_comparison as _run_continual_learning_comparison,
)

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score, 
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.neighbors import KernelDensity

# XGBoost
from xgboost import XGBClassifier

# River for Adaptive Random Forest (ARF)
try:
    from river import forest as river_forest
    from river import metrics as river_metrics
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("River library not available. ARF baseline will be disabled.")

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

# Statistics
from scipy import stats
from scipy.spatial.distance import cdist
import scipy
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import mahalanobis

# Progress bar
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
seed_everything(RANDOM_SEED)


class DurabilityAnalyzer:
    """Phase I: Diagnosing the Durability Deception.

    Handles dataset loading, temporal re-framing, model training, longitudinal
    evaluation, and visualization of performance decay.
    """

    def __init__(self, data_path: Union[str, Path], output_dir: Union[str, Path]):
        """Initialize the durability analyzer.

        Args:
            data_path: Path to a CICFlowMeter-style CSV (e.g., Darknet.csv)
            output_dir: Directory to save all outputs (figures, results, etc.)
        """

        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Randomness control (overridable via set_seed)
        self.random_seed: int = RANDOM_SEED

        # Data storage
        self.df: Optional[pd.DataFrame] = None
        self.feature_columns: List[str] = []
        self.label_column: str = "Label"
        self.timestamp_column: str = "Timestamp"

        # Label encoder for consistent class mapping
        self.label_encoder = LabelEncoder()
        self.class_names: List[str] = []

        # Time windows
        self.time_windows: List[pd.DataFrame] = []
        self.window_statistics: List[Dict[str, Any]] = []

        # Models
        self.models: Dict[str, Any] = {}
        self.model_configs: List[ModelConfig] = []
        self.results: Dict[str, Dict] = {}

        # Feature scaler (fit only on first window)
        self.scaler = StandardScaler()

        # Optional runtime configuration (set by run_analysis.py)
        self.enabled_model_names: Optional[set[str]] = None
        self.include_dl_models: bool = True
        self.dl_epochs: Optional[int] = None

        # Initialize colorblind-friendly palette
        self._init_color_palette()

        logger.info(f"DurabilityAnalyzer initialized with data path: {data_path}")
        logger.info(f"Output directory: {output_dir}")

    def set_seed(self, seed: int) -> None:
        """Set a run-specific random seed and seed all relevant RNGs."""
        self.random_seed = int(seed)
        seed_everything(self.random_seed)

    def _init_color_palette(self):
        """Initialize colorblind-friendly color palette with patterns."""
        # Colorblind-friendly colors (Okabe-Ito palette)
        self.colors = {
            'RF': '#E69F00',  # Orange
            'XGB': '#56B4E9',  # Sky Blue
            'BAG-DT': '#009E73',  # Bluish Green
            'DNN': '#CC79A7',  # Reddish Purple
            'LSTM': '#F0E442',  # Yellow
            'Transformer': '#D55E00',  # Vermillion
            'LightGuard': '#0072B2',  # Blue
            'GEM': '#999999',  # Gray
            'EWC': '#661100',  # Dark Red
            'ARF': '#332288',  # Dark Purple
            'drift_line': '#D55E00',  # Vermillion
            'window_bg': '#F0F0F0',
            'grid_color': '#CCCCCC'
        }
        
        # Patterns for black & white printing
        self.patterns = ['///', '\\\\\\', '---', '|||', '+++', 'xxx', 'ooo', '...']
        
        # Create a custom colormap for visualizations
        self.cmap = ListedColormap([self.colors['RF'], self.colors['XGB'], 
                                   self.colors['BAG-DT'], self.colors['DNN'],
                                   self.colors['LSTM'], self.colors['Transformer']])
    
    def _validate_dataset(self, df: pd.DataFrame) -> bool:
        """
        Validate the dataset structure and content.
        
        Args:
            df: Loaded dataframe
            
        Returns:
            bool: True if validation passes
        """
        logger.info("Validating dataset structure...")
        
        # Keep validation schema-light: not all datasets use CICFlowMeter naming.
        required_columns = [self.timestamp_column, self.label_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for timestamp consistency
        if df[self.timestamp_column].isnull().any():
            logger.warning("Timestamp column contains null values")
        
        # Check label consistency
        unique_labels = df[self.label_column].unique()
        logger.info(f"Found {len(unique_labels)} unique labels: {unique_labels}")
        
        if len(unique_labels) < 2:
            logger.error("Insufficient number of classes for classification")
            return False
        
        # Check sample size
        if len(df) < 1000:
            logger.warning(f"Small dataset size: {len(df)} samples")
        
        return True

    def _infer_schema_columns(self, df: pd.DataFrame) -> None:
        """Infer label/timestamp column names when they differ across datasets."""

        if self.label_column not in df.columns:
            label_candidates = [
                "Label",
                "label",
                "Class",
                "class",
                "class1",
                "Category",
                "category",
                "App",
                "app",
                "Application",
                "application",
                "Traffic",
                "traffic",
            ]
            for c in label_candidates:
                if c in df.columns:
                    logger.info(f"Using label column '{c}' (was '{self.label_column}')")
                    self.label_column = c
                    break

        if self.timestamp_column not in df.columns:
            ts_candidates = [
                "Timestamp",
                "timestamp",
                "Time",
                "time",
                "StartTime",
                "Start Time",
                "Flow Start",
                "Flow Start Time",
                "FlowStartTime",
                "FlowStart",
                "Date",
                "date",
            ]
            for c in ts_candidates:
                if c in df.columns:
                    logger.info(f"Using timestamp column '{c}' (was '{self.timestamp_column}')")
                    self.timestamp_column = c
                    break

        # Some datasets (and some ARFF exports) do not include a usable timestamp.
        # To keep the pipeline runnable, we synthesize a monotonic timestamp.
        if self.timestamp_column not in df.columns:
            synthetic = "Timestamp"
            if synthetic in df.columns:
                # If it exists but we didn't select it, just switch to it.
                self.timestamp_column = synthetic
            else:
                logger.warning(
                    "No timestamp column found; synthesizing a monotonic 'Timestamp' from row order. "
                    "Temporal windows will reflect file/row ordering, not real capture time."
                )
                df[synthetic] = np.arange(len(df), dtype=np.int64)
                self.timestamp_column = synthetic
    
    def _standardize_labels(self, labels: pd.Series) -> np.ndarray:
        """
        Standardize labels to ensure consistency across windows.
        
        Args:
            labels: Raw label series
            
        Returns:
            Encoded labels
        """
        # Clean label strings (robust to non-string inputs)
        cleaned_labels = labels.astype(str).str.strip().str.upper()
        
        # Standardize variations
        label_mapping = {
            'AUDIO-STREAMING': 'AUDIO-STREAM',
            'AUDIO-STREAM': 'AUDIO-STREAM',
            'VIDEO-STREAMING': 'VIDEO-STREAM',
            'VIDEO-STREAM': 'VIDEO-STREAM',
            'FILE-TRANSFER': 'FILE-TRANSFER',
            'CHAT': 'CHAT',
            'BROWSING': 'BROWSING',
            'EMAIL': 'EMAIL',
            'P2P': 'P2P',
            'VOIP': 'VOIP'
        }
        
        standardized_labels = cleaned_labels.map(lambda x: label_mapping.get(x, x))
        
        # Fit or transform using label encoder
        if not hasattr(self.label_encoder, 'classes_'):
            encoded = self.label_encoder.fit_transform(standardized_labels)
            self.class_names = list(self.label_encoder.classes_)
            logger.info(f"Label encoder fitted with classes: {self.class_names}")
        else:
            # Check for new classes
            current_classes = set(self.label_encoder.classes_)
            new_classes = set(standardized_labels.unique()) - current_classes
            
            if new_classes:
                logger.warning(f"New classes found in window: {new_classes}")
                # For this analysis, we'll map new classes to the most similar existing class
                # In production, this would trigger a model update
                for new_class in new_classes:
                    # Find closest existing class by string similarity
                    similarities = [(existing, self._string_similarity(new_class, existing)) 
                                  for existing in current_classes]
                    closest = max(similarities, key=lambda x: x[1])[0]
                    standardized_labels = standardized_labels.replace(new_class, closest)
                    logger.info(f"Mapped new class '{new_class}' to existing class '{closest}'")
            
            encoded = self.label_encoder.transform(standardized_labels)
        
        return encoded
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate simple string similarity."""
        s1_lower = s1.lower()
        s2_lower = s2.lower()
        
        if s1_lower == s2_lower:
            return 1.0
        
        # Check for common substrings
        common_words = ['stream', 'audio', 'video', 'transfer', 'file', 'chat']
        for word in common_words:
            if word in s1_lower and word in s2_lower:
                return 0.7
        
        return 0.0
    
    def load_and_preprocess_data(self) -> bool:
        """
        Load the dataset and perform initial preprocessing.
        
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Loading dataset from {self.data_path}")

            # Load dataset (file or directory of CSVs)
            if self.data_path.is_dir():
                csv_files = sorted([p for p in self.data_path.rglob("*.csv") if p.is_file()])
                if not csv_files:
                    logger.error(f"No CSV files found under directory: {self.data_path}")
                    return False

                logger.info(f"Found {len(csv_files)} CSV files; concatenating...")
                frames = []
                for p in csv_files:
                    frames.append(pd.read_csv(p, low_memory=False))
                self.df = pd.concat(frames, ignore_index=True)
            else:
                # Load dataset (avoid forcing Timestamp dtype; parse robustly below)
                self.df = pd.read_csv(self.data_path, low_memory=False)
            logger.info(f"Dataset loaded: {len(self.df)} rows, {len(self.df.columns)} columns")

            # Normalize common CICFlowMeter column variants (e.g., Source IP -> Src IP)
            normalize_cic_flowmeter_columns(self.df)

            # Infer label/timestamp column names when not using CICFlowMeter defaults
            self._infer_schema_columns(self.df)
            
            # Validate dataset
            if not self._validate_dataset(self.df):
                return False
            
            # Parse + sort timestamps to create a true temporal sequence
            logger.info("Parsing timestamps...")
            if not ensure_timestamp_columns(self.df, self.timestamp_column):
                logger.error("Failed to parse timestamps into usable values")
                return False

            n_invalid_ts = int(self.df['timestamp_numeric'].isna().sum())
            if n_invalid_ts:
                logger.warning(f"Dropping {n_invalid_ts} rows with invalid timestamps")
                self.df = self.df.dropna(subset=['timestamp_numeric']).reset_index(drop=True)

            logger.info("Sorting data by parsed timestamp...")
            self.df = self.df.sort_values('timestamp_numeric').reset_index(drop=True)
            
            # Extract numeric features (exclude metadata columns)
            metadata_columns = [
                'Flow ID', 'Src IP', 'Dst IP', 'Protocol',
                'timestamp_dt', 'timestamp_numeric',
                self.label_column, self.timestamp_column,
                # Keep backward compatibility with canonical CIC names
                'Label', 'Timestamp',
            ]
            self.feature_columns = [col for col in self.df.columns 
                                  if col not in metadata_columns and 
                                  pd.api.types.is_numeric_dtype(self.df[col])]
            
            logger.info(f"Identified {len(self.feature_columns)} numeric features")
            
            # Handle missing values in features
            logger.info("Handling missing values...")
            for col in self.feature_columns:
                if self.df[col].isnull().any():
                    # Use median for numerical features
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    logger.debug(f"Filled missing values in {col} with median: {median_val}")
            
            # Standardize labels
            logger.info("Standardizing labels...")
            self.df['Label_encoded'] = self._standardize_labels(self.df[self.label_column])
            
            # Remove columns with zero variance
            zero_var_cols = []
            for col in self.feature_columns:
                if self.df[col].std() == 0:
                    zero_var_cols.append(col)
            
            if zero_var_cols:
                logger.warning(f"Removing {len(zero_var_cols)} columns with zero variance")
                self.feature_columns = [col for col in self.feature_columns 
                                      if col not in zero_var_cols]
            
            logger.info(f"Final feature count: {len(self.feature_columns)}")
            
            # Save preprocessed data
            preprocessed_path = self.output_dir / 'preprocessed_data.csv'
            self.df.to_csv(preprocessed_path, index=False)
            logger.info(f"Preprocessed data saved to {preprocessed_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading and preprocessing data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

            
    def create_time_windows(
        self,
        n_windows: int = 8,
        min_samples_per_class: int = 10,
        balance_classes: bool = True
    ) -> bool:
        """
        Partition dataset into chronological time windows with consistent class encoding.
        
        Args:
            n_windows: Number of windows to create
            min_samples_per_class: Minimum samples per class per window
            balance_classes: If True, oversample classes that are underrepresented
            
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Creating {n_windows} chronological time windows...")

            # ---------------------------------------------
            # 1️⃣ Ensure parsed timestamps exist
            # ---------------------------------------------
            if not ensure_timestamp_columns(self.df, self.timestamp_column):
                logger.error("Timestamp columns not available; run load_and_preprocess_data() first")
                return False

            if self.df['timestamp_numeric'].isna().any():
                n_invalid = int(self.df['timestamp_numeric'].isna().sum())
                logger.warning(f"Dropping {n_invalid} rows with invalid timestamps")
                self.df = self.df.dropna(subset=['timestamp_numeric']).reset_index(drop=True)

            self.df = self.df.sort_values('timestamp_numeric').reset_index(drop=True)
    
            # ---------------------------------------------
            # 2️⃣ Prepare class info
            # ---------------------------------------------
            all_class_ids = list(range(len(self.class_names)))  # all possible classes
            self.time_windows = []
            self.window_statistics = []
    
            total_samples = len(self.df)
            window_size = total_samples // n_windows
    
            # ---------------------------------------------
            # 3️⃣ Create windows
            # ---------------------------------------------
            for i in range(n_windows):
                start_idx = i * window_size
                end_idx = (i + 1) * window_size if i < n_windows - 1 else total_samples
                window_df = self.df.iloc[start_idx:end_idx].copy()
    
                # Class distribution
                class_counts = window_df['Label_encoded'].value_counts().to_dict()
                present_classes = list(class_counts.keys())
                missing_classes = [c for c in all_class_ids if c not in present_classes]
    
                if missing_classes:
                    logger.warning(f"Window {i+1}: Missing classes {missing_classes}")
    
                # ---------------------------------------------
                # 4️⃣ Optional class balancing
                # ---------------------------------------------
                if balance_classes:
                    for class_id in missing_classes:
                        # Try to get samples from other windows
                        prev_samples = self.df[self.df['Label_encoded'] == class_id]
                        if not prev_samples.empty:
                            sample_count = min_samples_per_class
                            oversample = prev_samples.sample(
                                n=sample_count,
                                replace=True,
                                random_state=self.random_seed
                            )
                            window_df = pd.concat([window_df, oversample], ignore_index=True)
                            logger.info(f"Oversampled {sample_count} samples for missing class {class_id} in window {i+1}")
    
                    # Check for classes with insufficient samples
                    for class_id, count in window_df['Label_encoded'].value_counts().items():
                        if count < min_samples_per_class:
                            needed = min_samples_per_class - count
                            prev_samples = self.df[self.df['Label_encoded'] == class_id]
                            if not prev_samples.empty:
                                oversample = prev_samples.sample(
                                    n=needed,
                                    replace=True,
                                    random_state=self.random_seed
                                )
                                window_df = pd.concat([window_df, oversample], ignore_index=True)
                                logger.info(f"Oversampled {needed} additional samples for class {class_id} in window {i+1}")
    
                # Shuffle window
                window_df = window_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
    
                # ---------------------------------------------
                # 5️⃣ Compute window statistics
                # ---------------------------------------------
                class_distribution = {int(k): int(v) for k, v in window_df['Label_encoded'].value_counts().items()}
    
                stats = {
                    'window_id': i + 1,
                    'n_samples': int(len(window_df)),
                    'class_distribution': class_distribution,
                    'timestamp_min': int(window_df['timestamp_numeric'].min()),
                    'timestamp_max': int(window_df['timestamp_numeric'].max()),
                    'datetime_range': (
                        str(window_df['timestamp_dt'].min()) if 'timestamp_dt' in window_df.columns else '',
                        str(window_df['timestamp_dt'].max()) if 'timestamp_dt' in window_df.columns else ''
                    ),
                    'features_mean': window_df[self.feature_columns].mean().to_list(),
                    'features_std': window_df[self.feature_columns].std().to_list()
                }
    
                self.time_windows.append(window_df)
                self.window_statistics.append(stats)
    
                logger.info(f"Window {i+1}: {len(window_df)} samples, classes present: {list(class_distribution.keys())}")
    
            # ---------------------------------------------
            # 6️⃣ Save statistics to CSV (JSON-serializable)
            # ---------------------------------------------
            stats_df = pd.DataFrame(self.window_statistics)
            stats_df.to_csv(self.output_dir / "window_statistics.csv", index=False)
    
            logger.info(f"Created {len(self.time_windows)} time windows successfully")
            return True
    
        except Exception as e:
            logger.error(f"Error creating time windows: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    
    def initialize_models(self):
        """Initialize the baseline models as specified in the paper."""
        logger.info("Initializing baseline models...")
        
        self.model_configs = [
            ModelConfig(
                name='RF',
                model_class=RandomForestClassifier,
                params={
                    'n_estimators': 100,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'random_state': self.random_seed,
                    'n_jobs': -1,
                    'verbose': 0
                },
                color=self.colors['RF'],
                marker='o',
                linestyle='-'
            ),
            ModelConfig(
                name='XGB',
                model_class=XGBClassifier,
                params={
                    'n_estimators': 100,
                    'max_depth': 10,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'use_label_encoder': False,
                    'eval_metric': 'mlogloss',
                    'random_state': self.random_seed,
                    'n_jobs': -1,
                    'verbosity': 0
                },
                color=self.colors['XGB'],
                marker='s',
                linestyle='--'
            ),
            ModelConfig(
                name='BAG-DT',
                model_class=BaggingClassifier,
                params={
                    'estimator': DecisionTreeClassifier(  # Changed from 'base_estimator' to 'estimator'
                        max_depth=10,
                        random_state=self.random_seed
                    ),
                    'n_estimators': 50,
                    'max_samples': 0.8,
                    'max_features': 0.8,
                    'bootstrap': True,
                    'bootstrap_features': False,
                    'random_state': self.random_seed,
                    'n_jobs': -1
                },
                color=self.colors['BAG-DT'],
                marker='^',
                linestyle='-.'
            ),
            ModelConfig(
                name='DNN',
                model_class=MLPClassifier,
                params={
                    'hidden_layer_sizes': (128, 64, 32),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.0001,
                    'batch_size': 256,
                    'learning_rate': 'adaptive',
                    'learning_rate_init': 0.001,
                    'max_iter': 100,
                    'early_stopping': True,
                    'validation_fraction': 0.1,
                    'random_state': self.random_seed,
                    'verbose': False
                },
                color=self.colors['DNN'],
                marker='D',
                linestyle=':'
            ),
            # Issue 3: Add LSTM baseline
            ModelConfig(
                name='LSTM',
                model_class=LSTMClassifier,
                params={
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout': 0.3,
                    'learning_rate': 0.001,
                    'batch_size': 64,
                    'epochs': 50,
                    'random_state': self.random_seed
                },
                color=self.colors['LSTM'],
                marker='p',
                linestyle='-'
            ),
            # Issue 3: Add Transformer baseline
            ModelConfig(
                name='Transformer',
                model_class=TransformerClassifier,
                params={
                    'd_model': 64,
                    'nhead': 4,
                    'num_layers': 2,
                    'dropout': 0.3,
                    'learning_rate': 0.001,
                    'batch_size': 64,
                    'epochs': 50,
                    'random_state': self.random_seed
                },
                color=self.colors['Transformer'],
                marker='h',
                linestyle='--'
            )
        ]

        # Optionally skip deep learning baselines
        if not self.include_dl_models:
            self.model_configs = [c for c in self.model_configs if c.name not in {"LSTM", "Transformer"}]

        # Optionally restrict to a subset of models
        if self.enabled_model_names is not None:
            wanted = {m.upper() for m in self.enabled_model_names}
            self.model_configs = [c for c in self.model_configs if c.name.upper() in wanted]

        # Optionally override DL epochs
        if self.dl_epochs is not None:
            for c in self.model_configs:
                if c.name in {"LSTM", "Transformer"}:
                    c.params["epochs"] = int(self.dl_epochs)
        
        for config in self.model_configs:
            self.models[config.name] = config.model_class(**config.params)
            logger.info(f"Initialized {config.name} model")

    
    def train_on_first_window(self):
        """Train all models on the first time window only."""
        if not self.time_windows:
            logger.error("No time windows available. Run create_time_windows() first.")
            return
        
        logger.info("Training models on first time window...")
        
        # Prepare first window data
        window_df = self.time_windows[0]
        X = window_df[self.feature_columns].values
        y = window_df['Label_encoded'].values
        
        # Clean features before fitting scaler
        X_clean = self._clean_features(X)
        
        # Fit scaler on cleaned first window only
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Train each model
        for config in self.model_configs:
            model_name = config.name
            logger.info(f"Training {model_name}...")
            
            try:
                self.models[model_name].fit(X_scaled, y)
                
                # Evaluate on training window
                y_pred = self.models[model_name].predict(X_scaled)
                train_acc = accuracy_score(y, y_pred)
                train_f1 = f1_score(y, y_pred, average='macro')
                
                logger.info(f"{model_name} - Train Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
        
        logger.info("All models trained on first window")

    
    def evaluate_longitudinal_performance(self):
        """
        Evaluate each static model on all time windows without updates.
        This is the core diagnostic experiment.
        """
        if not self.models:
            logger.error("Models not initialized. Run initialize_models() and train_on_first_window() first.")
            return
        
        logger.info("Starting longitudinal evaluation of static models...")
        
        # Initialize results storage
        self.results = {
            config.name: {
                'accuracy': [],
                'f1_macro': [],
                'f1_per_class': [],
                'recall_per_class': [],
                'precision_per_class': [],
                'predictions': [],
                'true_labels': []
            }
            for config in self.model_configs
        }
        
        # Evaluate on each window
        for window_idx, window_df in enumerate(tqdm(self.time_windows, desc="Evaluating windows")):
            X = window_df[self.feature_columns].values
            y_true = window_df['Label_encoded'].values
            
            try:
                # Handle infinite and NaN values before scaling
                X_clean = self._clean_features(X)
                
                # Scale features using the scaler fit on first window
                X_scaled = self.scaler.transform(X_clean)
                
                for config in self.model_configs:
                    model_name = config.name
                    model = self.models[model_name]
                    
                    try:
                        # Predict
                        y_pred = model.predict(X_scaled)
                        y_pred_proba = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
                        
                        # Calculate metrics
                        acc = accuracy_score(y_true, y_pred)
                        f1_macro = f1_score(y_true, y_pred, average='macro')
                        f1_per_class = f1_score(y_true, y_pred, average=None)
                        recall_per_class = recall_score(y_true, y_pred, average=None)
                        precision_per_class = precision_score(y_true, y_pred, average=None)
                        
                        # Store results
                        self.results[model_name]['accuracy'].append(acc)
                        self.results[model_name]['f1_macro'].append(f1_macro)
                        self.results[model_name]['f1_per_class'].append(f1_per_class)
                        self.results[model_name]['recall_per_class'].append(recall_per_class)
                        self.results[model_name]['precision_per_class'].append(precision_per_class)
                        self.results[model_name]['predictions'].append(y_pred)
                        self.results[model_name]['true_labels'].append(y_true)
                        
                    except Exception as e:
                        logger.error(f"Error evaluating {model_name} on window {window_idx + 1}: {str(e)}")
                        # Store NaN for failed evaluations
                        n_classes = len(self.class_names)
                        self.results[model_name]['accuracy'].append(np.nan)
                        self.results[model_name]['f1_macro'].append(np.nan)
                        self.results[model_name]['f1_per_class'].append(np.full(n_classes, np.nan))
                        self.results[model_name]['recall_per_class'].append(np.full(n_classes, np.nan))
                        self.results[model_name]['precision_per_class'].append(np.full(n_classes, np.nan))
            
            except Exception as e:
                logger.error(f"Error processing window {window_idx + 1}: {str(e)}")
                # Mark all models as failed for this window
                for config in self.model_configs:
                    model_name = config.name
                    n_classes = len(self.class_names)
                    self.results[model_name]['accuracy'].append(np.nan)
                    self.results[model_name]['f1_macro'].append(np.nan)
                    self.results[model_name]['f1_per_class'].append(np.full(n_classes, np.nan))
                    self.results[model_name]['recall_per_class'].append(np.full(n_classes, np.nan))
                    self.results[model_name]['precision_per_class'].append(np.full(n_classes, np.nan))
        
        # Save results to file
        self._export_phase1_window_metrics()
        logger.info("Longitudinal evaluation completed")


    def _export_phase1_window_metrics(self) -> None:
        """Export Phase I per-window metrics for aggregation (e.g., 95% CI plots)."""

        if not self.results:
            return

        rows: list[dict[str, Any]] = []
        for model_name, metrics in self.results.items():
            n = len(metrics.get("accuracy", []))
            for window_idx in range(n):
                recall_list = metrics.get("recall_per_class", [])
                precision_list = metrics.get("precision_per_class", [])

                avg_recall = (
                    float(np.nanmean(recall_list[window_idx]))
                    if window_idx < len(recall_list)
                    else float("nan")
                )
                avg_precision = (
                    float(np.nanmean(precision_list[window_idx]))
                    if window_idx < len(precision_list)
                    else float("nan")
                )

                rows.append(
                    {
                        "seed": int(self.random_seed),
                        "model": str(model_name),
                        "window": int(window_idx + 1),
                        "accuracy": float(metrics["accuracy"][window_idx]),
                        "f1_macro": float(metrics["f1_macro"][window_idx]),
                        "avg_recall": avg_recall,
                        "avg_precision": avg_precision,
                    }
                )

        if not rows:
            return

        out_path = self.output_dir / "phase1_window_metrics.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)


    def _clean_features(self, X: np.ndarray) -> np.ndarray:
        """
        Clean feature matrix by handling infinite and extreme values.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Cleaned feature matrix
        """
        X_clean = X.copy()
        
        # Replace infinite values with NaN
        X_clean[np.isinf(X_clean)] = np.nan
        
        # Find columns with extreme values (more than 10 standard deviations from mean)
        for col_idx in range(X_clean.shape[1]):
            col_data = X_clean[:, col_idx]
            col_mean = np.nanmean(col_data)
            col_std = np.nanstd(col_data)
            
            if col_std > 0:  # Avoid division by zero
                # Cap extreme values at mean ± 10 * std
                upper_bound = col_mean + 10 * col_std
                lower_bound = col_mean - 10 * col_std
                
                # Replace extreme values with bounds
                extreme_mask = (col_data > upper_bound) | (col_data < lower_bound)
                X_clean[extreme_mask, col_idx] = np.where(
                    col_data[extreme_mask] > upper_bound,
                    upper_bound,
                    lower_bound
                )
        
        # Fill remaining NaN values with column medians
        for col_idx in range(X_clean.shape[1]):
            col_data = X_clean[:, col_idx]
            col_median = np.nanmedian(col_data)
            nan_mask = np.isnan(col_data)
            X_clean[nan_mask, col_idx] = col_median
        
        return X_clean


    
    def _save_results(self):
        """Save evaluation results to files."""
        # Save summary results
        summary_data = []
        for model_name, metrics in self.results.items():
            for window_idx in range(len(metrics['accuracy'])):
                summary_data.append({
                    'model': model_name,
                    'window': window_idx + 1,
                    'accuracy': metrics['accuracy'][window_idx],
                    'f1_macro': metrics['f1_macro'][window_idx],
                    'avg_recall': np.nanmean(metrics['recall_per_class'][window_idx]) if window_idx < len(metrics['recall_per_class']) else np.nan,
                    'avg_precision': np.nanmean(metrics['precision_per_class'][window_idx]) if window_idx < len(metrics['precision_per_class']) else np.nan
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / 'performance_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Performance summary saved to {summary_path}")
        
        # Save detailed results - handle dataclass objects properly
        window_stats_list = []
        for stats in self.window_statistics:
            # Check if stats is a WindowStatistics object or a dictionary
            if hasattr(stats, 'window_id'):
                # It's a WindowStatistics object
                window_stats_list.append({
                    'window_id': stats.window_id,
                    'n_samples': stats.n_samples,
                    'class_distribution': stats.class_distribution,
                    'datetime_range': stats.datetime_range if hasattr(stats, 'datetime_range') else ('', '')
                })
            elif isinstance(stats, dict) and 'window_id' in stats:
                # It's already a dictionary
                window_stats_list.append(stats)
            else:
                # Unknown type, create a minimal dictionary
                window_stats_list.append({
                    'window_id': 0,
                    'n_samples': 0,
                    'class_distribution': {},
                    'datetime_range': ('', '')
                })
        
        details = {
            'class_names': [str(name) for name in self.class_names],  # Ensure strings
            'model_configs': [
                {
                    'name': config.name,
                    'model_class': str(config.model_class.__name__),
                    'params': {k: str(v) if callable(v) else v for k, v in config.params.items()},
                    'color': config.color,
                    'marker': config.marker,
                    'linestyle': config.linestyle
                }
                for config in self.model_configs
            ],
            'window_statistics': window_stats_list
        }
        
        details_path = self.output_dir / 'analysis_details.json'
        with open(details_path, 'w') as f:
            json.dump(details, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Analysis details saved to {details_path}")

    
    
    def calculate_statistical_significance(self):
        """Calculate statistical significance of performance decay."""
        logger.info("Calculating statistical significance of performance decay...")
        
        significance_results = {}
        
        for config in self.model_configs:
            model_name = config.name
            f1_scores = self.results[model_name]['f1_macro']
            
            # Remove NaN values
            valid_scores = [score for score in f1_scores if not np.isnan(score)]
            window_indices = list(range(1, len(valid_scores) + 1))
            
            if len(valid_scores) < 3:
                logger.warning(f"Insufficient data for statistical test on {model_name}")
                continue
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                window_indices, valid_scores
            )
            
            # Calculate decay rate
            decay_per_window = -slope * 100  # Convert to percentage
            
            significance_results[model_name] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_err': std_err,
                'decay_per_window_percent': decay_per_window,
                'total_decay_percent': (valid_scores[0] - valid_scores[-1]) * 100,
                'is_significant': p_value < 0.01 and slope < 0
            }
            
            logger.info(f"{model_name}: Slope = {slope:.6f}, p-value = {p_value:.6f}, "
                      f"Significant decay = {significance_results[model_name]['is_significant']}")
        
        # Save significance results
        sig_df = pd.DataFrame(significance_results).T
        sig_path = self.output_dir / 'statistical_significance.csv'
        sig_df.to_csv(sig_path, index=False)
        
        logger.info(f"Statistical significance results saved to {sig_path}")
        
        return significance_results

    def visualize_performance_decay(self):
        return _visualize_performance_decay(self)
        
    
    def visualize_concept_drift(self):
        return _visualize_concept_drift(self, random_seed=self.random_seed)
    

    def generate_class_specific_analysis(self):
        return _generate_class_specific_analysis(self)
    
    def run_full_analysis(self):
        return run_full_analysis_phase1(self)


class EnhancedDurabilityAnalyzer(DurabilityAnalyzer):
    """
    Extended analyzer that includes LightGuard framework for Phase II analysis.
    """
    
    def __init__(self, data_path: Union[str, Path], output_dir: Union[str, Path]):
        """Initialize enhanced analyzer with LightGuard capabilities."""
        super().__init__(data_path, output_dir)
        
        # LightGuard framework
        self.lightguard: Optional[LightGuardFramework] = None
        self.lightguard_results: Dict[str, List] = {}
        
        # Ablation variants
        self.lightguard_variants: Dict[str, Any] = {}
        
        # Baseline models for comparison
        self.periodic_retrain_model: Optional[Any] = None
        
        logger.info("EnhancedDurabilityAnalyzer with LightGuard initialized")


    
    def initialize_lightguard(self, base_model_name: str = 'XGB', 
                            buffer_size_percent: float = 5.0,
                            drift_threshold: float = 0.05) -> bool:
        """
        Initialize LightGuard framework with selected base model.
        
        Args:
            base_model_name: Name of base model ('RF', 'XGB', 'BAG-DT', 'DNN')
            buffer_size_percent: Buffer size as percentage of first window
            drift_threshold: MMD threshold for drift detection
            
        Returns:
            True if successful
        """
        if not self.models:
            logger.error("Models not initialized. Run initialize_models() first.")
            return False
        
        if base_model_name not in self.models:
            logger.error(f"Model {base_model_name} not found. Available: {list(self.models.keys())}")
            return False
        
        # Calculate buffer size
        if self.time_windows:
            first_window_size = len(self.time_windows[0])
            buffer_size = int(first_window_size * buffer_size_percent / 100)
        else:
            buffer_size = 1000  # Default
        
        # Get base model
        base_model = self.models[base_model_name]
        
        # For XGBoost, we need to set enable_categorical to avoid warnings
        if base_model_name == 'XGB':
            # Reinitialize with proper parameters for continual learning
            xgb_params = {
                'n_estimators': 100,
                'max_depth': 6,  # Reduced for faster updates
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'use_label_encoder': False,
                'eval_metric': 'mlogloss',
                'random_state': self.random_seed,
                'n_jobs': -1,
                'verbosity': 0
            }
            from xgboost import XGBClassifier
            base_model = XGBClassifier(**xgb_params)
        
        # Create LightGuard framework
        self.lightguard = LightGuardFramework(
            base_model=base_model,
            feature_columns=self.feature_columns,
            class_names=self.class_names,
            buffer_size=buffer_size,
            drift_threshold=drift_threshold,
            update_fraction=0.25
        )
        
        logger.info(f"LightGuard initialized with {base_model_name} base model, "
                   f"buffer_size={buffer_size} ({buffer_size_percent}%)")
        
        return True


        
    def evaluate_lightguard_longitudinal(self):
        """
        Evaluate LightGuard framework on all time windows with adaptive updates.
        """
        if self.lightguard is None:
            logger.error("LightGuard not initialized. Run initialize_lightguard() first.")
            return
        
        if not self.time_windows:
            logger.error("No time windows available.")
            return
        
        logger.info("Starting longitudinal evaluation of LightGuard framework...")
        
        # Initialize results storage
        self.lightguard_results = {
            'accuracy': [],
            'f1_macro': [],
            'f1_weighted': [],
            'avg_precision': [],
            'avg_recall': [],
            'window_updates': [],  # Whether update occurred in each window
            'update_times': [],
            'memory_usage': [],
            'drift_scores': []
        }
        
        # Track which windows had updates
        update_windows = []
        
        # Evaluate on each window
        for window_idx, window_df in enumerate(tqdm(self.time_windows, desc="LightGuard Evaluation")):
            X = window_df[self.feature_columns].values
            y_true = window_df['Label_encoded'].values
            
            try:
                # Clean features
                X_clean = self._clean_features(X)
                
                # For first window, train initial model and fit scaler
                if window_idx == 0:
                    # Fit scaler on first window
                    X_scaled = self.lightguard.scaler.fit_transform(X_clean)
                    
                    # Train on first window
                    self.lightguard.base_model.fit(X_scaled, y_true)
                    
                    # Initialize reference distribution
                    self.lightguard.reference_features = X_scaled[:100].copy()
                    self.lightguard.reference_labels = y_true[:100].copy()
                    
                    # Add samples to buffer - use the fitted scaler
                    self.lightguard._update_buffer(X_clean, y_true, X_scaled)
                
                # For subsequent windows
                else:
                    # Scale features using the already fitted scaler
                    X_scaled = self.lightguard.scaler.transform(X_clean)
                    
                    # Get predictions for drift detection
                    if hasattr(self.lightguard.base_model, 'predict_proba'):
                        y_proba = self.lightguard.base_model.predict_proba(X_scaled)
                        y_pred = np.argmax(y_proba, axis=1)
                    else:
                        y_pred = self.lightguard.base_model.predict(X_scaled)
                        y_proba = None
                    
                    # Detect drift
                    drift_alert = self.lightguard.detect_drift(
                        X_current=X_scaled,
                        y_pred=y_pred,
                        y_proba=y_proba,
                        window_id=window_idx + 1
                    )
                    
                    # Update model if drift detected
                    if drift_alert.is_drift:
                        self.lightguard.update_model(X_clean, y_true, window_idx + 1)
                        update_windows.append(window_idx + 1)
                
                # Evaluate model on current window
                # Make sure we use the right X_scaled variable
                if window_idx == 0:
                    # X_scaled already defined above
                    pass
                else:
                    X_scaled = self.lightguard.scaler.transform(X_clean)
                
                metrics = self.lightguard.evaluate(X_clean, y_true, window_idx + 1)
                
                # Store results
                self.lightguard_results['accuracy'].append(metrics['accuracy'])
                self.lightguard_results['f1_macro'].append(metrics['f1_macro'])
                self.lightguard_results['f1_weighted'].append(metrics['f1_weighted'])
                self.lightguard_results['avg_precision'].append(metrics['avg_precision'])
                self.lightguard_results['avg_recall'].append(metrics['avg_recall'])
                self.lightguard_results['window_updates'].append(window_idx + 1 in update_windows)
                
                # Get framework stats
                stats = self.lightguard.get_stats()
                self.lightguard_results['update_times'].append(stats['avg_update_time'])
                self.lightguard_results['memory_usage'].append(stats['avg_memory_usage'])
                
                # Track drift score
                if window_idx > 0:
                    self.lightguard_results['drift_scores'].append(drift_alert.mmd_score)
                else:
                    self.lightguard_results['drift_scores'].append(0.0)
                
                logger.debug(f"Window {window_idx + 1}: Accuracy={metrics['accuracy']:.4f}, "
                           f"Update={window_idx + 1 in update_windows}")
            
            except Exception as e:
                logger.error(f"Error evaluating LightGuard on window {window_idx + 1}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
                # Store NaN for failed evaluations
                self.lightguard_results['accuracy'].append(np.nan)
                self.lightguard_results['f1_macro'].append(np.nan)
                self.lightguard_results['f1_weighted'].append(np.nan)
                self.lightguard_results['avg_precision'].append(np.nan)
                self.lightguard_results['avg_recall'].append(np.nan)
                self.lightguard_results['window_updates'].append(False)
                self.lightguard_results['update_times'].append(np.nan)
                self.lightguard_results['memory_usage'].append(np.nan)
                self.lightguard_results['drift_scores'].append(np.nan)
        
        # Save LightGuard results
        self._save_lightguard_results(update_windows)
        
        logger.info(f"LightGuard evaluation completed. Updates occurred in windows: {update_windows}")

    
    
    def _save_lightguard_results(self, update_windows: List[int]):
        """Save LightGuard evaluation results."""
        # Create results dataframe
        results_data = []
        for window_idx in range(len(self.lightguard_results['accuracy'])):
            results_data.append({
                'window': int(window_idx + 1),
                'accuracy': to_serializable(self.lightguard_results['accuracy'][window_idx]),
                'f1_macro': to_serializable(self.lightguard_results['f1_macro'][window_idx]),
                'f1_weighted': to_serializable(self.lightguard_results['f1_weighted'][window_idx]),
                'avg_precision': to_serializable(self.lightguard_results['avg_precision'][window_idx]),
                'avg_recall': to_serializable(self.lightguard_results['avg_recall'][window_idx]),
                'update_occurred': bool(self.lightguard_results['window_updates'][window_idx]),
                'drift_score': to_serializable(self.lightguard_results['drift_scores'][window_idx] 
                                                   if window_idx < len(self.lightguard_results['drift_scores']) 
                                                   else None),
                'avg_update_time': to_serializable(self.lightguard_results['update_times'][window_idx]),
                'avg_memory_mb': to_serializable(self.lightguard_results['memory_usage'][window_idx])
            })
        
        results_df = pd.DataFrame(results_data)
        results_path = self.output_dir / 'lightguard_results.csv'
        results_df.to_csv(results_path, index=False)
        
        # Save framework statistics
        if self.lightguard:
            stats = self.lightguard.get_stats()
            stats_path = self.output_dir / 'lightguard_stats.json'
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2, cls=NumpyEncoder)  # Use custom encoder
        
        logger.info(f"LightGuard results saved to {results_path}")
    
    def visualize_lightguard_performance(self):
        return _visualize_lightguard_performance(self)

    def visualize_tradeoff_analysis(self):
        return _visualize_tradeoff_analysis(self)
    
    def _get_model_size(self, model: Any) -> float:
        """Estimate model size in MB."""
        import pickle
        import io
        
        try:
            # Serialize model to estimate size
            buffer = io.BytesIO()
            pickle.dump(model, buffer)
            size_bytes = buffer.getbuffer().nbytes
            return size_bytes / 1024 / 1024  # Convert to MB
        except:
            return 10.0  # Default estimate for XGBoost
    
    def run_ablation_study(self):
        return _run_ablation_study(self)
    
    def _run_ablation_variant(self, use_buffer: bool = True, use_detector: bool = True) -> Dict:
        """Run a specific ablation variant."""
        # Create a copy of the base model
        base_model = self.models['XGB']  # Use XGB as base
        model_params = base_model.get_params()
        variant_model = type(base_model)(**model_params)
        
        # Initialize variant framework with modified parameters
        variant_lg = LightGuardFramework(
            base_model=variant_model,
            feature_columns=self.feature_columns,
            class_names=self.class_names,
            buffer_size=100 if use_buffer else 0,  # Small buffer or none
            drift_threshold=self.lightguard.drift_threshold if use_detector else 0.0,  # Always drift if no detector
            update_fraction=0.25
        )
        
        # Run evaluation
        variant_results = {
            'accuracy': [],
            'f1_macro': [],
            'window_updates': []
        }
        
        update_windows = []
        
        for window_idx, window_df in enumerate(self.time_windows):
            X = window_df[self.feature_columns].values
            y_true = window_df['Label_encoded'].values
            
            X_clean = self._clean_features(X)
            X_scaled = self.scaler.transform(X_clean)
            
            if window_idx == 0:
                # Train on first window
                variant_lg.base_model.fit(X_scaled, y_true)
                variant_lg.reference_features = X_scaled[:100].copy()
                variant_lg.reference_labels = y_true[:100].copy()
                
                if use_buffer:
                    variant_lg._update_buffer(X_clean, y_true, X_scaled)
                    
                # Evaluate on first window (after training)
                y_pred = variant_lg.base_model.predict(X_scaled)
                variant_results['accuracy'].append(accuracy_score(y_true, y_pred))
                variant_results['f1_macro'].append(f1_score(y_true, y_pred, average='macro'))
                variant_results['window_updates'].append(False)
            else:
                # CRITICAL: Evaluate BEFORE any updates to prevent cheating
                y_pred = variant_lg.base_model.predict(X_scaled)
                variant_results['accuracy'].append(accuracy_score(y_true, y_pred))
                variant_results['f1_macro'].append(f1_score(y_true, y_pred, average='macro'))
                
                # For no detector variant, update every window (AFTER evaluation)
                if not use_detector:
                    variant_lg.update_model(X_clean, y_true, window_idx + 1)
                    update_windows.append(window_idx + 1)
                    variant_results['window_updates'].append(True)
                else:
                    # Normal drift detection
                    if hasattr(variant_lg.base_model, 'predict_proba'):
                        y_proba = variant_lg.base_model.predict_proba(X_scaled)
                    else:
                        y_proba = None
                    
                    drift_alert = variant_lg.detect_drift(
                        X_current=X_scaled,
                        y_pred=y_pred,
                        y_proba=y_proba,
                        window_id=window_idx + 1
                    )
                    
                    if drift_alert.is_drift:
                        variant_lg.update_model(X_clean, y_true, window_idx + 1)
                        update_windows.append(window_idx + 1)
                        variant_results['window_updates'].append(True)
                    else:
                        variant_results['window_updates'].append(False)
        
        return {
            'avg_f1': np.nanmean(variant_results['f1_macro']),
            'n_updates': sum(variant_results['window_updates']),
            'performance_history': variant_results['f1_macro']
        }

    
    def _visualize_ablation_study(self, ablation_results: Dict[str, Dict]):
        """Visualize ablation study results."""
        # Create a new figure for the ablation study visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        
        # Plot 1: Performance comparison (bar chart - keep as bars)
        ax = axes[0]
        variants = list(ablation_results.keys())
        avg_f1_scores = [ablation_results[v]['avg_f1'] for v in variants]
        
        # Colors with hatches
        colors = ['#E69F00', '#56B4E9', '#009E73']
        hatches = ['///', '\\\\\\', '---']
        
        bars = ax.bar(variants, avg_f1_scores, 
                     color=colors[:len(variants)], 
                     alpha=0.85,
                     edgecolor='black',
                     linewidth=1.5)
        
        # Apply hatches
        for bar, hatch in zip(bars, hatches[:len(variants)]):
            bar.set_hatch(hatch)
        
        ax.set_xlabel('Ablation Variant', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average F1-Score', fontsize=14, fontweight='bold')
        ax.set_title('Component Importance Analysis', fontsize=16, fontweight='bold')
        ax.set_xticklabels(variants, rotation=45, ha='right', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim([0, 1.0])
        
        # Add value labels
        for bar, score in zip(bars, avg_f1_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Plot 2: Performance decay comparison
        ax = axes[1]
        window_indices = list(range(1, len(self.time_windows) + 1))
        
        for i, (variant, results) in enumerate(ablation_results.items()):
            performance_history = results['performance_history']
            valid_mask = ~np.isnan(performance_history)
            
            if np.any(valid_mask):
                valid_values = np.array(performance_history)[valid_mask]
                valid_indices = np.array(window_indices)[valid_mask]


                # Plot line
                ax.plot(
                    valid_indices,
                    valid_values,
                    color=colors[i],
                    marker=['o', 's', '^'][i],
                    linestyle=['--', '-.', '-'][i],
                    linewidth=1.5,
                    markersize=4,
                    alpha=0.8,
                    label=variant,
                )
        
        ax.set_xlabel('Time Window', fontsize=14, fontweight='bold')
        ax.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
        ax.set_title('Performance Decay in Ablation Variants', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1.0])
        ax.legend(fontsize=11, loc='best')
        
        # Calculate and annotate performance drops
        for variant, results in ablation_results.items():
            if len(results['performance_history']) >= 2:
                initial = results['performance_history'][0]
                final = results['performance_history'][-1]
                drop_pct = ((initial - final) / initial * 100) if initial > 0 else 0
                
                if variant == 'No Buffer':  # Catastrophic forgetting case
                    ax.annotate(f'Catastrophic Forgetting:\n{drop_pct:.1f}% drop',
                               xy=(window_indices[-1], final),
                               xytext=(-80, -30),
                               textcoords='offset points',
                               arrowprops=dict(arrowstyle='->', color='red', linewidth=1.5),
                               fontsize=10,
                               color='red',
                               fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        # Plot 3: Update frequency per variant (real, non-synthetic)
        ax = axes[2]

        n_updates = [ablation_results[v].get('n_updates', 0) for v in variants]
        bars = ax.bar(range(len(variants)), n_updates, color=colors[:len(variants)], alpha=0.85)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=45, ha='right', fontsize=12)
        ax.set_xlabel('Ablation Variant', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Updates', fontsize=14, fontweight='bold')
        ax.set_title('Update Frequency by Variant', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        for bar, val in zip(bars, n_updates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(0.5, 0.01 * (max(n_updates) if n_updates else 1)),
                f'{int(val)}',
                ha='center',
                fontsize=11,
                fontweight='bold',
            )
        
        # Save figure
        output_path = self.output_dir / 'visualization_5_ablation_study.pdf'
        apply_journal_style()
        saved_path = save_figure(fig, output_path)
        logger.info(f"Ablation study visualization saved to {saved_path}")

        plt.close(fig)

    # ============================================================================
    # Issue 2: Drift Detection Threshold Ablation
    # ============================================================================
    
    def run_drift_threshold_ablation(self, thresholds: List[float] = None):
        return _run_drift_threshold_ablation(self, thresholds=thresholds)
    
    def _visualize_drift_threshold_ablation(self, results: Dict[float, Dict]):
        """Visualize drift threshold ablation results."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        
        thresholds = list(results.keys())
        avg_f1 = [results[t]['avg_f1'] for t in thresholds]
        n_updates = [results[t]['n_updates'] for t in thresholds]
        drift_alerts = [results[t]['drift_alerts'] for t in thresholds]
        
        # Plot 1: F1 vs Threshold
        ax = axes[0]
        ax.plot(thresholds, avg_f1, 'o-', color='#0072B2', linewidth=2, markersize=8)
        ax.set_xlabel('Drift Threshold (MMD)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average F1-Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance vs Drift Threshold', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Updates vs Threshold
        ax = axes[1]
        ax.bar(range(len(thresholds)), n_updates, color='#E69F00', alpha=0.8)
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([str(t) for t in thresholds])
        ax.set_xlabel('Drift Threshold (MMD)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Updates', fontsize=12, fontweight='bold')
        ax.set_title('Update Frequency vs Threshold', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Efficiency (F1 / Updates)
        ax = axes[2]
        efficiency = [f1 / (n + 1) for f1, n in zip(avg_f1, n_updates)]
        ax.plot(thresholds, efficiency, 's-', color='#009E73', linewidth=2, markersize=8)
        ax.set_xlabel('Drift Threshold (MMD)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Efficiency (F1 / Updates)', fontsize=12, fontweight='bold')
        ax.set_title('Efficiency vs Threshold', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Mark optimal threshold
        optimal_idx = np.argmax(efficiency)
        ax.axvline(x=thresholds[optimal_idx], color='red', linestyle='--', alpha=0.7)
        ax.annotate(f'Optimal: {thresholds[optimal_idx]}',
                   xy=(thresholds[optimal_idx], efficiency[optimal_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='red')
        
        apply_journal_style()
        saved_path = save_figure(fig, self.output_dir / 'drift_threshold_ablation.pdf')
        logger.info(f"Drift threshold ablation visualization saved to {saved_path}")
        plt.close(fig)

    # ============================================================================
    # Issue 5: Section 5.3 - Comparison with State-of-the-Art Baselines
    # ============================================================================
    
    def run_continual_learning_comparison(self):
        return _run_continual_learning_comparison(self, random_seed=self.random_seed)
    
    def _calculate_bwt(self, f1_history: List[float]) -> float:
        """Calculate Backward Transfer (BWT)."""
        if len(f1_history) < 2:
            return 0.0
        bwt = 0.0
        for i in range(1, len(f1_history)):
            bwt += f1_history[i] - f1_history[0]
        return bwt / (len(f1_history) - 1)
    
    def _calculate_fwt(self, f1_history: List[float]) -> float:
        """Calculate Forward Transfer (FWT)."""
        if len(f1_history) < 2:
            return 0.0
        # Measure how well initial training transfers
        return np.mean(f1_history[1:]) - f1_history[0]
    
    def _calculate_forgetting(self, f1_history: List[float]) -> float:
        """Calculate average forgetting."""
        if len(f1_history) < 2:
            return 0.0
        max_f1 = max(f1_history)
        return max_f1 - f1_history[-1]
    
    def _evaluate_static_baseline(self) -> Dict:
        """
        Evaluate STATIC baseline - model trained once and never updated.
        This shows the problem LightGuard solves (performance decay over time).
        """
        f1_history = []
        
        # Initialize base model (same as LightGuard's base)
        base_model = XGBClassifier(
            n_estimators=100, max_depth=10, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=self.random_seed, verbosity=0
        )
        
        for window_idx, window_df in enumerate(self.time_windows):
            X = window_df[self.feature_columns].values
            y = window_df['Label_encoded'].values
            X_clean = self._clean_features(X)
            X_scaled = self.scaler.transform(X_clean)
            
            if window_idx == 0:
                # Train ONLY on first window - no updates after
                base_model.fit(X_scaled, y)
            
            # Evaluate on each window (with the static model from window 1)
            y_pred = base_model.predict(X_scaled)
            f1 = f1_score(y, y_pred, average='macro')
            f1_history.append(f1)
        
        return {
            'f1_history': f1_history,
            'avg_f1': np.nanmean(f1_history),
            'final_f1': f1_history[-1] if f1_history else 0,
            'bwt': self._calculate_bwt(f1_history),
            'fwt': self._calculate_fwt(f1_history),
            'forgetting': self._calculate_forgetting(f1_history)
        }

    def _evaluate_ewc_baseline(self) -> Dict:
        """
        Evaluate Elastic Weight Consolidation baseline with FAIR memory budget.
        Uses same 5% memory buffer as LightGuard.
        """
        f1_history = []
        
        # Use SAME memory budget as LightGuard (5%)
        memory_size = int(len(self.time_windows[0]) * 0.05)
        
        # Get all classes from dataset
        all_classes = np.unique(np.concatenate([
            window_df['Label_encoded'].values for window_df in self.time_windows
        ]))
        
        # Initialize base model
        base_model = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=self.random_seed, verbosity=0
        )
        
        # Memory buffer for EWC (stores important samples)
        memory_X = None
        memory_y = None
        
        for window_idx, window_df in enumerate(self.time_windows):
            X = window_df[self.feature_columns].values
            y = window_df['Label_encoded'].values
            X_clean = self._clean_features(X)
            X_scaled = self.scaler.transform(X_clean)
            
            if window_idx == 0:
                # Train on first window only
                base_model.fit(X_scaled, y)
                
                # Store samples ensuring class balance
                indices = []
                for cls in np.unique(y):
                    cls_idx = np.where(y == cls)[0]
                    n_samples = min(len(cls_idx), memory_size // len(np.unique(y)))
                    indices.extend(np.random.choice(cls_idx, size=n_samples, replace=False))
                memory_X = X_scaled[indices]
                memory_y = y[indices]
            else:
                # EWC: Retrain on memory buffer + class-balanced sample from current window
                sample_size = min(memory_size, len(X_scaled))
                
                # Ensure we sample from all available classes
                current_indices = []
                for cls in np.unique(y):
                    cls_idx = np.where(y == cls)[0]
                    n_cls_samples = min(len(cls_idx), sample_size // len(np.unique(y)))
                    if n_cls_samples > 0:
                        current_indices.extend(np.random.choice(cls_idx, size=n_cls_samples, replace=False))
                
                if len(current_indices) > 0:
                    X_train = np.vstack([memory_X, X_scaled[current_indices]])
                    y_train = np.concatenate([memory_y, y[current_indices]])
                    
                    # Fit with combined data
                    try:
                        base_model.fit(X_train, y_train)
                    except ValueError:
                        # Fallback: just use memory if class issue persists
                        base_model.fit(memory_X, memory_y)
            
            # Evaluate on CURRENT window
            y_pred = base_model.predict(X_scaled)
            f1 = f1_score(y, y_pred, average='macro', zero_division=0)
            f1_history.append(f1)
        
        return {
            'f1_history': f1_history,
            'avg_f1': np.nanmean(f1_history),
            'final_f1': f1_history[-1] if f1_history else 0,
            'bwt': self._calculate_bwt(f1_history),
            'fwt': self._calculate_fwt(f1_history),
            'forgetting': self._calculate_forgetting(f1_history)
        }
    
    def _evaluate_gem_baseline(self) -> Dict:
        """
        Evaluate Gradient Episodic Memory baseline with FAIR memory budget.
        Uses same 5% memory buffer as LightGuard.
        """
        f1_history = []
        
        # Use SAME memory budget as LightGuard (5%)
        memory_size = int(len(self.time_windows[0]) * 0.05)
        
        # Initialize base model
        base_model = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=self.random_seed, verbosity=0
        )
        
        # Episodic memory buffer
        memory_X = None
        memory_y = None
        
        for window_idx, window_df in enumerate(self.time_windows):
            X = window_df[self.feature_columns].values
            y = window_df['Label_encoded'].values
            X_clean = self._clean_features(X)
            X_scaled = self.scaler.transform(X_clean)
            
            if window_idx == 0:
                # Train on first window
                base_model.fit(X_scaled, y)
                
                # Initialize memory with class-balanced samples
                indices = []
                for cls in np.unique(y):
                    cls_idx = np.where(y == cls)[0]
                    n_samples = min(len(cls_idx), memory_size // len(np.unique(y)))
                    indices.extend(np.random.choice(cls_idx, size=n_samples, replace=False))
                memory_X = X_scaled[indices]
                memory_y = y[indices]
            else:
                # GEM: Train on memory + class-balanced sample from current window
                current_indices = []
                for cls in np.unique(y):
                    cls_idx = np.where(y == cls)[0]
                    n_cls_samples = min(len(cls_idx), memory_size // len(np.unique(y)))
                    if n_cls_samples > 0:
                        current_indices.extend(np.random.choice(cls_idx, size=n_cls_samples, replace=False))
                
                if len(current_indices) > 0:
                    X_combined = np.vstack([memory_X, X_scaled[current_indices]])
                    y_combined = np.concatenate([memory_y, y[current_indices]])
                    
                    try:
                        base_model.fit(X_combined, y_combined)
                    except ValueError:
                        # Fallback: use memory only
                        base_model.fit(memory_X, memory_y)
            
            # Evaluate
            y_pred = base_model.predict(X_scaled)
            f1 = f1_score(y, y_pred, average='macro', zero_division=0)
            f1_history.append(f1)
        
        return {
            'f1_history': f1_history,
            'avg_f1': np.nanmean(f1_history),
            'final_f1': f1_history[-1] if f1_history else 0,
            'bwt': self._calculate_bwt(f1_history),
            'fwt': self._calculate_fwt(f1_history),
            'forgetting': self._calculate_forgetting(f1_history)
        }
    
    def _evaluate_arf_baseline(self) -> Dict:
        """Evaluate Adaptive Random Forest baseline using river library."""
        f1_history = []
        
        if not RIVER_AVAILABLE:
            return {'f1_history': [], 'avg_f1': 0, 'final_f1': 0, 'bwt': 0, 'fwt': 0, 'forgetting': 0}
        
        # Initialize ARF model
        arf = river_forest.ARFClassifier(seed=self.random_seed)
        
        for window_idx, window_df in enumerate(self.time_windows):
            X = window_df[self.feature_columns].values
            y = window_df['Label_encoded'].values
            X_clean = self._clean_features(X)
            X_scaled = self.scaler.transform(X_clean)
            
            # Online learning - learn from each sample
            for i in range(len(X_scaled)):
                x_dict = {f'f{j}': float(X_scaled[i, j]) for j in range(X_scaled.shape[1])}
                arf.learn_one(x_dict, int(y[i]))
            
            # Evaluate on window
            correct = 0
            y_pred = []
            for i in range(len(X_scaled)):
                x_dict = {f'f{j}': float(X_scaled[i, j]) for j in range(X_scaled.shape[1])}
                pred = arf.predict_one(x_dict)
                y_pred.append(pred if pred is not None else 0)
            
            f1 = f1_score(y, y_pred, average='macro', zero_division=0)
            f1_history.append(f1)
        
        return {
            'f1_history': f1_history,
            'avg_f1': np.nanmean(f1_history),
            'final_f1': f1_history[-1] if f1_history else 0,
            'bwt': self._calculate_bwt(f1_history),
            'fwt': self._calculate_fwt(f1_history),
            'forgetting': self._calculate_forgetting(f1_history)
        }
    
    def _visualize_sota_comparison(self, results: Dict[str, Dict]):
        """Visualize comparison with state-of-the-art baselines."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        
        methods = list(results.keys())
        colors = [self.colors.get(m, '#333333') for m in methods]
        
        # Plot 1: Performance over time
        ax = axes[0]
        window_indices = list(range(1, len(self.time_windows) + 1))
        
        for i, method in enumerate(methods):
            f1_history = results[method]['f1_history']
            if f1_history:
                values = np.array(f1_history)
                ax.plot(window_indices[:len(values)], values, 
                       'o-', color=colors[i], linewidth=2, 
                       markersize=6, label=method)
        
        ax.set_xlabel('Time Window', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Plot 2: Average F1
        ax = axes[1]
        avg_f1 = [results[m]['avg_f1'] for m in methods]
        bars = ax.bar(range(len(methods)), avg_f1, color=colors, alpha=0.8)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel('Average F1-Score', fontsize=12, fontweight='bold')
        ax.set_title('Overall Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, avg_f1):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
        
        # Plot 3: Continual Learning Metrics (BWT, FWT, Forgetting)
        ax = axes[2]
        x = np.arange(len(methods))
        width = 0.25
        
        bwt = [results[m]['bwt'] for m in methods]
        fwt = [results[m]['fwt'] for m in methods]
        forgetting = [results[m]['forgetting'] for m in methods]
        
        ax.bar(x - width, bwt, width, label='BWT', color='#0072B2', alpha=0.8)
        ax.bar(x, fwt, width, label='FWT', color='#E69F00', alpha=0.8)
        ax.bar(x + width, forgetting, width, label='Forgetting', color='#D55E00', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
        ax.set_title('Continual Learning Metrics', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        apply_journal_style()
        saved_path = save_figure(fig, self.output_dir / 'sota_comparison.pdf')
        logger.info(f"SOTA comparison visualization saved to {saved_path}")
        plt.close(fig)
    
    def _save_comparison_results(self, results: Dict[str, Dict]):
        """Save comparison results to file."""
        comparison_data = []
        for method, metrics in results.items():
            comparison_data.append({
                'method': method,
                'avg_f1': metrics['avg_f1'],
                'final_f1': metrics['final_f1'],
                'bwt': metrics['bwt'],
                'fwt': metrics['fwt'],
                'forgetting': metrics['forgetting']
            })
        
        df = pd.DataFrame(comparison_data)
        df.to_csv(self.output_dir / 'sota_comparison_results.csv', index=False)
        logger.info("SOTA comparison results saved")
