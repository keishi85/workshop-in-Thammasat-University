import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold


class EEGDataLoader:
    """
    Data loader for EEG features extracted from dementia patients (AD, FTD) and healthy controls (CN).

    Loads pre-processed EEG features from CSV files and provides a clean interface for
    machine learning model training with proper cross-validation support.

    Features:
    - Loads and combines AD, FTD, and CN feature CSV files
    - Supports task-specific data extraction (binary or multi-class classification)
    - Provides StratifiedGroupKFold for subject-level cross-validation (prevents data leakage)

    Usage:
        loader = EEGDataLoader(data_dir='features')
        X, y, groups = loader.get_data(task='AD_vs_CN')

        for fold, (train_idx, test_idx) in enumerate(loader.get_kfold_indices(X, y, groups)):
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_test, y_test = X.iloc[test_idx], y[test_idx]
            # Train your model here
    """

    def __init__(self, data_dir='features'):
        """
        Initialize the data loader by loading and combining all CSV files.

        Parameters:
        -----------
        data_dir : str, default='features'
            Directory containing the feature CSV files (AD_features.csv, FTD_features.csv, CN_features.csv)

        Raises:
        -------
        FileNotFoundError
            If data_dir doesn't exist or required CSV files are missing
        ValueError
            If CSV files are empty or have column mismatches
        """
        self.data_dir = Path(data_dir)

        # Validate directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Define expected CSV files
        csv_files = {
            'AD': self.data_dir / 'AD_features.csv',
            'FTD': self.data_dir / 'FTD_features.csv',
            'CN': self.data_dir / 'CN_features.csv'
        }

        # Check all files exist
        missing_files = [name for name, path in csv_files.items() if not path.exists()]
        if missing_files:
            raise FileNotFoundError(
                f"Missing CSV files for: {', '.join(missing_files)}\n"
                f"Expected files in {self.data_dir}:\n"
                f"  - AD_features.csv\n"
                f"  - FTD_features.csv\n"
                f"  - CN_features.csv"
            )

        # Load CSV files
        print(f"Loading data from {self.data_dir}...")
        dataframes = []
        for class_name, filepath in csv_files.items():
            df = pd.read_csv(filepath)

            # Validate not empty
            if len(df) == 0:
                raise ValueError(f"{filepath.name} is empty")

            dataframes.append(df)
            print(f"  {class_name}: {len(df)} samples")

        # Combine all DataFrames
        self.data = pd.concat(dataframes, axis=0, ignore_index=True)

        # Validate column consistency
        expected_columns = ['Subject_ID', 'Label']
        for col in expected_columns:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Check for missing values
        missing_count = self.data.isnull().sum().sum()
        if missing_count > 0:
            print(f"WARNING: Found {missing_count} missing values in the data")
            print(self.data.isnull().sum()[self.data.isnull().sum() > 0])

        # Store metadata
        self.n_samples = len(self.data)
        self.n_features = len(self.data.columns) - 2  # Exclude Subject_ID and Label
        self.feature_columns = [col for col in self.data.columns if col not in ['Subject_ID', 'Label']]

        print(f"\nTotal: {self.n_samples} samples, {self.n_features} features")
        print(f"Class distribution: AD={len(self.data[self.data['Label']==0])}, "
              f"FTD={len(self.data[self.data['Label']==1])}, "
              f"CN={len(self.data[self.data['Label']==2])}")
        print(f"Unique subjects: {self.data['Subject_ID'].nunique()}")
        print("Data loaded successfully!\n")

    def get_data(self, task='AD_vs_CN'):
        """
        Extract task-specific data with appropriate label remapping.

        Parameters:
        -----------
        task : str, default='AD_vs_CN'
            Task type to extract:
            - 'AD_vs_CN': Binary classification (AD vs CN)
            - 'FTD_vs_CN': Binary classification (FTD vs CN)
            - 'AD_vs_FTD': Binary classification (AD vs FTD)
            - 'all': Three-class classification (AD vs FTD vs CN)

        Returns:
        --------
        X : pandas.DataFrame
            Feature matrix (n_samples, n_features)
            Excludes Subject_ID and Label columns
        y : numpy.ndarray
            Labels (n_samples,)
            Binary tasks remapped to 0 and 1
        groups : numpy.ndarray
            Subject IDs (n_samples,)
            For use with GroupKFold to prevent data leakage

        Raises:
        -------
        ValueError
            If task name is not recognized
        """
        if task == 'AD_vs_CN':
            # Binary: AD (0) vs CN (2) -> remap to 0 and 1
            df_filtered = self.data[self.data['Label'].isin([0, 2])].copy()
            df_filtered['Label'] = df_filtered['Label'].map({0: 0, 2: 1})

        elif task == 'FTD_vs_CN':
            # Binary: FTD (1) vs CN (2) -> remap to 0 and 1
            df_filtered = self.data[self.data['Label'].isin([1, 2])].copy()
            df_filtered['Label'] = df_filtered['Label'].map({1: 0, 2: 1})

        elif task == 'AD_vs_FTD':
            # Binary: AD (0) vs FTD (1) -> keep original labels
            df_filtered = self.data[self.data['Label'].isin([0, 1])].copy()

        elif task == 'all':
            # Multi-class: keep all data and original labels
            df_filtered = self.data.copy()

        else:
            raise ValueError(
                f"Unknown task: '{task}'\n"
                f"Supported tasks: 'AD_vs_CN', 'FTD_vs_CN', 'AD_vs_FTD', 'all'"
            )

        # Extract features, labels, and groups
        X = df_filtered.drop(columns=['Subject_ID', 'Label'])
        y = df_filtered['Label'].values
        groups = df_filtered['Subject_ID'].values

        return X, y, groups

    def get_kfold_indices(self, X, y, groups, n_splits=5):
        """
        Generate cross-validation indices using StratifiedGroupKFold.

        This ensures:
        1. Stratified: Class distribution is maintained across folds
        2. Grouped: Same subject's epochs don't appear in both train and test sets

        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Labels
        groups : numpy.ndarray
            Subject IDs (for grouping)
        n_splits : int, default=5
            Number of folds

        Yields:
        -------
        train_idx : numpy.ndarray
            Training set indices for this fold
        test_idx : numpy.ndarray
            Test set indices for this fold
        """
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(X, y, groups):
            yield train_idx, test_idx

    def get_dl_kfold_indices(self, X, y, groups, n_splits=5, val_size=0.25):
        """
        Generate Train/Validation/Test indices for Deep Learning.

        Implements a 'Nested' split approach:
        1. Outer Loop: Splits data into (Train + Val) and Test using StratifiedGroupKFold.
        2. Inner Loop: Splits (Train + Val) into Train and Validation.

        This ensures:
        - No data leakage (Subject-level isolation across ALL 3 sets)
        - Class balance is maintained in ALL 3 sets

        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Labels
        groups : numpy.ndarray
            Subject IDs (for grouping)
        n_splits : int, default=5
            Number of outer folds (Test sets).
            Example: 5 splits -> Test set is 20%
        val_size : float, default=0.25
            Proportion of the "Train+Val" set to use for Validation.
            Example: If Test is 20% (remaining 80%), setting val_size=0.25
            means Val is 0.25 * 80% = 20% of total.
            Final Split ratio -> Train: 60%, Val: 20%, Test: 20%

        Yields:
        -------
        train_idx : numpy.ndarray
            Training set indices for this fold
        val_idx : numpy.ndarray
            Validation set indices for this fold
        test_idx : numpy.ndarray
            Test set indices for this fold
        """
        # Outer Split: Separate Test set
        outer_skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_val_idx, test_idx in outer_skf.split(X, y, groups):

            # Extract the Train+Val subset data
            X_tv = X.iloc[train_val_idx]
            y_tv = y[train_val_idx]
            groups_tv = groups[train_val_idx]

            # Calculate inner splits to achieve desired validation ratio
            # logic: n_splits_inner = 1 / val_size
            # e.g., val_size=0.25 -> 1/0.25 = 4 splits
            inner_splits = int(1 / val_size)

            inner_skf = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=42)

            # Perform one inner split to get Train and Val
            # We only need the first fold of this inner split
            train_idx_rel, val_idx_rel = next(inner_skf.split(X_tv, y_tv, groups_tv))

            # Map relative indices back to original indices
            train_idx = train_val_idx[train_idx_rel]
            val_idx = train_val_idx[val_idx_rel]

            yield train_idx, val_idx, test_idx

    def get_feature_names(self):
        """
        Get list of feature column names.

        Returns:
        --------
        list
            List of feature column names (95 features)
        """
        return self.feature_columns

    def get_class_distribution(self, task='AD_vs_CN'):
        """
        Get class distribution for a specific task.

        Parameters:
        -----------
        task : str, default='AD_vs_CN'
            Task type (same options as get_data)

        Returns:
        --------
        dict
            Dictionary with class counts
        """
        _, y, _ = self.get_data(task=task)
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))

    def __repr__(self):
        """String representation of the data loader."""
        return (
            f"EEGDataLoader(\n"
            f"  data_dir='{self.data_dir}',\n"
            f"  n_samples={self.n_samples},\n"
            f"  n_features={self.n_features},\n"
            f"  n_subjects={self.data['Subject_ID'].nunique()},\n"
            f"  classes={{AD: {len(self.data[self.data['Label']==0])}, "
            f"FTD: {len(self.data[self.data['Label']==1])}, "
            f"CN: {len(self.data[self.data['Label']==2])}}}\n"
            f")"
        )


if __name__ == '__main__':
    print("="*70)
    print("EEG Data Loader - Usage Example")
    print("="*70)
    print()

    # Initialize loader
    loader = EEGDataLoader(data_dir='features')
    print(loader)
    print()

    # Get data for AD vs CN classification
    print("="*70)
    print("Task: AD vs CN (Binary Classification)")
    print("="*70)
    X, y, groups = loader.get_data(task='AD_vs_CN')

    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Groups shape: {groups.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"  - Class 0 (AD): {np.sum(y==0)} samples")
    print(f"  - Class 1 (CN): {np.sum(y==1)} samples")
    print(f"Unique subjects: {len(np.unique(groups))}")
    print()

    # Example cross-validation loop
    print("Cross-validation with StratifiedGroupKFold (5 folds):")
    print("-" * 70)
    for fold, (train_idx, test_idx) in enumerate(loader.get_kfold_indices(X, y, groups, n_splits=5)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

        # Get unique subjects in train and test
        train_subjects = np.unique(groups[train_idx])
        test_subjects = np.unique(groups[test_idx])

        # Verify no subject overlap (critical for preventing data leakage)
        overlap = set(train_subjects) & set(test_subjects)

        print(f"Fold {fold+1}:")
        print(f"  Train: {len(X_train):5d} samples (Class 0: {np.sum(y_train==0):5d}, Class 1: {np.sum(y_train==1):5d}) | {len(train_subjects):2d} subjects")
        print(f"  Test:  {len(X_test):5d} samples (Class 0: {np.sum(y_test==0):5d}, Class 1: {np.sum(y_test==1):5d}) | {len(test_subjects):2d} subjects")
        print(f"  Subject overlap: {len(overlap)} (should be 0 for valid GroupKFold)")

        # Here's where team members would do: model.fit(X_train, y_train)

    print()
    print("="*70)
    print("Testing Other Tasks")
    print("="*70)

    # Test FTD vs CN
    print("\n[FTD vs CN]")
    X, y, groups = loader.get_data(task='FTD_vs_CN')
    print(f"Samples: {len(y)}, Class distribution: {np.bincount(y)} (FTD={np.sum(y==0)}, CN={np.sum(y==1)})")

    # Test AD vs FTD
    print("\n[AD vs FTD]")
    X, y, groups = loader.get_data(task='AD_vs_FTD')
    print(f"Samples: {len(y)}, Class distribution: {np.bincount(y)} (AD={np.sum(y==0)}, FTD={np.sum(y==1)})")

    # Test all classes
    print("\n[All Classes (3-class)]")
    X, y, groups = loader.get_data(task='all')
    print(f"Samples: {len(y)}, Class distribution: {np.bincount(y)} (AD={np.sum(y==0)}, FTD={np.sum(y==1)}, CN={np.sum(y==2)})")

    print()
    print("="*70)
    print("Feature Information")
    print("="*70)
    feature_names = loader.get_feature_names()
    print(f"Total features: {len(feature_names)}")
    print(f"First 10 features: {feature_names[:10]}")
    print(f"Last 10 features: {feature_names[-10:]}")

    print()
    print("="*70)
    print("Deep Learning: Train/Val/Test Split (Nested StratifiedGroupKFold)")
    print("="*70)

    # Test deep learning split with AD vs CN
    X, y, groups = loader.get_data(task='AD_vs_CN')
    print(f"\nTask: AD vs CN")
    print(f"Total samples: {len(y)}")
    print(f"Expected split (n_splits=5, val_size=0.25): Train~60%, Val~20%, Test~20%\n")

    print("Train/Val/Test splits with subject-level isolation:")
    print("-" * 70)

    for fold, (train_idx, val_idx, test_idx) in enumerate(loader.get_dl_kfold_indices(X, y, groups, n_splits=5, val_size=0.25)):
        # Extract data
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
        groups_train = groups[train_idx]
        groups_val = groups[val_idx]
        groups_test = groups[test_idx]

        # Get unique subjects
        train_subjects = set(np.unique(groups_train))
        val_subjects = set(np.unique(groups_val))
        test_subjects = set(np.unique(groups_test))

        # Check for data leakage (subject overlap)
        overlap_train_val = train_subjects & val_subjects
        overlap_train_test = train_subjects & test_subjects
        overlap_val_test = val_subjects & test_subjects

        # Calculate percentages
        total = len(y)
        train_pct = len(train_idx) / total * 100
        val_pct = len(val_idx) / total * 100
        test_pct = len(test_idx) / total * 100

        print(f"Fold {fold+1}:")
        print(f"  Train: {len(train_idx):5d} samples ({train_pct:5.1f}%) | Class 0: {np.sum(y_train==0):5d}, Class 1: {np.sum(y_train==1):5d} | {len(train_subjects):2d} subjects")
        print(f"  Val:   {len(val_idx):5d} samples ({val_pct:5.1f}%) | Class 0: {np.sum(y_val==0):5d}, Class 1: {np.sum(y_val==1):5d} | {len(val_subjects):2d} subjects")
        print(f"  Test:  {len(test_idx):5d} samples ({test_pct:5.1f}%) | Class 0: {np.sum(y_test==0):5d}, Class 1: {np.sum(y_test==1):5d} | {len(test_subjects):2d} subjects")
        print(f"  Subject overlap (Train-Val): {len(overlap_train_val)} (should be 0)")
        print(f"  Subject overlap (Train-Test): {len(overlap_train_test)} (should be 0)")
        print(f"  Subject overlap (Val-Test): {len(overlap_val_test)} (should be 0)")

    print()
    print("="*70)
    print("All tests completed successfully!")
    print("="*70)
