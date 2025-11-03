"""
Data loading and preprocessing for NF-UNSW-NB15 dataset.
Implements the encoding scheme from the paper.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Handles loading and preprocessing of NF-UNSW-NB15 dataset."""
    
    def __init__(self, data_path: str, random_state: int = 123):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to the NF-UNSW-NB15 dataset CSV file
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        self.features = [
            'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT',
            'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'OUT_BYTES', 
            'IN_PKTS', 'OUT_PKTS', 'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS'
        ]
        # Features to use (excluding IP addresses and ports as per paper)
        self.selected_features = [
            'PROTOCOL',           # IPv4 protocol
            'L7_PROTO',          # Layer 7 protocol
            'TCP_FLAGS',         # TCP flags
            'OUT_PKTS',          # Out packets count
            'IN_PKTS',           # In packets count
            'IN_BYTES',          # In bytes count
            'OUT_BYTES',         # Out bytes count
            'FLOW_DURATION_MILLISECONDS'  # Flow duration
        ]
        self.encoding_tables = {}
        
    def load_and_preprocess(self, test_size: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess the dataset.
        
        Args:
            test_size: Proportion of dataset to use for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        # Select features
        X = df[self.selected_features].copy()
        y = df['Label'].copy()  # Assuming 'Label' column exists (0=benign, 1=malicious)
        
        # Handle any missing values
        X = X.fillna(0)
        
        # Balance the dataset by resampling benign samples
        print("Balancing dataset...")
        X_benign = X[y == 0]
        X_malicious = X[y == 1]
        y_benign = y[y == 0]
        y_malicious = y[y == 1]
        
        # Resample benign to match malicious count
        X_benign_resampled, y_benign_resampled = resample(
            X_benign, y_benign,
            n_samples=len(X_malicious),
            random_state=self.random_state
        )
        
        # Combine
        X_balanced = pd.concat([X_benign_resampled, X_malicious])
        y_balanced = pd.concat([y_benign_resampled, y_malicious])
        
        print(f"Balanced dataset size: {len(X_balanced)} samples")
        print(f"Malicious: {sum(y_balanced == 1)}, Benign: {sum(y_balanced == 0)}")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced,
            test_size=test_size,
            random_state=1
        )
        
        # Convert to numpy arrays
        X_train = X_train.values
        X_test = X_test.values
        y_train = y_train.values
        y_test = y_test.values
        
        # Build encoding tables
        print("Building encoding tables...")
        self._build_encoding_tables(X_train)
        
        # Encode features to angles
        print("Encoding features to quantum angles...")
        X_train_encoded = self._encode_features(X_train)
        X_test_encoded = self._encode_features(X_test)
        
        # Convert labels to -1, 1 for hinge loss (0 -> 1, 1 -> -1)
        y_train = np.where(y_train == 0, 1, -1)
        y_test = np.where(y_test == 0, 1, -1)
        
        return X_train_encoded, X_test_encoded, y_train, y_test
    
    def _build_encoding_tables(self, X_train: np.ndarray):
        """
        Build encoding tables for each feature.
        Maps feature values to angles in [0, π] with 0.25° granularity.
        
        Args:
            X_train: Training data
        """
        min_granularity = np.radians(0.25)  # 0.25 degrees in radians
        max_angle = np.pi
        
        for i, feature_name in enumerate(self.selected_features):
            feature_values = X_train[:, i]
            unique_values = np.unique(feature_values)
            n_unique = len(unique_values)
            
            # Determine encoding strategy based on number of unique values
            if n_unique <= 100:  # Categorical-like features
                # Split [0, π] evenly among unique values
                angles = np.linspace(0, max_angle, n_unique)
                encoding_dict = {val: angle for val, angle in zip(unique_values, angles)}
            else:  # Continuous features - use percentile binning
                # Number of bins limited by granularity
                n_bins = int(max_angle / min_granularity)
                n_bins = min(n_bins, 720)  # Max 720 bins (π / 0.25° in radians)
                
                # Calculate percentiles
                percentiles = np.linspace(0, 100, n_bins + 1)
                bins = np.percentile(feature_values, percentiles)
                bins = np.unique(bins)  # Remove duplicates
                
                # Create encoding dictionary
                encoding_dict = {}
                for val in unique_values:
                    # Find which bin this value belongs to
                    bin_idx = np.digitize(val, bins) - 1
                    bin_idx = max(0, min(bin_idx, len(bins) - 2))
                    # Map to angle
                    angle = (bin_idx / (len(bins) - 1)) * max_angle
                    encoding_dict[val] = angle
            
            self.encoding_tables[feature_name] = encoding_dict
            
            print(f"Feature {feature_name}: {n_unique} unique values -> {len(encoding_dict)} angles")
    
    def _encode_features(self, X: np.ndarray) -> np.ndarray:
        """
        Encode features to angles using the encoding tables.
        
        Args:
            X: Feature matrix
            
        Returns:
            Encoded feature matrix with angles in [0, π]
        """
        X_encoded = np.zeros_like(X, dtype=np.float32)
        
        for i, feature_name in enumerate(self.selected_features):
            encoding_dict = self.encoding_tables[feature_name]
            for j in range(X.shape[0]):
                value = X[j, i]
                # Use encoding table, default to 0 if value not found
                X_encoded[j, i] = encoding_dict.get(value, 0.0)
        
        return X_encoded
    
    def calculate_certainty_factor(self, probabilities: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calculate certainty factor as defined in the paper.
        C = |α₀|² - |α₁|² where |0⟩ is the expected state for True (benign, label=1)
        
        Args:
            probabilities: Probability of measuring |0⟩ state
            y_true: True labels (1 for benign, -1 for malicious)
            
        Returns:
            Certainty factors
        """
        prob_0 = probabilities
        prob_1 = 1 - probabilities
        
        certainty = np.where(
            y_true == 1,  # If true label is benign (1)
            prob_0 - prob_1,  # C = |α₀|² - |α₁|²
            prob_1 - prob_0   # C = |α₁|² - |α₀|² (inverted for malicious)
        )
        
        return certainty