import numpy as np
import pandas as pd
import os
from scipy import signal
from scipy.integrate import simpson


def compute_relative_band_power(data, fs=500, nperseg=1000):
    """
    Compute relative power for EEG frequency bands using Welch's method.

    Parameters:
    -----------
    data : ndarray, shape (n_epochs, n_times, n_channels)
        EEG data array
    fs : int, default=500
        Sampling frequency in Hz
    nperseg : int, default=1000
        Length of each segment for Welch's method

    Returns:
    --------
    relative_power : ndarray, shape (n_epochs, n_channels * n_bands)
        Relative power features (95 features = 19 channels x 5 bands)
        Order: [ch0_delta, ch0_theta, ch0_alpha, ch0_beta, ch0_gamma,
                ch1_delta, ch1_theta, ...]
    """
    n_epochs, n_times, n_channels = data.shape

    # Define frequency bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 35),
        'Gamma': (35, 50)
    }
    n_bands = len(bands)

    # Initialize output array
    relative_power = np.zeros((n_epochs, n_channels, n_bands))

    # Process each epoch and channel
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            # Extract signal for this epoch and channel
            signal_data = data[epoch_idx, :, ch_idx]

            # Compute PSD using Welch's method
            freqs, psd = signal.welch(signal_data, fs=fs, nperseg=nperseg)

            # Calculate absolute power for each band
            band_powers = []
            for band_name, (low_freq, high_freq) in bands.items():
                # Find frequency indices within the band
                freq_idx = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]

                # Integrate PSD over the frequency band using Simpson's rule
                band_power = simpson(psd[freq_idx], x=freqs[freq_idx])
                band_powers.append(band_power)

            # Convert to numpy array
            band_powers = np.array(band_powers)

            # Calculate total power (sum of all bands)
            total_power = np.sum(band_powers)

            # Compute relative power
            if total_power > 0:
                relative_power[epoch_idx, ch_idx, :] = band_powers / total_power
            else:
                # Handle edge case where total power is zero
                relative_power[epoch_idx, ch_idx, :] = 0

    # Reshape to (n_epochs, n_channels * n_bands)
    relative_power = relative_power.reshape(n_epochs, n_channels * n_bands)

    return relative_power


def save_features_to_csv(features, labels, subjects, channel_names, output_dir='.'):
    """
    Save features split by class label into separate CSV files with channel names.

    Parameters:
    -----------
    features : ndarray, shape (n_epochs, n_features)
        Feature array (95 features = 19 channels x 5 bands)
    labels : ndarray, shape (n_epochs,)
        Class labels (0=AD, 1=FTD, 2=CN)
    subjects : ndarray, shape (n_epochs,)
        Subject IDs for each epoch
    channel_names : list
        List of 19 channel names (e.g., ['Fp1', 'Fp2', ...])
    output_dir : str, default='.'
        Directory to save CSV files

    Creates three CSV files:
    - AD_features.csv (Label 0)
    - FTD_features.csv (Label 1)
    - CN_features.csv (Label 2)

    Each file contains columns: Subject_ID, Label, {Channel}_{Band} features
    """
    # Validate input shapes
    n_epochs = features.shape[0]
    assert labels.shape[0] == n_epochs, "Labels length must match features"
    assert subjects.shape[0] == n_epochs, "Subjects length must match features"
    assert len(channel_names) == 19, "Must provide exactly 19 channel names"
    assert features.shape[1] == 95, "Features must have 95 columns (19 channels x 5 bands)"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define band names (order matches compute_relative_band_power)
    band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    # Generate feature column names
    # Order: [Ch0_Band0, Ch0_Band1, ..., Ch0_Band4, Ch1_Band0, ...]
    feature_columns = []
    for channel in channel_names:
        for band in band_names:
            feature_columns.append(f"{channel}_{band}")

    # Create master DataFrame
    df = pd.DataFrame(features, columns=feature_columns)
    # Convert Subject_ID to 1-indexed (add 1 to each subject ID)
    df.insert(0, 'Subject_ID', subjects + 1)
    df.insert(1, 'Label', labels)

    # Define class mapping
    class_info = {
        0: ('AD', 'Alzheimer\'s Disease'),
        1: ('FTD', 'Frontotemporal Dementia'),
        2: ('CN', 'Control (Healthy)')
    }

    # Split and save by class
    print("Saving features split by class label...")
    print("-" * 60)

    for label_value, (filename_prefix, class_name) in class_info.items():
        # Filter by label
        class_df = df[df['Label'] == label_value]

        if len(class_df) == 0:
            print(f"WARNING: No samples found for {class_name} (Label {label_value})")
            continue

        # Save to CSV
        filepath = os.path.join(output_dir, f"{filename_prefix}_features.csv")
        class_df.to_csv(filepath, index=False)

        print(f"{class_name} (Label {label_value}):")
        print(f"  - Samples: {len(class_df)}")
        print(f"  - File: {filepath}")

    print("-" * 60)
    print(f"Total samples: {n_epochs}")
    print(f"Columns per file: {len(df.columns)} (Subject_ID + Label + 95 features)")
    print("Done!")
