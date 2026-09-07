# =============================================================================
# Advanced EEG Cleanup & Feature Extraction Pipeline
# Extracted from advanced_cleanup.ipynb (all code cells, in order)
# All Persian comments/strings translated to English
# =============================================================================

import mne
import numpy as np

# -----------------------------------------------------------------------------
# Section 1: Load Raw EEG Data
# -----------------------------------------------------------------------------
raw = mne.io.read_raw_fif(r"D:\Signal_processing\EEG_Pre_processing\cleaned_eeg_data_raw.fif", preload=True)

print("DONE")

# -----------------------------------------------------------------------------
# Section 2: Extract EEG Channels and Detect Bad Channels (3-Sigma Rule)
# -----------------------------------------------------------------------------
# Extract only EEG channels data (excluding event/stimulus channels)
eeg_picks = mne.pick_types(raw.info, eeg=True, stim=False, exclude=[])
eeg_data = raw.get_data(picks=eeg_picks)
eeg_ch_names = [raw.info['ch_names'][i] for i in eeg_picks]

# Calculate standard deviation for each channel to identify anomalies
channel_stds = np.std(eeg_data, axis=1)

# Define upper and lower statistical thresholds (3-Sigma Rule)
threshold_high = np.mean(channel_stds) + 3 * np.std(channel_stds)
threshold_low = np.mean(channel_stds) - 2 * np.std(channel_stds)

# Identify bad channels
bad_channels = []
for idx, ch_name in enumerate(eeg_ch_names):
    if channel_stds[idx] > threshold_high or channel_stds[idx] < threshold_low:
        bad_channels.append(ch_name)

# Handle bad channels: mark and interpolate using spatial neighbors
if bad_channels:
    print(f"⚠️ Bad EEG channels detected: {bad_channels}")
    raw.info['bads'] = bad_channels
    # Interpolate bad channels using Spherical Spline Interpolation
    raw.interpolate_bads(reset=True, mode='accurate')
    print("✅ Bad EEG channels successfully interpolated!")
else:
    print("✅ No bad or noisy EEG channels detected. All 19 electrodes are clean!")

# -----------------------------------------------------------------------------
# Section 3: Epoching and Artifact Rejection
# -----------------------------------------------------------------------------
# Create fixed-length events every 2.0 seconds
events = mne.make_fixed_length_events(raw, duration=2.0)

# Define peak-to-peak rejection threshold for EEG channels (100 microvolts)
reject_criteria = dict(eeg=100e-6)

# Create Epochs and automatically reject noisy segments
epochs = mne.Epochs(
    raw,
    events,
    tmin=0,
    tmax=2.0,
    baseline=None,
    reject=reject_criteria,
    preload=True
)

# Display summary of clean vs total epochs
print(f"✅ Created {len(epochs)} clean epochs out of {len(events)} total segments.")
print(f"📊 Drop log summary: {epochs.drop_log_stats():.2f}% of epochs rejected due to heavy artifacts.")

# Save the clean epochs to disk for spectral analysis (PSD) and machine learning
epochs.save('cleaned_eeg_epochs-epo.fif', overwrite=True)

print("🎉 Advanced Cleanup Complete! Epochs successfully saved as 'cleaned_eeg_epochs-epo.fif'")

# -----------------------------------------------------------------------------
# Section 4: Frequency Band Definition and PSD Computation
# -----------------------------------------------------------------------------
# 1. Define frequency bands of interest
FREQ_BANDS = {
    'Delta': (1.0, 4.0),
    'Theta': (4.0, 8.0),
    'Alpha': (8.0, 12.0),
    'Beta': (12.0, 30.0)
}

# Ensure 'event' channel is dropped from the epochs object
if 'event' in epochs.ch_names:
    epochs.drop_channels(['event'])

print("Sampling Frequency:", epochs.info['sfreq'])

# The output shape is (n_epochs, n_channels, n_times)
print("Number of time samples per epoch:", epochs.get_data().shape[2])

# 2. Compute Power Spectral Density (PSD) using Welch's method
# -----------------------------------------------------------------------------
# Parameters adjusted for sfreq=250Hz and n_times=500 (2-second epochs)
spectrum = epochs.compute_psd(
    method='welch',
    fmin=1.0,
    fmax=30.0,
    n_fft=256,       # Optimal window length for 501 samples
    n_overlap=128    # 50% overlap between segments
)

# psds shape: (n_epochs, n_channels, n_freqs)
# freqs shape: (n_freqs,)
psds, freqs = spectrum.get_data(return_freqs=True)

# Calculate frequency step resolution (df ~ 0.97 Hz)
df = freqs[1] - freqs[0]

# Total power integration over 1-30 Hz
total_power = np.sum(psds, axis=-1) * df
abs_powers = {}
rel_powers = {}

# -----------------------------------------------------------------------------
# Section 5: Absolute and Relative Band Power Extraction
# -----------------------------------------------------------------------------
# Extract Absolute and Relative Power for each defined band
for band_name, (f_low, f_high) in FREQ_BANDS.items():
    # Boolean mask for target frequency band
    band_mask = np.logical_and(freqs >= f_low, freqs <= f_high)

    # Calculate Absolute Band Power via numerical integration
    band_abs = np.sum(psds[:, :, band_mask], axis=-1) * df

    # Calculate Relative Band Power
    band_rel = band_abs / total_power

    abs_powers[band_name] = band_abs  # Shape: (n_epochs, n_channels)
    rel_powers[band_name] = band_rel  # Shape: (n_epochs, n_channels)

import pandas as pd

# -----------------------------------------------------------------------------
# Section 6: Visualization of PSD and Topographic Maps
# -----------------------------------------------------------------------------
# Plot Power Spectral Density (PSD) across channels
# (plots the averaged PSD curve across all channels)
spectrum.plot(picks='eeg')

# Plot topographic map of power across frequency bands
spectrum.plot_topomap(ch_type='eeg', bands=FREQ_BANDS)

# -----------------------------------------------------------------------------
# Section 7: Per-Channel Band Power Summary Table
# -----------------------------------------------------------------------------
# 1. Get channel names directly from the spectrum object to avoid mismatch
ch_names = spectrum.ch_names

# 2. Build summary dictionary with matched array lengths
summary_data = {'Channel': ch_names}

for band_name, rel_power_matrix in rel_powers.items():
    # rel_power_matrix shape is (n_epochs, n_channels)
    # Average across all epochs (axis=0) to get mean power per channel
    mean_band_power = np.mean(rel_power_matrix, axis=0)
    summary_data[band_name] = mean_band_power

# 3. Create and display the DataFrame
df_summary = pd.DataFrame(summary_data)

# Round numbers to 4 decimal places for clean visualization
df_summary = df_summary.round(4)

# Print full table
print(df_summary)

import matplotlib.pyplot as plt
spectrum.plot(dB=True)
plt.show()

# -----------------------------------------------------------------------------
# Section 8: Build Feature Matrix (Horizontal Concatenation of Band Features)
# -----------------------------------------------------------------------------
# Get the number of EEG channels directly from the epochs object
# (without depending on power_array)
eeg_ch_names = epochs.copy().pick('eeg').ch_names
print("number of EEG channels:", len(eeg_ch_names))
print("channels name:", eeg_ch_names)

print("Channel Names:")
print(eeg_ch_names)
print(f"\n number of columns: {len(eeg_ch_names)}")

# Make sure eeg_ch_names contains only EEG channels
eeg_ch_names = epochs.copy().pick('eeg').ch_names

# Build the feature matrix from the relative power of each band
feature_dfs = []

for band_name, power_array in rel_powers.items():
    col_names = [f"{ch}_{band_name}_rel" for ch in eeg_ch_names]
    # Verify dimension match
    print(f"Band {band_name}: shape={power_array.shape}, target columns={len(col_names)}")
    df_band = pd.DataFrame(power_array, columns=col_names)
    feature_dfs.append(df_band)

# Horizontal merge of the matrices (side-by-side column concatenation)
X = pd.concat(feature_dfs, axis=1)

print("Final feature matrix shape:", X.shape)

# -----------------------------------------------------------------------------
# Section 9: Long-Format Band Power Table and CSV Export
# -----------------------------------------------------------------------------
# ============================================================
# Band Power - epoch+ DataFrame
# ============================================================

import pandas as pd

ch_names = epochs.copy().pick('eeg').ch_names
subject_id = "Ghafari_002"
session = "Pre"

records = []

for band in ['Delta', 'Theta', 'Alpha', 'Beta']:
    abs_mean = abs_powers[band].mean(axis=0)      # mean across epochs
    rel_mean = rel_powers[band].mean(axis=0)

    # standard deviation across epochs
    abs_std = abs_powers[band].std(axis=0)
    rel_std = rel_powers[band].std(axis=0)

    for i, ch in enumerate(ch_names):
        records.append({
            "subject_id": subject_id,
            "session": session,
            "channel": ch,
            "band": band,
            "abs_power_mean": abs_mean[i],
            "abs_power_std": abs_std[i],
            "rel_power_mean": rel_mean[i],
            "rel_power_std": rel_std[i]
        })

df_bandpower = pd.DataFrame(records)

print(df_bandpower.head(12))
print("\nShape:", df_bandpower.shape)

# Save to CSV
df_bandpower.to_csv(f"{subject_id}_{session}_bandpower_final.csv", index=False)
print("✅ Band Power Saved")

# -----------------------------------------------------------------------------
# Section 10: Hjorth Parameters + Sample Entropy (Per-Epoch → Average)
# ============================================================

import numpy as np
import pandas as pd
from antropy import sample_entropy, hjorth_params

# ----------------------------------------------------------
# 1. Get clean EEG data from epochs
# Shape: (n_epochs, n_channels, n_times)
# ----------------------------------------------------------
data = epochs.get_data(picks="eeg")          # only EEG channels
ch_names = epochs.copy().pick("eeg").ch_names
n_epochs, n_channels, n_times = data.shape

print(f"Data shape: {data.shape}")
print(f"Channels: {ch_names}")

# ----------------------------------------------------------
# 2. Initialize storage lists
# ----------------------------------------------------------
records = []

# ----------------------------------------------------------
# 3. Loop over each channel
# ----------------------------------------------------------
for ch_idx, ch_name in enumerate(ch_names):

    # Extract all epochs of this channel → shape: (n_epochs, n_times)
    ch_data = data[:, ch_idx, :]

    # Storage for this channel
    activity_list = []
    mobility_list = []
    complexity_list = []
    sampen_list = []

    # ------------------------------------------------------
    # 4. Loop over each epoch of the current channel
    # ------------------------------------------------------
    for ep in range(n_epochs):
        signal = ch_data[ep, :]               # single epoch signal

        # --- Hjorth Parameters ---
        # antropy.hjorth_params returns (mobility, complexity)
        mobility, complexity = hjorth_params(signal)

        # Hjorth Activity = variance of the signal
        activity = np.var(signal)

        # --- Sample Entropy ---
        # Common parameters for EEG: order=2, metric='chebyshev'
        sampen = sample_entropy(signal, order=2, metric="chebyshev")

        # Store results of this epoch
        activity_list.append(activity)
        mobility_list.append(mobility)
        complexity_list.append(complexity)
        sampen_list.append(sampen)

    # ------------------------------------------------------
    # 5. Average across epochs for this channel
    # ------------------------------------------------------
    records.append({
        "subject_id": "Ghafari_002",
        "session": "Pre",
        "channel": ch_name,
        "hjorth_activity_mean": np.mean(activity_list),
        "hjorth_activity_std": np.std(activity_list),
        "hjorth_mobility_mean": np.mean(mobility_list),
        "hjorth_mobility_std": np.std(mobility_list),
        "hjorth_complexity_mean": np.mean(complexity_list),
        "hjorth_complexity_std": np.std(complexity_list),
        "sample_entropy_mean": np.mean(sampen_list),
        "sample_entropy_std": np.std(sampen_list)
    })

# ----------------------------------------------------------
# 6. Create final DataFrame
# ----------------------------------------------------------
df_complexity = pd.DataFrame(records)

print(df_complexity.head(10))
print("\nShape:", df_complexity.shape)   # should be (19, 11)
