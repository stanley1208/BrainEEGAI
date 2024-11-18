import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.integrate import simps


# def bandpower(data, sf, band, window_sec=None, relative=False):
#     """Compute the average power of the signal x in a specific frequency band.
#
#     Parameters
#     ----------
#     data : 1d-array
#         Input signal in the time-domain.
#     sf : float
#         Sampling frequency of the data.
#     band : list
#         Lower and upper frequencies of the band of interest.
#     window_sec : float
#         Length of each window in seconds.
#         If None, window_sec = (1 / min(band)) * 2
#     relative : boolean
#         If True, return the relative power (= divided by the total power of the signal).
#         If False (default), return the absolute power.
#
#     Return
#     ------
#     bp : float
#         Absolute or relative band power.
#     """
#     from scipy.signal import welch
#     from scipy.integrate import simps
#     band = np.asarray(band)
#     low, high = band
#
#     # Define window length
#     if window_sec is not None:
#         nperseg = window_sec * sf
#     else:
#         nperseg = (2 / low) * sf
#
#     # Compute the modified periodogram (Welch)
#     freqs, psd = welch(data, sf, nperseg=nperseg)
#
#     # Frequency resolution
#     freq_res = freqs[1] - freqs[0]
#
#     # Find closest indices of band in frequency vector
#     idx_band = np.logical_and(freqs >= low, freqs <= high)
#
#     # Integral approximation of the spectrum using Simpson's rule.
#     bp = simps(psd[idx_band], dx=freq_res)
#
#     if relative:
#         bp /= simps(psd, dx=freq_res)
#     return bp


def bandpower(data, sf, band, method='welch', window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Requires MNE-Python >= 0.14.

    Parameters
    ----------
    data : 1d-array
      Input signal in the time-domain.
    sf : float
      Sampling frequency of the data.
    band : list
      Lower and upper frequencies of the band of interest.
    method : string
      Periodogram method: 'welch' or 'multitaper'
    window_sec : float
      Length of each window in seconds. Useful only if method == 'welch'.
      If None, window_sec = (1 / min(band)) * 2.
    relative : boolean
      If True, return the relative power (= divided by the total power of the signal).
      If False (default), return the absolute power.

    Return
    ------
    bp : float
      Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    from mne.time_frequency import psd_array_multitaper

    band = np.asarray(band)
    low, high = band

    # Compute the modified periodogram (Welch)
    if method == 'welch':
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf

        freqs, psd = welch(data, sf, nperseg=nperseg)

    elif method == 'multitaper':
        psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                          normalization='full', verbose=0)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp



def plot_spectrum_methods(data, sf, window_sec, band=None, dB=False):
    """Plot the periodogram, Welch's and multitaper PSD.

    Requires MNE-Python >= 0.14.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds for Welch's PSD
    dB : boolean
        If True, convert the power to dB.
    """
    from mne.time_frequency import psd_array_multitaper
    from scipy.signal import welch, periodogram
    sns.set(style="white", font_scale=1.2)
    # Compute the PSD
    freqs, psd = periodogram(data, sf)
    freqs_welch, psd_welch = welch(data, sf, nperseg=window_sec*sf)
    psd_mt, freqs_mt = psd_array_multitaper(data, sf, adaptive=True,
                                            normalization='full', verbose=0)
    sharey = False

    # Optional: convert power to decibels (dB = 10 * log10(power))
    if dB:
        psd = 10 * np.log10(psd)
        psd_welch = 10 * np.log10(psd_welch)
        psd_mt = 10 * np.log10(psd_mt)
        sharey = True

    # Start plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=sharey)
    # Stem
    sc = 'slategrey'
    ax1.stem(freqs, psd, linefmt=sc, basefmt=" ", markerfmt=" ")
    ax2.stem(freqs_welch, psd_welch, linefmt=sc, basefmt=" ", markerfmt=" ")
    ax3.stem(freqs_mt, psd_mt, linefmt=sc, basefmt=" ", markerfmt=" ")
    # Line
    lc, lw = 'k', 2
    ax1.plot(freqs, psd, lw=lw, color=lc)
    ax2.plot(freqs_welch, psd_welch, lw=lw, color=lc)
    ax3.plot(freqs_mt, psd_mt, lw=lw, color=lc)
    # Labels and axes
    ax1.set_xlabel('Frequency (Hz)')
    if not dB:
        ax1.set_ylabel('Power spectral density (V^2/Hz)')
    else:
        ax1.set_ylabel('Decibels (dB / Hz)')
    ax1.set_title('Periodogram')
    ax2.set_title('Welch')
    ax3.set_title('Multitaper')
    if band is not None:
        ax1.set_xlim(band)
    ax1.set_ylim(ymin=0)
    ax2.set_ylim(ymin=0)
    ax3.set_ylim(ymin=0)
    sns.despine()
    plt.show()


def strongest_band_in_periods(data, sf, bands, window_sec):
    """
    Find the strongest frequency band for each time period.

    Parameters
    ----------
    data : 1d-array
        EEG signal data.
    sf : float
        Sampling frequency.
    bands : dict
        Frequency bands (e.g., {"Delta": [0.5, 4], "Theta": [4, 8]}).
    window_sec : float
        Length of each window in seconds.

    Returns
    -------
    results : list
        List of strongest bands for each segment.
    """
    nperseg = int(window_sec * sf)
    step = nperseg
    results = []

    for start in range(0, len(data) - nperseg + 1, step):
        segment = data[start:start + nperseg]
        print(segment)
        band_powers = {band: bandpower(segment, sf, freq_range) for band, freq_range in bands.items()}
        strongest_band = max(band_powers, key=band_powers.get)
        results.append((start / sf, (start + nperseg) / sf, strongest_band))

    return results




# Path of the uploaded file in this environment
file_path='C:/Users/user/Desktop/WenEEG.xlsx'
df = pd.read_excel(file_path, engine='openpyxl',skiprows=4)

print(df.head())  # Print the first few rows to verify the data


sns.set(font_scale=1.2)

df['EXG Channel 0'] = pd.to_numeric(df['EXG Channel 0'], errors='coerce')
# Define sampling frequency and time vector
sf = 100.  # Sampling frequency in Hz
time = np.arange(len(df)) / sf  # Create a time vector based on the number of samples

# Plot the "EXG Channel 0" signal
# fig, ax = plt.subplots(1, 1, figsize=(12, 4))
# plt.plot(time, df['EXG Channel 0'], lw=1.5, color='k')
# plt.xlabel("Time (seconds)")
# plt.ylabel("Voltage (EXG Channel 0)")
# plt.xlim([time.min(), time.max()])
# plt.title('N3 sleep EEG data - EXG Channel 0')
# sns.despine()
# plt.show()


# Define window length (4 seconds)
win=4*sf
freqs,psd=signal.welch(df['EXG Channel 0'],sf,nperseg=win)
#
# # Plot the power spectrum
# sns.set(font_scale=1.2,style='white')
# plt.figure(figsize=(8,4))
# plt.plot(freqs,psd,lw=2,color='k')
# plt.xlabel("Frequency (Hz)")
# plt.ylabel('Power spectral density (V^2/HZ)')
# plt.ylim([0,psd.max()*1.1])
# plt.title("Welch's periodogram")
# plt.xlim([0,freqs.max()])
# sns.despine()
# plt.show()


# Define delta lower and upper limits
low,high=0.5,4

# Find intersecting values in frequency vector
idx_delta=np.logical_and(freqs>=low,freqs<=high)

# # Plot the power spectral density and fill the delta area
# plt.figure(figsize=(7,4))
# plt.plot(freqs,psd, lw=2,color='k')
# plt.fill_between(freqs,psd,where=idx_delta,color='skyblue')
# plt.xlabel("Frequency (Hz)")
# plt.ylabel('Power spectral density (V^2/HZ)')
# plt.xlim([0,10])
# plt.ylim([0,psd.max()*1.1])
# plt.title("Welch's periodogram")
# sns.despine()
# plt.show()


# Frequency resolution
freq_res=freqs[1]-freqs[0]  # = 1 / 4 = 0.25

# Compute the absolute power by approximating the area under the curve
delta_power=simps(psd[idx_delta],dx=freq_res)
print("Absolute delta power: %.3f  uV^2"%delta_power)



# Relative delta power (expressed as a percentage of total power)
total_power=simps(psd, dx=freq_res)
delta_rel_power=delta_power/total_power
print('Relative delta power: %.3f'%delta_rel_power)


# Define the duration of the window to be 4 seconds
win_sec=4


# # # Delta/beta ratio based on the absolute power
# db=bandpower(df['EXG Channel 0'],sf,[0.5,4],win_sec)/bandpower(df['EXG Channel 0'],sf,[12,30],win_sec)
#
# # Delta/beta ratio based on the relative power
# db_rel=bandpower(df['EXG Channel 0'],sf,[0.5,4],win_sec,True)/bandpower(df['EXG Channel 0'],sf,[12,30],win_sec,True)
#
# print('Delta/beta ratio (absolute): %.3f)'%db)
# print('Delta/beta ratio (relative): %.3f)'%db_rel)



# Multitaper delta power
bp=bandpower(df['EXG Channel 0'],sf,[0.5,4],'multitaper')
bp_rel=bandpower(df['EXG Channel 0'],sf,[0.5,4],'multitaper',relative=True)
print('Absolute delta power: %.3f'%bp)
print('Relative delta power (multitaper): %.3f'%bp_rel)


# Delta-beta ratio
# One advantage of the multitaper is that we don't need to define a window length.
db=bandpower(df['EXG Channel 0'],sf,[0.5,4],'multitaper')/bandpower(df['EXG Channel 0'],sf,[12,30],'multitaper')
# Ratio based on the relative power
db_rel=bandpower(df['EXG Channel 0'],sf,[0.5,4],'multitaper',relative=True)/bandpower(df['EXG Channel 0'],sf,[12,30],'multitaper',relative=True)

print('delta/beta ratio(absolute): %.3f'%db)
print('delta/beta ratio(relative): %.3f'%db_rel)




# Example: plot the 0.5 - 2 Hz band
plot_spectrum_methods(df['EXG Channel 0'], sf, 4, [0.5, 2], dB=True)





# Define frequency bands
bands = {
    "Delta": [0.5, 4],
    "Theta": [4, 8],
    "Alpha": [8, 12],
    "Beta": [12, 30],
}

# Parameters
window_sec = 4  # 4-second windows

# Find the strongest band in each time period
results = strongest_band_in_periods(df['EXG Channel 0'], sf, bands, window_sec)

# Print results
for start, end, band in results:
    print(f"Time: {start:.2f}-{end:.2f}s | Strongest Band: {band}")






