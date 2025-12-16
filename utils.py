import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import stft, istft, windows


def nextpow2(n: int) -> int:
    """Return p such that 2**p >= n and p is minimal."""
    if n <= 1:
        return 0
    return int(np.ceil(np.log2(n)))

# Define equivalent of mystft and myistft 
def mystft(x, fs=44100, winlen=2048, hop=None, nfft=None):
    if hop is None:
        hop = winlen // 2
    if nfft is None:
        nfft = 2 ** nextpow2(winlen + 1) 
    window= windows.hamming(winlen, sym=False)
    f, t, Zxx = stft(x, fs=fs, window=window, nfft=nfft, nperseg=winlen, 
                     noverlap=winlen - hop, boundary=None,padded=False, 
                     return_onesided=True) 
    return Zxx 

def myistft(Zxx, fs=44100, winlen=2048, hop=None): 
    if hop is None:
        hop = winlen // 2
    window= windows.hamming(winlen, sym=False)
    _, x_rec = istft(Zxx, fs=fs, window=window, nperseg=winlen,
                     noverlap=winlen - hop, input_onesided=True, boundary=None)
    return np.real(x_rec)

def get_nframes(signal_length, winlen=2048, hoplen=None):
    if hoplen is None:
        hoplen = winlen // 2
    return 1 + int(np.floor((signal_length - winlen) / hoplen))

def get_stft_freqs(fs=44100, winlen=2048):
    return np.linspace(0, fs/2, winlen+1)

def multichannel_fft_convolve(rir_clean: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """
    rir_clean: (nmic, Lh)
    signal:   (Lx,)
    returns:  (nmic, Lx + Lh - 1)
    """
    rir = np.asarray(rir_clean)
    x = np.asarray(signal)

    nmic, Lh = rir.shape
    Lx = x.size
    Lout = Lx + Lh - 1

    # Next power of two for speed
    L = 1 << (Lout - 1).bit_length()

    # Real FFTs to save memory/compute
    X = np.fft.rfft(x, n=L)                 # (Lf,)
    H = np.fft.rfft(rir, n=L, axis=1)       # (nmic, Lf)

    # Broadcast multiply in freq domain
    Y = H * X[None, :]                      # (nmic, Lf)

    # Back to time, trim to 'full' length
    y = np.fft.irfft(Y, n=L, axis=1)[:, :Lout]   # (nmic, Lout)
    return y

def trim_signal_to_activity_old(signal, fs, duration_sec=5, thresh=0.001):
    """
    Trim the input signal to a n-second segment with high activity.

    signal: (L,)
    fs: Sampling frequency
    duration_sec: Duration of the segment to extract in seconds
    thresh: RMS threshold to consider "active"
    """
        
    rms_winlen = fs
    rms_hop = fs // 2
    num_frames_needed = int(duration_sec * fs // rms_hop)

    # Precompute squared signal
    power = signal ** 2

    # Efficient moving average using cumulative sum
    cumsum = np.cumsum(np.insert(power, 0, 0))
    rms_activity = np.sqrt((cumsum[rms_winlen:] - cumsum[:-rms_winlen]) / rms_winlen)
    rms_activity = np.pad(rms_activity, (rms_winlen // 2, rms_winlen - rms_winlen // 2 - 1), mode='constant')

    # Subsample frames
    logical_frames = rms_activity[::rms_hop] > thresh

    # Use sliding window with `as_strided` for fast search
    if len(logical_frames) < num_frames_needed:
        raise RuntimeError('Signal too short for activity window.')

    windows = sliding_window_view(logical_frames, window_shape=num_frames_needed)

    valid_indices = np.where(np.all(windows, axis=1))[0]
    if valid_indices.size == 0:
        raise RuntimeError('No second segment with full activity found.')

    start_idx = valid_indices[0] * rms_hop
    end_idx = start_idx + duration_sec * fs
    signal = signal[start_idx:end_idx]
    
    return signal, start_idx, end_idx


def trim_signal_to_activity(signal, fs, duration_sec=5, thresh=1e-3):
    """
    Trim the input signal to a contiguous n-second segment where 1 s RMS
    (hop = fs/2) stays above `thresh` on every hop.

    Returns:
        segment (np.ndarray)
        start_idx (int)  # 0-based
        end_idx   (int)  # exclusive
    """
    signal = np.asarray(signal)
    L = signal.size
    if L == 0:
        raise RuntimeError("Empty signal.")

    rms_winlen = int(fs)          # 1-second RMS window
    rms_hop = int(fs // 2)        # 0.5-second hop
    if rms_winlen <= 0 or rms_hop <= 0:
        raise ValueError("`fs` must be a positive integer.")

    # Need enough samples to cover at least one RMS frame
    if L < rms_winlen:
        # Fallback: single RMS over the whole signal
        rms_val = np.sqrt(np.mean(signal.astype(float)**2))
        if rms_val <= thresh or L < int(duration_sec * fs):
            raise RuntimeError("Signal too short or below threshold for requested duration.")
        start_idx = 0
        end_idx = min(L, int(duration_sec * fs))
        return signal[start_idx:end_idx], start_idx, end_idx

    # Frame the power with VALID windows starting at sample 0 (no padding)
    power = signal.astype(float)**2
    frame_matrix = sliding_window_view(power, window_shape=rms_winlen)[::rms_hop]  # shape: (num_frames, rms_winlen)
    rms_frames = np.sqrt(frame_matrix.mean(axis=1))

    logical_frames = rms_frames > thresh
    num_frames = logical_frames.size

    # How many consecutive frames are needed to span `duration_sec`
    num_frames_needed = int(np.floor((duration_sec * fs) / rms_hop))
    if num_frames < num_frames_needed or num_frames_needed <= 0:
        raise RuntimeError("Signal too short for activity window.")

    # Find first index where ALL frames in the window are active
    win_view = sliding_window_view(logical_frames, window_shape=num_frames_needed)
    valid_indices = np.flatnonzero(win_view.all(axis=1))
    if valid_indices.size == 0:
        raise RuntimeError("No contiguous segment with full activity found.")

    # Map frame index back to sample index (frame start = i * hop)
    valid_start_idx = valid_indices * rms_hop   
    # end_idx = start_idx + int(duration_sec * fs)
    # end_idx = min(end_idx, L)  # clamp to signal length
    # return signal[start_idx:end_idx], start_idx, end_idx
    return valid_start_idx.astype(int)