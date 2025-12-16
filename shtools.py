import numpy as np
from utils import nextpow2
from scipy.signal import lfilter
from scipy.special import sph_harm, spherical_jn, spherical_yn
from scipy.linalg import cho_factor, cho_solve
import warnings


def get_k(freq, c=343.0):
    # Get wavenumber k.
    return 2 * np.pi * freq / c         
        
def get_acn_order_array(order: int) -> np.ndarray:
    """Vectorized ACN order array: [0, 1,1,1, 2,2,2,2,2, ...]."""
    if order < 0:
        raise ValueError("order must be >= 0")
    # counts per order: 1, 3, 5, ..., 2n+1
    counts = 2 * np.arange(order + 1) + 1
    return np.repeat(np.arange(order + 1), counts)

def sph_bn(n_arr, x_arr, isRigid=False):
    """
    Spherical radial function b_n for open (isRigid=False) or rigid (isRigid=True) sphere.

    Parameters
    ----------
    n_arr : array_like of ints, shape (N,) or scalar
        Orders n.
    x_arr : array_like of floats or complex, shape (K,) or scalar
        Arguments x = k*r.
    isRigid : bool, default False
        If True, applies rigid-sphere boundary condition (Neumann),
        i.e., b_n = j_n(x) - [ j_n'(x) * h_n^(1)(x) / h_n^(1)'(x) ]  (with NaNs -> 0)
        and returns the complex conjugate (to match the original MATLAB code).

    Returns
    -------
    bn : ndarray, shape (N, K)
        Values of b_n for each n in n_arr and x in x_arr.
    """
    # Ensure 2D broadcastable shapes: (N,1) and (1,K)
    n = np.atleast_1d(n_arr).astype(int).reshape(-1, 1)
    x = np.atleast_1d(x_arr).reshape(1, -1)

    # Base term j_n(x)
    jn = spherical_jn(n, x)  # (N,K)

    if not isRigid:
        # Open sphere: b_n = j_n(x)
        return jn        

    # Avoid divisions by zero -> set NaNs/Infs to 0 like MATLAB code
    with np.errstate(divide='ignore', invalid='ignore'):
        # Rigid sphere correction: j_n'(x) * h_n^(1)(x) / h_n^(1)'(x)
        dj = spherical_jn(n, x, derivative=True)
        h1 = spherical_jn(n, x) + 1j*spherical_yn(n, x)
        dh = spherical_jn(n, x, derivative=True) + 1j*spherical_yn(n, x, derivative=True)
        corr = dj * h1 / dh
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    bn = jn - corr

    # Match MATLAB’s final conjugation for rigid arrays
    return np.conj(bn)

def get_complex_sh(order, el, az):
    """
    Complex spherical harmonics up to 'order' (inclusive).

    Parameters
    ----------
    order : int
        Highest SH order N (returns (N+1)^2 rows).
    el : array_like
        Elevation(s) in radians (0..pi), length Q or scalar.
    az : array_like
        Azimuth(s) in radians (0..2*pi), length Q or scalar.

    Returns
    -------
    y : np.ndarray, complex128
        Array of shape ((order+1)**2, Q) with rows arranged in ACN-like
        stacking per order: for each n=0..N, m=-n..n.
        (Same layout as the original MATLAB function.)
    """
    el = np.asarray(el).reshape(-1)   # theta
    az = np.asarray(az).reshape(-1)   # phi
    Q = el.size
    rows = (order + 1) ** 2
    Y = np.empty((rows, Q), dtype=np.complex128)
    idx = 0
    for n in range(order + 1):
        m_vals = np.arange(-n, n + 1)
        # OJO: sph_harm(m, n, phi, theta) => (m, n, az, el)
        Y[idx:idx + 2*n + 1, :] = sph_harm(m_vals[:, None], n, az[None, :], el[None, :])
        idx += 2*n + 1
    return Y

def get_real_sh(order, el, az):
    """
    Real spherical harmonics up to 'order' (inclusive), matching SHTools.getRealSH.

    Parameters
    ----------
    order : int
        SH order N. Output has (N+1)^2 rows.
    el : array_like
        Elevation(s) in radians (0..pi), length Q or scalar.
    az : array_like
        Azimuth(s) in radians (0..2*pi), length Q or scalar.

    Returns
    -------
    Y : np.ndarray, float64
        Shape ((order+1)**2, Q). Rows are stacked by order n=0..N and mode m=-n..n,
        using the same normalization and real-basis construction as the MATLAB code:
        m=0 term, positive m with sqrt(2)*cos(m*az), negative m with sqrt(2)*sin(|m|*az),
        and the (-1)^m factor in the normalization.
    """
    el = np.asarray(el).reshape(-1)   # theta
    az = np.asarray(az).reshape(-1)   # phi
    Q = el.size
    rows = (order + 1) ** 2
    Y = np.empty((rows, Q), dtype=np.float64)
    idx = 0
    for n in range(order+1):
        for m in range(-n,n+1,1):
            sh_value = sph_harm(abs(m),n,az, el) # m >= 0
            if m < 0:
                # Negative orders contain the imaginary part
                Y[idx,:] = np.sqrt(2) * (-1)**m * sh_value.imag
            elif m == 0:
                # Zero order is real
                Y[idx,:] = sh_value.real
            else:
                # Positive orders contain the real part
                Y[idx,] = np.sqrt(2) * (-1)**m * sh_value.real
            idx+=1
    return Y
    
def pinv_regularized(A: np.ndarray, b: np.ndarray, weight: float = 0.01) -> np.ndarray:
    """
    Solve (A x ≈ b) with Tikhonov regularization (ridge):
        min_x ||A x - b||_2^2 + weight^2 ||x||_2^2
    Implemented by stacking for stability (works with complex too).

    A: (M, N), b: (M, T)  ->  x: (N, T)
    """
    M, N = A.shape
    # Stack A over (weight * I)
    A_aug = np.vstack([A, weight * np.eye(N, dtype=A.dtype)])
    # Stack b over zeros
    b_aug = np.vstack([b, np.zeros((N, b.shape[1]), dtype=b.dtype)])
    # Least-squares solve once; rcond chosen by NumPy default
    x, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    return x

def pinv_tikhonov_cholesky(A: np.ndarray, B: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    """
    Solve X = argmin ||A X - B||_F^2 + lam^2 ||X||_F^2
    via normal equations: (A^H A + lam^2 I) X = A^H B
    using Cholesky (Hermitian PD).
    A: (M, N), B: (M, T)  ->  X: (N, T)
    """
    # G = A^H A + lam^2 I
    G = A.conj().T @ A
    n = G.shape[0]
    G[np.diag_indices(n)] += lam * lam
    c, lower = cho_factor(G, overwrite_a=True, check_finite=False)
    return cho_solve((c, lower), A.conj().T @ B, check_finite=False)

def alpha_omni_mic_array(
    P: np.ndarray,
    order: int,
    kr,
    el: np.ndarray,
    az: np.ndarray,
    isRigid: bool = False,
    mode: str = "inv",                 # 'inv' (pseudoinverse) or 'mm' (mode matching)
    compensate_bessel_zero: bool = False,
    apply_regularization: bool = False,
    weight=1.0,
    reg_weight: float = 0.01
) -> np.ndarray:
    """
    Python port of MATLAB SHTools.alphaOmniMicArray.

    Parameters
    ----------
    P : ndarray, shape (Q, T)
        Sound pressure matrix (Q microphones x T time frames).
    order : int
        Intended sound-field order (should be <= max(kr) typically).
    kr : float or array_like
        k * r for each mic. Can be scalar or array of length Q.
    el, az : array_like, shape (Q,)
        Microphone elevations and azimuths (radians).
    isRigid : bool
        1 for rigid sphere, 0 for open.
    mode : {'inv','mm'}
        'inv': pseudoinverse (with optional regularization).
        'mm' : mode matching (diagonal inverse bn).
    compensate_bessel_zero : bool
        If True, floor |bn| to 0.05 (preserve phase) to avoid divisions near zeros.
    apply_regularization : bool
        If True (with mode='inv'), use Tikhonov regularized pseudo-inverse.
    weight : float or array_like
        Array weight per microphone (Q,) or scalar 1.0.
    reg_weight : float
        Tikhonov regularization weight for the 'inv' branch.

    Returns
    -------
    Alpha : ndarray, shape ((order+1)^2, T)
        Estimated sound-field SH coefficients.
    """
    # Ensure array shapes
    P = np.asarray(P)
    el = np.asarray(el).reshape(-1)
    az = np.asarray(az).reshape(-1)

    Q = el.size
    if P.shape[0] != Q:
        raise ValueError(f"P has {P.shape[0]} rows but el/az define Q={Q} mics.")

    # kr can be scalar or length-Q
    kr_arr = np.asarray(kr, dtype=float).reshape(-1)
    if kr_arr.size == 1:
        kr_arr = np.full(Q, kr_arr.item(), dtype=float)
    elif kr_arr.size != Q:
        raise ValueError("kr must be scalar or length-Q array.")

    # Important if true order of the sound field > N (as in MATLAB):
    # N_true = max(order, max(ceil(kr)))
    N_true = max(int(order), int(np.ceil(np.max(kr_arr))))

    # Orders array in ACN stacking: (N_true+1)^2
    n_arr_true = get_acn_order_array(N_true)

    # bn: ((N_true+1)^2, Q) (or ((N_true+1)^2, 1) if you pass scalar kr)
    bn_true = sph_bn(n_arr_true, kr_arr, isRigid=isRigid)  # shape ((N_true+1)^2, Q)

    if mode.lower() == "inv":
        # Build Y_mat at N_true: ((N_true+1)^2, Q)
        Y_mat = get_complex_sh(N_true, el, az) * bn_true  # elementwise across columns
        # Solve Alpha from Y^T * Alpha ≈ P  =>  (Q x NN) * (NN x T) ≈ (Q x T)
        YT = Y_mat.T  # shape (Q, NN_true)
        if apply_regularization:
            #Alpha = pinv_regularized(YT, P, weight=reg_weight)
            Alpha = pinv_tikhonov_cholesky(YT, P, weight=reg_weight)
        else:
            # Complex pseudo-inverse solve
            Alpha = np.linalg.pinv(YT) @ P

        # If requested order is smaller than N_true, truncate
        if order < N_true:
            keep = (order + 1) ** 2
            Alpha = Alpha[:keep, :]

        return Alpha

    elif mode.lower() == "mm":
        # Use intended order N for mode matching
        n_arr = get_acn_order_array(order)

        # bn for N (truncate bn_true or recompute for clarity)
        # Recompute for clarity/readability:
        bn = sph_bn(n_arr, kr_arr, isRigid=isRigid)  # ((N+1)^2, Q)

        # Optional flooring on bn magnitude (preserve phase)
        if compensate_bessel_zero:
            mag = np.abs(bn)
            ang = np.angle(bn)
            mag = np.maximum(mag, 0.05)
            bn = mag * np.exp(1j * ang)

        # Y_mat = conj(SH_N(el,az)) ./ bn   (elementwise div per column)
        Y = get_complex_sh(order, el, az)  # ((N+1)^2, Q)
        Y_mat = np.conj(Y) / bn

        # Apply per-mic weights (multiply each column)
        if np.isscalar(weight):
            if weight != 1.0:
                Y_mat = Y_mat * weight
        else:
            w = np.asarray(weight).reshape(1, -1)   # (1, Q)
            if w.shape[1] != Q:
                raise ValueError("`weight` must be scalar or length-Q array.")
            Y_mat = Y_mat * w

        # Alpha = Y_mat @ P  -> ((N+1)^2, T)
        Alpha = Y_mat @ P
        return Alpha

    else:
        raise ValueError("`mode` must be 'inv' or 'mm'.")
    
def sound_pressure_from_measured_alpha(Alpha: np.ndarray, kr, el: np.ndarray, az: np.ndarray, isRigid: bool):
    """
    Python port of SHTools.soundPressureFromMeasuredAlpha (MATLAB)

    Parameters
    ----------
    Alpha : ndarray, shape ((N+1)^2, T)
        Spherical harmonic coefficients (measured or estimated).
    kr : float or array_like
        Wave number * microphone radius (k * r). Can be scalar or vector of length Q.
    el, az : ndarray, shape (Q,)
        Microphone elevations and azimuths in radians.
    isRigid : bool
        True for rigid-sphere array, False for open-sphere array.
    Returns
    -------
    P : ndarray, shape (Q, T)
        Reconstructed sound pressure at each microphone and time frame.
    """
    Alpha = np.asarray(Alpha)
    el = np.asarray(el).reshape(-1)
    az = np.asarray(az).reshape(-1)

    Q = el.size
    if az.size != Q:
        raise ValueError("`el` and `az` must have the same length (Q).")

    # Infer spherical harmonic order N from Alpha dimension
    NN = Alpha.shape[0]
    N_float = np.sqrt(NN) - 1.0
    N = int(round(N_float))
    if (N + 1) ** 2 != NN:
        raise ValueError("The number of rows in Alpha is not a valid (N+1)^2 value.")

    # kr can be scalar or vector of length Q
    kr_arr = np.asarray(kr, dtype=float).reshape(-1)
    if kr_arr.size == 1:
        kr_arr = np.full(Q, kr_arr.item(), dtype=float)
    elif kr_arr.size != Q:
        raise ValueError("`kr` must be scalar or a vector of length Q.")

    # Warnings equivalent to MATLAB implementation
    if N < int(np.ceil(np.max(kr_arr))):
        warnings.warn("Not enough modes in Alpha for the given kr.", RuntimeWarning)
    if NN > Q:
        warnings.warn("Not enough microphones in the array for the given order.", RuntimeWarning)

    # ACN order array: [0, 1,1,1, 2,2,2,2,2, ...]
    n_arr = get_acn_order_array(N)  # shape ((N+1)^2,)

    # Compute radial function bn: ((N+1)^2, Q)
    bn = sph_bn(n_arr, kr_arr, isRigid=isRigid)

    # Compute complex spherical harmonics matrix: ((N+1)^2, Q)
    Y = get_complex_sh(N, el, az)

    # Apply bn to each column of Y (element-wise)
    Y_mat = Y * bn

    # Compute sound pressure: P = (Y_mat^T) * Alpha  ->  (Q x T)
    P = Y_mat.T @ Alpha
    return P

# --- FUNCIONES CALCULO HOA (PORTED FROM PEDRO VERA MATLAB CODE) ---
def compute_bn_fir_filters(sh_order: int, fs: float, b_n_len: int,
                           c: float, rmic: float, isRigid: bool):
    """
    Python port of MATLAB compute_bn_fir_filters.m
    Compute SH b_n radial filters and their inverse FIRs (linear phase).
    Parameters
    ----------
    sh_order : Spherical harmonic order.
    fs : sampling frequency [Hz].
    b_n_len : Length of FIR filters in samples.
    c : Speed of sound [m/s].
    rmic : Radius of the microphone array [m].
    isRigid : Whether the microphone array is rigid (True) or open (False).

    Returns
    -------
    b_n_in_time : (ncoeff, b_n_len) float64
    b_n_inverse_in_time : (ncoeff, b_n_len) float64
    """

    # -----------------------------
    # Thresholds per SH order
    # -----------------------------
    thresholds_order = np.ones(sh_order + 1, dtype=float) * 0.01
    if sh_order >= 1:
        thresholds_order[1] = 0.2  # MATLAB: thresholds_order(2)=0.2

    ncoeff = (sh_order + 1) ** 2
    thresholds = np.zeros(ncoeff, dtype=float)
    thresholds[0] = thresholds_order[0]

    ind = 1
    for n in range(1, sh_order + 1):
        numModes = 2 * n  # so ind : ind+numModes has length (2n+1)
        thresholds[ind: ind + numModes + 1] = thresholds_order[n]
        ind += numModes + 1

    # -----------------------------
    # FFT size & frequency grid
    # -----------------------------
    nfft = 2 ** nextpow2(b_n_len + 1)
    nf_half = nfft // 2 + 1

    freqs_stft_filters = (fs / 2.0) * np.linspace(0.0, 1.0, nf_half)
    nf = freqs_stft_filters.size

    # -----------------------------
    # Compute b_n originals & thresholded
    # -----------------------------
    n_arr = get_acn_order_array(sh_order)  # (ncoeff,)
    b_n_originals = np.zeros((ncoeff, nf), dtype=np.complex128)
    b_n = np.zeros((ncoeff, nf), dtype=np.complex128)

    prev_mag = None
    for fidx, f_hz in enumerate(freqs_stft_filters):
        k_tmp = 2.0 * np.pi * f_hz / c
        bn_orig = sph_bn(n_arr, k_tmp * rmic, isRigid=isRigid).reshape(-1)  # (ncoeff,)
        b_n_originals[:, fidx] = bn_orig
        b_n[:, fidx] = bn_orig

        mag = np.abs(bn_orig)
        ph = np.angle(bn_orig)

        if fidx == 0:
            # MATLAB: b_n(:,1) = max(abs(..), thresholds).*exp(1i*angle(..))
            mag_thr = np.maximum(mag, thresholds)
            b_n[:, fidx] = mag_thr * np.exp(1j * ph)
        else:
            # Only limit when increasing in frequency: (|bn(f)| - |bn(f-1)|) >= 0
            derivative = (mag - prev_mag) >= 0.0
            if np.any(derivative):
                mag_thr = np.maximum(np.abs(b_n[derivative, fidx]), thresholds[derivative])
                b_n[derivative, fidx] = mag_thr * np.exp(1j * np.angle(b_n[derivative, fidx]))

        prev_mag = mag

    # MATLAB: last bin (Nyquist) must be real
    b_n[:, -1] = np.abs(b_n[:, -1])

    # -----------------------------
    # Build linear-phase FIRs in time
    # -----------------------------
    b_n_in_time = np.zeros((ncoeff, b_n_len), dtype=np.float64)
    b_n_inverse_in_time = np.zeros((ncoeff, b_n_len), dtype=np.float64)

    # omega grid for 0..Nyquist bins
    omega = 2.0 * np.pi * np.arange(nf_half) / nfft
    delay = b_n_len / 2.0  # linear phase delay

    eps = 1e-12

    for shc in range(ncoeff):
        # ---- FIR for b_n (linear phase) ----
        H = np.abs(b_n[shc, :]) * np.exp(-1j * omega * delay)
        H[-1] = np.real(H[-1])  # enforce Nyquist real (safety)

        h_time = np.fft.irfft(H, n=nfft)
        b_n_in_time[shc, :] = np.real(h_time[:b_n_len])

        # ---- Inverse FIR: 1 / B_n_in_freq, then linear phase ----
        B = np.fft.rfft(b_n_in_time[shc, :], n=nfft)  # (nf_half,)
        Bmag = np.abs(B)
        inv_mag = 1.0 / np.maximum(Bmag, eps)

        Hinv = inv_mag * np.exp(-1j * omega * delay)
        Hinv[-1] = np.real(Hinv[-1])

        hinv_time = np.fft.irfft(Hinv, n=nfft)
        b_n_inverse_in_time[shc, :] = np.real(hinv_time[:b_n_len])

    return b_n_in_time, b_n_inverse_in_time


def hoa_omni_mic_array(audio: np.ndarray, fs: int, order: int, rmic: float, mic_az: np.ndarray, mic_el: np.ndarray,
                       isRigid: bool, b_n_len: float) -> np.ndarray:
    """
    Encode audio signals from omnidirectional microphone array into HOA spherical harmonic coefficients.
    Parameters
    ----------
    audio : np.ndarray, shape (nmic, nsamples)
        Audio signals from the omnidirectional microphone array.
    fs : int
        Sampling frequency in Hz.
    order : int
        Spherical harmonic order.
    rmic : float 
        Radius of the microphone array sphere (in meters).
    mic_az : np.ndarray, shape (nmic,)
        Microphone azimuths in radians.
    mic_el : np.ndarray, shape (nmic,)
        Microphone elevations in radians.     
    isRigid : bool, 
        Whether the microphone array is rigid. Default is True.
    b_n_len : float, 
        Length of the FIR filters used for b_n. Default is 2048.
    Returns
    -------
    hoa_signal : np.ndarray, shape ((order+1)^2, nsamples)
        HOA spherical harmonic coefficients.
    """
    c = 343.0  # Speed of sound in m/s   
    b_n_len = 2048

    # 1) Descomposición esferica: Y, S, p_nm
    Y = get_real_sh(order, mic_el, mic_az)      # shape: (ncoeff, nmic)
    S = np.linalg.pinv(Y.T)                     # shape: (ncoeff, nmic)
    audio_SH = np.tensordot(S, audio, axes=([1], [0]))

    # 2) Diseño de filtros FIR b_n y de su inversa en tiempo
    _, b_n_inverse_in_time = compute_bn_fir_filters(order, fs, b_n_len, c,rmic, isRigid)

    # 3) Filtrado en tiempo 
    ncoeff = (order + 1) ** 2
    nsamples = audio_SH.shape[1]
    
    hoa_signal = np.zeros((ncoeff, nsamples + b_n_len))

    for shc in range(ncoeff):  # MATLAB: for shc=1:ncoeff
        hoa_signal[shc, :nsamples] = audio_SH[shc, :]
        hoa_signal[shc, :] = lfilter(b_n_inverse_in_time[shc, :], 1.0, hoa_signal[shc, :])

    return hoa_signal

def sound_pressure_from_hoa(hoa: np.ndarray, fs: int, order: int, rmic: float, mic_az: np.ndarray, mic_el: np.ndarray, 
                            isRigid: bool, b_n_len: float) -> np.ndarray:
    """
    Reconstruct sound pressure at microphones from real spherical harmonic coefficients.
    Parameters
    ----------
    hoa : np.ndarray, shape ((order+1)^2, nsamples)
        hoa encoding the spherical harmonic coefficients.
    fs : int
        Sampling frequency in Hz.
    order : int
        Spherical harmonic order.
    freqs_stft : np.ndarray, shape (nfreq,)
        Frequencies corresponding to the STFT bins.
    rmic : float
        Radius of the microphone array sphere (in meters).
    mic_az : np.ndarray, shape (nmic,)
        Microphone azimuths in radians.
    mic_el : np.ndarray, shape (nmic,)
        Microphone elevations in radians.
    isRigid : bool
        Whether the microphone array is rigid.
    b_n_len : float
        Length of the FIR filters used for b_n.
    Returns
    -------
    P : np.ndarray, shape (nmic, nfreq, nframes)
        Reconstructed sound pressure at each microphone, frequency bin, and time frame.
    """
    c = 343.0  # Speed of sound in m/s
    nsamples = hoa.shape[1]
    
    # Spherical harmonics decomposition
    Y = get_real_sh(order, mic_el, mic_az)  
    ncoeff = (order + 1) ** 2
    
    b_n_in_time, _ = compute_bn_fir_filters(order, fs, b_n_len, c, rmic, isRigid)
    
    audio_SH = np.zeros((ncoeff, nsamples + b_n_len))
    for shc in range(ncoeff):  # MATLAB: for shc = 1:ncoeff
        audio_SH[shc, :nsamples] = hoa[shc, :]
        audio_SH[shc, :] = lfilter(b_n_in_time[shc, :], 1.0, audio_SH[shc, :])

    # Estima la presión a partir de los coeficientes SH (version SHTools)
    return np.tensordot(Y.T, audio_SH, axes=([1], [0]))