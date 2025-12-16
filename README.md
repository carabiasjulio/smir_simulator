# SMIR Simulator

A MATLAB and Python toolkit for generating simulated Room Impulse Responses (RIRs) and encoding them in Higher Order Ambisonics (HOA) format using spherical microphone arrays.

## Overview

This project generates realistic room impulse responses for the Eigenmike spherical microphone array using the SMIR-Generator. The toolkit supports:

- Multi-room RIR generation with customizable acoustic parameters
- HOA encoding up to 6th order spherical harmonics
- Both reverberant and clean (anechoic) impulse responses
- Parallel processing for efficient batch generation
- Python utilities for signal processing and analysis

## Features

- **Acoustic Room Simulation**: Generate RIRs for rooms with configurable dimensions and reverberation times
- **Spherical Microphone Arrays**: Support for Eigenmike (32 capsules) with rigid/open sphere models
- **HOA Encoding**: Encode microphone signals into Higher Order Ambisonics format
- **Flexible Configuration**: YAML-based parameter configuration
- **Batch Processing**: Generate multiple rooms and source positions with parallel workers
- **Signal Processing Tools**: Python utilities for STFT, filtering, and multichannel convolution

## Requirements

### MATLAB
- MATLAB R2024b or later
- Signal Processing Toolbox

### Python
Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- `numpy` - Array processing
- `scipy` - Signal processing and special functions
- `mat73` - MATLAB v7.3 file support
- `matplotlib` - Visualization
- `soundfile` - Audio I/O
- `pyyaml` - Configuration file parsing
- `scikit-image` - Image processing (for tests)

## Project Structure

```
├── config.yaml                          # Main configuration file
├── simulated_rir_generator.m            # Generate RIRs for Eigenmike
├── simulated_hoa_rir_generator.m        # Encode RIRs to HOA format
├── demo_A2B2A.m                         # MATLAB demo: Ambisonics to binaural
├── demo_A2B2A.ipynb                     # Python demo notebook
├── demo_A2B2A_from_rir_mat.ipynb        # Python demo: Load and process RIRs
├── compute_bn_fir_filters.m             # Compute radial filters for HOA
├── utils.py                             # Python utilities (STFT, convolution)
├── shtools.py                           # Spherical harmonics tools
├── homtools.py                          # Eigenmike and HOA grid utilities
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Git ignore file
├── rirs/                                # Generated RIRs directory
│   ├── rir/                            # Room impulse responses
│   ├── rir_clean/                      # Anechoic responses
│   └── info/                           # Room and RIR metadata (CSV)
├── src/
│   ├── SMIR-Generator/                 # SMIR generator (MEX files)
│   ├── audioprocessing/                # SH tools and STFT class
│   ├── npy-matlab/                     # NumPy file I/O for MATLAB
│   └── tprod/                          # Tensor product utilities
```

## Usage

### 1. Configure Parameters

Edit [config.yaml](config.yaml) to set:
- Room dimensions and reverberation times
- Sampling frequency and processing parameters
- Eigenmike configuration (rigid/open sphere)
- Source positions and types

### 2. Generate RIRs

Generate simulated RIRs for multiple rooms:

```matlab
% Generate 10 rooms starting from ID 0, using 4 parallel workers
simulated_rir_generator(0, 10, 4);
```

This creates:
- `rirs/rir/room{id}_{position}.mat` - Reverberant RIRs (32 channels)
- `rirs/rir_clean/room{id}_{position}_clean.mat` - Anechoic RIRs
- `rirs/info/rir_info_*.csv` - Source positions and angles
- `rirs/info/room_info_*.csv` - Room dimensions and RT60

### 3. Encode to HOA Format

Convert Eigenmike RIRs to Higher Order Ambisonics:

```matlab
% Encode all RIRs in ./rirs/rir to HOA format
simulated_hoa_rir_generator("./rirs/rir");
```

Output files: `rirs/rir/hoa_room{id}_{position}.mat`

### 4. Demo: Ambisonics to Binaural

Run the demo to spatialize audio using HOA RIRs:

**MATLAB:**
```matlab
demo_A2B2A
```

**Python (from audio file):**
```python
jupyter notebook demo_A2B2A.ipynb
```

**Python (from RIR .mat file):**
```python
jupyter notebook demo_A2B2A_from_rir_mat.ipynb
```

This notebook demonstrates:
- Loading RIR files from MATLAB format with wildcard pattern matching
- Spherical harmonic decomposition and encoding
- Beamforming visualization
- Converting HOA back to A-format and computing SNR

## Configuration Parameters

Key parameters in [config.yaml](config.yaml):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `general.c` | Speed of sound (m/s) | 343 |
| `general.fs` | Sampling frequency (Hz) | 44100 |
| `rir_generator.N_harm` | Max spherical harmonics order | 20 |
| `rir_generator.reflectionOrder` | Max reflection order | 10 |
| `rir_generator.t60` | Reverberation times (s) | [0.2, 0.3, 0.4, 0.5] |
| `rir_generator.room_width` | Room width range (m) | [4.0, 8.0] |
| `rir_generator.room_length` | Room length range (m) | [5.0, 10.0] |
| `rir_generator.room_height` | Room height range (m) | [3.0, 4.0] |
| `rir_generator.npoints` | Sphere sampling points | 46 |
| `eigenmike.mic_r` | Eigenmike radius (m) | 0.042 |
| `hoa_encoding.sh_order` | HOA encoding order | 6 |

## Python Utilities

### Signal Processing ([utils.py](utils.py))

```python
from utils import mystft, myistft, multichannel_fft_convolve, trim_signal_to_activity

# STFT/iSTFT
X = mystft(signal, fs=44100)
signal_rec = myistft(X, fs=44100)

# Multichannel convolution
output = multichannel_fft_convolve(rir, signal)

# Find active signal segments
valid_indices = trim_signal_to_activity(signal, fs, duration_sec=5, thresh=0.001)
```

### Spherical Harmonics ([shtools.py](shtools.py))

```python
from shtools import get_real_sh, sph_bn, encode_hoa

# Compute real spherical harmonics
Y = get_real_sh(order, elevation, azimuth)

# Radial functions for rigid sphere
bn = sph_bn(n_array, kr_array, isRigid=True)
```

### HOA Tools ([homtools.py](homtools.py))

```python
from homtools import get_eigenmike, reduced_eigenmike

# Load Eigenmike configuration
hom = get_eigenmike(r, az, el, gain, w, isRigid=True)

# Create reduced microphone array
hom_reduced = reduced_eigenmike(Q=16, hom=hom)
```

## Output Files

### RIR Files (`.mat`)
- **Variables**: `sim_rir` (reverberant) or `sim_rir_clean` (anechoic)
- **Dimensions**: (32 microphones × time samples)
- **Format**: Single precision floating point

### HOA Files (`.mat`)
- **Variable**: `rir_hoa`
- **Dimensions**: (49 SH coefficients × time samples) for 6th order
- **Format**: Radially-compensated HOA impulse responses

### Metadata Files (`.csv`)

**room_info.csv**:
- `room_id`, `x`, `y`, `z` (dimensions in meters), `t60` (seconds)

**rir_info.csv**:
- `room_id`, `rir_id`, `x`, `y`, `z` (source position), `az`, `el` (angles in radians)

## Examples

### Generate Custom Dataset

```matlab
% Configure in config.yaml, then:
simulated_rir_generator(0, 50, 8);  % 50 rooms, 8 workers
simulated_hoa_rir_generator("./rirs/rir");
```

### Load and Process RIR in Python

```python
import glob
import scipy.io as sio

# Find and load RIR with wildcard pattern
rir_files = glob.glob('rirs/rir/room0_0*.mat')
if rir_files:
    rir_data = sio.loadmat(rir_files[0])
    # Find the matching key dynamically
    rir_keys = [k for k in rir_data.keys() if not k.startswith('__') and 'sim_rir' in k]
    rir = rir_data[rir_keys[0]]  # Shape: (32, nsamples)

# Load HOA-encoded RIR
hoa_files = glob.glob('rirs/rir/hoa_room0_0*.mat')
if hoa_files:
    hoa_data = sio.loadmat(hoa_files[0])
    rir_hoa = hoa_data['rir_hoa']  # Shape: (49, nsamples)
```

## Technical Details

### HOA Encoding Process

1. **Spherical Harmonics Transform**: Project 32-channel Eigenmike signals onto spherical harmonics basis (N=6, 49 coefficients)
2. **Radial Compensation**: Apply frequency-dependent radial filters to compensate for rigid sphere scattering
3. **FIR Filtering**: Compute 2048-tap FIR filters for each SH order using spherical Bessel functions

### Spatial Sampling

- Source positions sampled on a sphere using stratified sampling
- Elevation limited to [60°, 130°] to ensure sources remain in room
- Distance from array center: 0.5-1.5 m (configurable)

## References

- **SMIR-Generator**: [SMIR-Generator repository](src/SMIR-Generator/)
- **Eigenmike**: em32 Eigenmike microphone array (mh Acoustics)
- **Ambisonics**: Higher Order Ambisonics encoding following ACN/SN3D conventions
## Version Control

A `.gitignore` file is included to exclude:
- Python cache files and compiled modules (`__pycache__/`)
- Log files (`logs/`)
- Virtual environments
- IDE configuration files
- System files (macOS, MATLAB temporary files)

## Recent Updates

- **Generic RIR Loading**: Added support for wildcard file pattern matching using `glob` for more flexible RIR file handling
- **Dynamic Variable Keys**: Automatic detection of variable names in `.mat` files (`sim_rir`, `sim_rir_clean`, etc.)
- **Enhanced Demo Notebook**: `demo_A2B2A_from_rir_mat.ipynb` now includes beamforming visualization and SNR computation
- **Configuration Flexibility**: Support for `srir_type` parameter in config to select between clean and reverberant RIRs
## Author

**Julio J. Carabias-Orti**  
Date: December 2025

## License

See individual component licenses in `src/` subdirectories.
