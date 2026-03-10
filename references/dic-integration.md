# DIC Data Integration for PINN Inverse Problems

## Table of Contents
1. [Supported Software and File Formats](#1-supported-software-and-file-formats)
2. [Universal Loader with Format Auto-Detection](#2-universal-loader-with-format-auto-detection)
3. [Rigid-Body Subtraction](#3-rigid-body-subtraction)
4. [Coordinate Alignment (ICP)](#4-coordinate-alignment-icp)
5. [Mapping DIC Points to Chart Coordinates](#5-mapping-dic-points-to-chart-coordinates)
6. [Data Loss with Noise Handling](#6-data-loss-with-noise-handling)
7. [Noise and Uncertainty Reference](#7-noise-and-uncertainty-reference)
8. [Practical Pipeline Checklist](#8-practical-pipeline-checklist)

---

## 1. Supported Software and File Formats

### 1.1 Correlated Solutions — Vic-3D and Vic-2D

**Most common in soft-matter and biomechanics labs.**

Vic-3D output: `.csv` with header row. Typical columns:
```
x [mm], y [mm], z [mm], u [mm], v [mm], w [mm], exx [-], eyy [-], exy [-], sigma [-]
```
`sigma` = correlation coefficient (0 = perfect, 1 = worst; invert for confidence).

Vic-2D (no Z): pixel units — must supply `px_to_mm` calibration factor.

```python
def load_vic3d(path: str, units: str = "mm") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Vic-3D output CSV. Returns x_obs (N,3), u_obs (N,3), conf (N,)."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip().str.lower()
    col_map = {
        'x [mm]': 'X', 'y [mm]': 'Y', 'z [mm]': 'Z',
        'u [mm]': 'ux', 'v [mm]': 'uy', 'w [mm]': 'uz',
        'x': 'X', 'y': 'Y', 'z': 'Z', 'u': 'ux', 'v': 'uy', 'w': 'uz',
        'sigma': 'sigma',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df = df.dropna(subset=['X', 'Y', 'Z', 'ux', 'uy', 'uz'])
    conf = 1.0 - df['sigma'].values if 'sigma' in df.columns else np.ones(len(df))
    x_obs = df[['X', 'Y', 'Z']].values.astype(np.float64)
    u_obs = df[['ux', 'uy', 'uz']].values.astype(np.float64)
    if units == "m":
        x_obs *= 1e-3; u_obs *= 1e-3
    return x_obs, u_obs, np.clip(conf, 0, 1)


def load_vic2d(path: str, px_to_mm: float, units: str = "mm") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Vic-2D output. px_to_mm = physical gauge / pixel count."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=['x', 'y', 'u', 'v'])
    x_obs = df[['x', 'y']].values * px_to_mm
    u_obs = df[['u', 'v']].values * px_to_mm
    conf = 1.0 - df['sigma'].values if 'sigma' in df.columns else np.ones(len(df))
    if units == "m":
        x_obs *= 1e-3; u_obs *= 1e-3
    return x_obs, u_obs, np.clip(conf, 0, 1)
```

---

### 1.2 GOM ARAMIS (Zeiss)

**Common in industrial and structural labs. CSV (semicolons) or HDF5.**

ARAMIS CSV typical header:
```
Point; X [mm]; Y [mm]; Z [mm]; dX [mm]; dY [mm]; dZ [mm]; Eps_x [-]; Eps_y [-]; Validity
```
Validity column: 1 = valid, 0 = invalid.

```python
def load_aramis(path: str, units: str = "mm") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load GOM ARAMIS CSV. Handles semicolon or tab delimiters."""
    with open(path) as f:
        sep = ';' if ';' in f.readline() else '\t'
    df = pd.read_csv(path, sep=sep, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    col_map = {'X [mm]': 'X', 'Y [mm]': 'Y', 'Z [mm]': 'Z',
               'dX [mm]': 'ux', 'dY [mm]': 'uy', 'dZ [mm]': 'uz', 'Validity': 'valid'}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if 'valid' in df.columns:
        df = df[df['valid'] == 1]
    df = df.dropna(subset=['X', 'Y', 'Z', 'ux', 'uy', 'uz'])
    x_obs = df[['X', 'Y', 'Z']].values.astype(np.float64)
    u_obs = df[['ux', 'uy', 'uz']].values.astype(np.float64)
    if units == "m":
        x_obs *= 1e-3; u_obs *= 1e-3
    return x_obs, u_obs, np.ones(len(df))


def load_aramis_h5(path: str, step: int = -1, units: str = "mm") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load GOM ARAMIS HDF5. step=-1 = last load step."""
    import h5py
    with h5py.File(path, 'r') as f:
        steps = sorted(f['stages'].keys(), key=lambda s: int(s.split('_')[-1]))
        s = steps[step]
        X  = f[f'stages/{s}/coordinates/x'][:]
        Y  = f[f'stages/{s}/coordinates/y'][:]
        Z  = f[f'stages/{s}/coordinates/z'][:]
        ux = f[f'stages/{s}/displacements/x'][:]
        uy = f[f'stages/{s}/displacements/y'][:]
        uz = f[f'stages/{s}/displacements/z'][:]
        valid = f[f'stages/{s}/point_validity'][:].astype(bool)
    x_obs = np.stack([X, Y, Z], axis=-1)[valid]
    u_obs = np.stack([ux, uy, uz], axis=-1)[valid]
    if units == "m":
        x_obs *= 1e-3; u_obs *= 1e-3
    return x_obs, u_obs, np.ones(len(x_obs))
```

---

### 1.3 Dantec ISTRA 4D

**Used in dynamic and high-speed testing labs. CSV per stage.**

ISTRA 4D CSV format:
```
Stage; Point ID; X0[mm]; Y0[mm]; Z0[mm]; U[mm]; V[mm]; W[mm]; exx; eyy; exy; quality
```
`quality`: 0–1 scale, higher = better (opposite of Vic-3D sigma).

```python
def load_istra4d(path: str, quality_threshold: float = 0.5, units: str = "mm"):
    df = pd.read_csv(path, sep=';', skipinitialspace=True)
    df.columns = df.columns.str.strip()
    if 'Stage' in df.columns:
        df = df[df['Stage'] == df['Stage'].max()]  # last stage only
    col_map = {'X0[mm]': 'X', 'Y0[mm]': 'Y', 'Z0[mm]': 'Z',
               'U[mm]': 'ux', 'V[mm]': 'uy', 'W[mm]': 'uz', 'quality': 'conf'}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if 'conf' in df.columns:
        df = df[df['conf'] >= quality_threshold]
    df = df.dropna(subset=['X', 'Y', 'Z', 'ux', 'uy', 'uz'])
    x_obs = df[['X', 'Y', 'Z']].values.astype(np.float64)
    u_obs = df[['ux', 'uy', 'uz']].values.astype(np.float64)
    conf  = df['conf'].values if 'conf' in df.columns else np.ones(len(df))
    if units == "m":
        x_obs *= 1e-3; u_obs *= 1e-3
    return x_obs, u_obs, conf
```

---

### 1.4 DICe (Sandia National Labs — Open Source)

**Python/C++ open-source DIC. Output: `.txt` space-delimited.**

```
# DICe output
# COORDINATE_X COORDINATE_Y DISPLACEMENT_X DISPLACEMENT_Y SIGMA MATCH_PERCENT ITERATIONS
123.45 234.56 0.012 -0.003 0.001 98.5 15
```

```python
def load_dice(path: str, px_to_mm: float = 1.0, units: str = "mm"):
    df = pd.read_csv(path, comment='#', sep=r'\s+', header=None)
    df.columns = ['X', 'Y', 'ux', 'uy', 'sigma', 'match', 'iter'][:len(df.columns)]
    df = df.dropna()
    x_obs = df[['X', 'Y']].values * px_to_mm
    u_obs = df[['ux', 'uy']].values * px_to_mm
    conf  = df['match'].values / 100.0 if 'match' in df.columns else np.ones(len(df))
    if units == "m":
        x_obs *= 1e-3; u_obs *= 1e-3
    return x_obs, u_obs, conf
```

---

### 1.5 ncorr (MATLAB — Open Source)

**MATLAB 2D DIC. Output: `.mat` workspace. Displacements stored as 2D pixel grids.**

```python
def load_ncorr(path: str, px_to_mm: float = 1.0):
    import scipy.io
    mat = scipy.io.loadmat(path)
    try:
        disp = mat['data_dic_save']['displacements'][0, 0]
        u_grid = disp['u'][0, 0]
        v_grid = disp['v'][0, 0]
        corrcoef = disp['corrcoef'][0, 0]
    except KeyError:
        u_grid = mat['u']; v_grid = mat['v']
        corrcoef = mat.get('corrcoef', np.zeros_like(u_grid))

    ny, nx = u_grid.shape
    Y_grid, X_grid = np.mgrid[0:ny, 0:nx]
    valid = ~(np.isnan(u_grid) | np.isnan(v_grid))
    x_obs = np.stack([X_grid[valid], Y_grid[valid]], axis=-1).astype(np.float64) * px_to_mm
    u_obs = np.stack([u_grid[valid], v_grid[valid]], axis=-1) * px_to_mm
    conf  = 1.0 - np.clip(corrcoef[valid], 0, 1)
    return x_obs, u_obs, conf
```

---

### 1.6 MatchID

**Commercial software common in Europe. CSV or Excel export.**

```python
def load_matchid(path: str, units: str = "mm"):
    df = pd.read_excel(path) if path.endswith(('.xlsx', '.xls')) else pd.read_csv(path)
    df.columns = df.columns.str.strip()
    col_search = {
        'X': ['X0', 'X_0', 'Xref', 'x0', 'X'],
        'Y': ['Y0', 'Y_0', 'Yref', 'y0', 'Y'],
        'Z': ['Z0', 'Z_0', 'Zref', 'z0', 'Z'],
        'ux': ['U', 'Ux', 'dX', 'u'], 'uy': ['V', 'Uy', 'dY', 'v'],
        'uz': ['W', 'Uz', 'dZ', 'w'], 'conf': ['Residual', 'residual', 'Confidence'],
    }
    rename = {}
    for target, candidates in col_search.items():
        for c in candidates:
            if c in df.columns:
                rename[c] = target; break
    df = df.rename(columns=rename).dropna(subset=['X', 'Y', 'Z', 'ux', 'uy', 'uz'])
    x_obs = df[['X', 'Y', 'Z']].values.astype(np.float64)
    u_obs = df[['ux', 'uy', 'uz']].values.astype(np.float64)
    conf  = 1.0 / (1.0 + df['conf'].values) if 'conf' in df.columns else np.ones(len(df))
    if units == "m":
        x_obs *= 1e-3; u_obs *= 1e-3
    return x_obs, u_obs, conf
```

---

### 1.7 Generic XYZ+UVW CSV

```python
def load_generic_csv(path: str, col_x='X', col_y='Y', col_z='Z',
                     col_ux='U', col_uy='V', col_uz='W',
                     col_conf=None, units="mm", delimiter=','):
    df = pd.read_csv(path, sep=delimiter, skipinitialspace=True)
    df = df.dropna(subset=[col_x, col_y, col_z, col_ux, col_uy, col_uz])
    x_obs = df[[col_x, col_y, col_z]].values.astype(np.float64)
    u_obs = df[[col_ux, col_uy, col_uz]].values.astype(np.float64)
    conf  = df[col_conf].values.astype(np.float64) if col_conf else np.ones(len(df))
    if units == "m":
        x_obs *= 1e-3; u_obs *= 1e-3
    return x_obs, u_obs, conf
```

---

## 2. Universal Loader with Format Auto-Detection

```python
import os, numpy as np, pandas as pd
from typing import Optional

def load_dic(path: str, software: Optional[str] = None,
             units: str = "mm", **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Universal DIC loader.

    Args:
        path:     Path to file (.csv, .txt, .mat, .xlsx, .h5)
        software: 'vic3d' | 'vic2d' | 'aramis' | 'aramis_h5' | 'istra4d' |
                  'dice' | 'ncorr' | 'matchid' | 'generic'
                  Auto-detected from file content if None.
        units:    'mm' (no conversion) or 'm' (multiplies by 1e-3)
        **kwargs: Forwarded to format-specific loader (e.g. px_to_mm, step, quality_threshold)

    Returns:
        x_obs (N,3), u_obs (N,3), conf (N,)  — all in requested units
    """
    ext = os.path.splitext(path)[1].lower()
    if software is None:
        software = _detect_dic_software(path, ext)
    loaders = {
        'vic3d':     lambda: load_vic3d(path, units=units),
        'vic2d':     lambda: load_vic2d(path, units=units, **kwargs),
        'aramis':    lambda: load_aramis(path, units=units),
        'aramis_h5': lambda: load_aramis_h5(path, units=units, **kwargs),
        'istra4d':   lambda: load_istra4d(path, units=units, **kwargs),
        'dice':      lambda: load_dice(path, units=units, **kwargs),
        'ncorr':     lambda: load_ncorr(path, **kwargs),
        'matchid':   lambda: load_matchid(path, units=units),
        'generic':   lambda: load_generic_csv(path, units=units, **kwargs),
    }
    if software not in loaders:
        raise ValueError(f"Unknown software '{software}'. Options: {list(loaders)}")
    return loaders[software]()


def _detect_dic_software(path: str, ext: str) -> str:
    if ext == '.mat': return 'ncorr'
    if ext in ('.h5', '.hdf5'): return 'aramis_h5'
    if ext in ('.xlsx', '.xls'): return 'matchid'
    if ext == '.txt': return 'dice'
    with open(path) as f:
        header = f.readline().lower()
    if 'sigma' in header and 'x [mm]' in header: return 'vic3d'
    if 'x0[mm]' in header or 'quality' in header: return 'istra4d'
    if 'residual' in header or 'x0' in header.replace(' ', ''): return 'matchid'
    if ';' in header and ('dx [mm]' in header or 'validity' in header): return 'aramis'
    return 'generic'
```

---

## 3. Rigid-Body Subtraction

Always do this before using DIC as PINN training targets.

```python
def subtract_rigid_body(
    x_obs: np.ndarray,              # (N, 3) reference positions
    u_obs: np.ndarray,              # (N, 3) measured total displacements
    conf: Optional[np.ndarray] = None,  # (N,) confidence weights
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted least-squares 6-DOF rigid body subtraction.
    Model: u_rb(X) = T + W×X, W skew-symmetric (linearized rotation).
    Returns (u_corrected, params=[Tx,Ty,Tz,wx,wy,wz]).
    """
    N = len(x_obs)
    A = np.zeros((3*N, 6), dtype=np.float64)
    for i, Xi in enumerate(x_obs):
        r = 3*i
        A[r:r+3, :3] = np.eye(3)
        A[r,   3] =  0;     A[r,   4] =  Xi[2]; A[r,   5] = -Xi[1]
        A[r+1, 3] = -Xi[2]; A[r+1, 4] =  0;     A[r+1, 5] =  Xi[0]
        A[r+2, 3] =  Xi[1]; A[r+2, 4] = -Xi[0]; A[r+2, 5] =  0
    b = u_obs.reshape(-1)
    if conf is not None:
        w = np.repeat(conf, 3)
        params, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
    else:
        params, *_ = np.linalg.lstsq(A, b, rcond=None)
    T, omega = params[:3], params[3:]
    W_skew = np.array([[0, -omega[2], omega[1]],
                       [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]])
    u_rb = (x_obs @ W_skew.T) + T
    return u_obs - u_rb, params
```

---

## 4. Coordinate Alignment (ICP)

```python
def align_dic_to_pinn(x_dic, u_dic, x_pinn_surface, max_corr_dist=2.0):
    """ICP: DIC frame → PINN frame. Rotates displacement vectors (R only, no t)."""
    import open3d as o3d
    src = o3d.geometry.PointCloud(); src.points = o3d.utility.Vector3dVector(x_dic)
    tgt = o3d.geometry.PointCloud(); tgt.points = o3d.utility.Vector3dVector(x_pinn_surface)
    T_init = np.eye(4); T_init[:3, 3] = x_pinn_surface.mean(0) - x_dic.mean(0)
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, max_correspondence_distance=max_corr_dist, init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
    R, t = result.transformation[:3, :3], result.transformation[:3, 3]
    return (R @ x_dic.T).T + t, (R @ u_dic.T).T, result.transformation
```

---

## 5. Mapping DIC Points to Chart Coordinates

```python
from scipy.spatial import KDTree
import torch

def map_to_charts(x_obs, u_obs, conf, decoders, masks, xi_grids,
                  mask_threshold=0.5, dist_tol_mm=1.5):
    chart_assignments = []
    for i, (decoder, mask, xi_grid) in enumerate(zip(decoders, masks, xi_grids)):
        with torch.no_grad():
            x_chart = decoder(torch.tensor(xi_grid, dtype=torch.float32)).numpy()
        dists, nn_idx = KDTree(x_chart).query(x_obs)
        xi_obs_i = xi_grid[nn_idx]
        with torch.no_grad():
            mask_v = mask(torch.tensor(xi_obs_i, dtype=torch.float32)).squeeze().numpy()
        valid = (mask_v > mask_threshold) & (dists < dist_tol_mm)
        chart_assignments.append({
            'chart_id': i, 'xi_obs': xi_obs_i[valid],
            'u_obs': u_obs[valid], 'conf': conf[valid], 'n_valid': int(valid.sum()),
        })
    return chart_assignments
```

---

## 6. Data Loss with Noise Handling

```python
import torch.nn.functional as F

def huber_data_loss(pinn_nets, chart_assignments, noise_floor_mm=0.02):
    """
    Confidence-weighted Huber loss. delta = 10 × noise_floor.
    Stop training when L_data ≈ noise_floor_mm² (e.g. 4e-4 mm² for 0.02 mm noise).
    Going below this overfits measurement noise.
    """
    losses = []
    for ca in chart_assignments:
        if ca['n_valid'] == 0: continue
        xi    = torch.tensor(ca['xi_obs'], dtype=torch.float32)
        u_tgt = torch.tensor(ca['u_obs'],  dtype=torch.float32)
        w_c   = torch.tensor(ca['conf'],   dtype=torch.float32)
        u_pred = pinn_nets[ca['chart_id']](xi)
        huber = F.huber_loss(u_pred, u_tgt, delta=10*noise_floor_mm, reduction='none')
        losses.append((w_c.unsqueeze(-1) * huber).mean())
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, requires_grad=True)
```

---

## 7. Noise and Uncertainty Reference

| Software | Typical noise floor | Notes |
|----------|-------------------|-------|
| Vic-3D stereo | 0.01–0.05 mm | Depends on subset size, speckle, standoff distance |
| GOM ARAMIS | 0.005–0.03 mm | Newer blue-light systems achieve lower end |
| Dantec ISTRA 4D | 0.01–0.05 mm | High-speed variant is noisier |
| DICe | 0.02–0.1 px × scale | User-dependent; open-source |
| ncorr (MATLAB) | 0.01–0.08 px × scale | Similar to DICe |
| MatchID | 0.005–0.02 mm | Commercial; generally high quality |

**Noise floor formula**: `noise_floor_mm = σ_u_px × (gauge_mm / image_width_px)`.
Example: 100 mm gauge on 2000 px image, σ_u = 0.05 px → noise = 0.0025 mm.

**Identifiability impact**: if `noise_floor / signal_amplitude > 0.1`, that parameter has poor observability from that data channel. Add load cases.

---

## 8. Practical Pipeline Checklist

```
□ Load DIC data:  x_obs, u_obs, conf = load_dic(path, software='vic3d', units='mm')
□ Check units:    ensure mm everywhere (or m everywhere) — do not mix
□ Filter:         conf > 0.3 minimum; drop NaN rows
□ Rigid body:     u_obs, params = subtract_rigid_body(x_obs, u_obs, conf)
□ Align:          x_obs, u_obs, T = align_dic_to_pinn(x_obs, u_obs, x_pinn_surface)
□ Map to charts:  chart_assignments = map_to_charts(x_obs, u_obs, conf, ...)
□ Set noise_floor_mm from DIC software spec (see table above)
□ Check κ(S):     sensitivity matrix before training; if > 1e4, add load cases
□ Train:          L_data = huber_data_loss(pinn_nets, chart_assignments, noise_floor_mm)
□ Stop when:      L_data ≈ noise_floor_mm² (do not chase below this)
□ Report:         L_data at convergence alongside parameter estimates ± uncertainty
```
