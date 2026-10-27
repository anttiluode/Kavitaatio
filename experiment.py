"""Kavitaatio: persistent Robin-wall bits, ellipse scattering and ray-lens checks.

NumPy-only, deterministic research spike; NOT a solver for OpenAI local smoothing.
Reflected Neumann walls are kappa=0; kappa>0 adds a Robin spring on wall faces.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

N = 96
C = 0.65
DT = 0.5
A, B = 36.0, 27.0
K = 8
STEPS = 350
START, STOP, STRIDE = 145, 345, 2
CENTER = (N - 1) / 2
NOISE = 0.04  # test noise relative to typical one-bit RMS


def geometry():
    yy, xx = np.mgrid[:N, :N]
    X, Y = xx - CENTER, yy - CENTER
    inside = (X / A) ** 2 + (Y / B) ** 2 <= 1.0
    f = np.sqrt(A * A - B * B)
    srcs = [(round(CENTER), round(CENTER - f)),
            (round(CENTER + 8), round(CENTER - f)),
            (round(CENTER - 8), round(CENTER - f))]
    receiver = (round(CENTER), round(CENTER + f))
    angle = np.arctan2(Y / B, X / A)
    sector = (np.floor(((angle + np.pi) / (2 * np.pi)) * K).astype(int) % K)
    return inside, sector, srcs, receiver


def shift_mask(arr, axis, sign):
    out = np.roll(arr, sign, axis=axis)
    if axis == 0: out[0 if sign == 1 else -1, :] = False
    else: out[:, 0 if sign == 1 else -1] = False
    return out


def wall_counts(inside, sector):
    """Count missing grid-neighbor faces assigned to the *interior* wall cell.

    Sector address uses its angle; repeated adjacent faces add matching springs.
    """
    counts = np.zeros((K, N, N), float)
    for ax in (0, 1):
        for sign in (-1, 1):
            missing = inside & ~shift_mask(inside, ax, sign)
            for k in range(K): counts[k] += missing & (sector == k)
    return counts


def simulate(bits, source_index, setup=None):
    inside, sector, srcs, recv = geometry() if setup is None else setup
    y, x = np.mgrid[:N, :N]
    counts = wall_counts(inside, sector)
    # Robin compliance is a fixed material parameter: it persists after waves die.
    robin_faces = np.einsum('k,kij->ij', np.asarray(bits, float) * 0.28, counts)
    u = np.zeros((N, N)); prev = np.zeros_like(u)
    sy, sx = srcs[source_index]
    pulse = np.exp(-((x-sx)**2 + (y-sy)**2) / (2 * 1.55**2)) * inside
    u += pulse
    traces = []
    snaps = []
    # A centered, explicitly Neumann/Robin graph-Laplacian: missing neighbors
    # contribute zero flux for kappa=0, or -kappa*u for the Robin wall.
    neigh = np.zeros((N,N))
    for axis in (0,1):
        for direction in (-1,1):
            neigh += inside & shift_mask(inside, axis, direction)
    for t in range(STEPS):
        lap = -neigh*u - robin_faces*u
        for ax in (0, 1):
            for sign in (-1, 1):
                shifted_u = np.roll(u, sign, axis=ax)
                if ax == 0: shifted_u[0 if sign == 1 else -1, :] = 0
                else: shifted_u[:, 0 if sign == 1 else -1] = 0
                lap += shifted_u * inside * shift_mask(inside, ax, sign)
        # Damped central difference. CFL: (C*DT)^2 = 0.105625 < 0.5.
        damping = 0.006
        nxt = (2*u - (1-damping*DT/2)*prev + (C*DT)**2*lap) / (1+damping*DT/2)
        nxt *= inside
        prev, u = u, nxt
        if START <= t < STOP and (t - START) % STRIDE == 0:
            ry, rx = recv
            traces.append([u[ry,rx],u[ry-1,rx],u[ry+1,rx],u[ry,rx-1],u[ry,rx+1]])
        if t in (235,280): snaps.append(u.copy())
    return np.asarray(traces), snaps


def orth_rank(columns, threshold=0.01):
    s = np.linalg.svd(columns, compute_uv=False)
    return int(np.sum(s > (s[0]*threshold if len(s) else 0))), [round(float(v),6) for v in s]


def check_ray_filter():
    """Directly implement multiplier from OpenAI wave-preprint §1.

    For a free positive half-wave at 2D frequency xi: imaginary shift by
    -i*rho*(1,omega) exactly produces exp[-rho*(|xi|-omega.xi)].
    This is a diagnostic on free fields; never called a cavity-wave theorem.
    """
    n=128
    freq = np.fft.fftfreq(n) * 2*np.pi
    kx, ky = np.meshgrid(freq,freq)
    r = np.hypot(kx,ky)
    om = (1.,0.)
    rho=10.
    H=np.exp(-rho*(r - kx))
    Hgeom=np.exp(-rho*r*(1 - np.divide(kx,r,out=np.zeros_like(r),where=r>0)))
    assert np.max(np.abs(H-Hgeom)) < 1e-12
    # Directional response at fixed |xi|, high frequency narrower in angle.
    # Numerical half-power width from the exact analytic frequency multiplier.
    widths = []
    for Nwave in (0.7, 1.4):
        angles = np.linspace(0,np.pi/2,10000)
        h=np.exp(-rho*Nwave*(1-np.cos(angles)))
        width=float(np.interp(0.5,h[::-1],angles[::-1]))
        widths.append(width)
    return {"multiplier_identity_max_abs_error":float(np.max(np.abs(H-Hgeom))),
            "angular_half_amplitude_widths_rad":widths,
            "high_to_low_angular_width_ratio":widths[1]/widths[0],
            "asymptotic_expected_ratio":float(1/np.sqrt(2))}


def spatial_ray_features(fields, recv, mode="parabolic"):
    """Virtual global-field analysis instrument, NOT a single local listener.

    The paper's exact free-half-wave complex shift multiplier is applied
    to each recorded real cavity snapshot only as a candidate representation.
    We first apply a fixed annulus to compare matched spectral content.
    """
    ky,kx=np.meshgrid(2*np.pi*np.fft.fftfreq(N),2*np.pi*np.fft.fftfreq(N),indexing="ij")
    r=np.hypot(kx,ky)
    radial=np.exp(-.5*((r-.7)/.28)**2)
    row=[]
    for field in fields:
        fft=np.fft.fft2(field)
        for j in range(K):
            angle=2*np.pi*j/K
            kd=kx*np.cos(angle)+ky*np.sin(angle)
            if mode == "parabolic":
                h=np.exp(-10*(r-kd))
            elif mode == "fixed_angle":
                dot=np.divide(kd,r,out=np.zeros_like(r),where=r>1e-12)
                delta=np.arccos(np.clip(dot,-1,1))
                h=np.exp(-.5*(delta/.50)**2)
            elif mode == "radial":
                h=np.exp(-.5*((r-(.25+j*.14))/.13)**2)
            else:
                raise ValueError(mode)
            signal=np.fft.ifft2(fft*radial*h)[recv]
            row.extend((signal.real,signal.imag))
    return np.asarray(row)


def analytic_signal(signal):
    """Analytic temporal signal; quadrature driving assumed in phase-cycling test."""
    n=len(signal)
    fft=np.fft.fft(signal,axis=0)
    keep=np.zeros(n)
    keep[0]=1
    if n%2==0:
        keep[n//2]=1
        keep[1:n//2]=2
    else:
        keep[1:(n+1)//2]=2
    return np.fft.ifft(fft*keep[:,None],axis=0)


def one_patch_recognition(mat, seed=442, repeats=300, sigma=None):
    """Known one-patch codebook, fresh Gaussian sensor noise, no learned NN."""
    rng=np.random.default_rng(seed)
    if sigma is None:
        sigma=NOISE*np.median(np.linalg.norm(mat,axis=0))/np.sqrt(len(mat))
    correct=[]
    for i in range(K):
        obs=mat[:,i,None]+rng.normal(0,sigma,(len(mat),repeats))
        predicted=np.argmin(np.sum((mat[:,:,None]-obs[:,None,:])**2,axis=0),axis=0)
        correct.append(float(np.mean(predicted==i)))
    return float(np.mean(correct))


def bench():
    setup=geometry()
    pattern=np.eye(K)
    samples=[]
    for source in range(3):
        base,base_snap=simulate(np.zeros(K),source,setup)
        d=[]; dsnap=[]
        for k in range(K):
            tr,scenes=simulate(pattern[k],source,setup)
            d.append(tr-base)
            dsnap.append([scenes[t]-base_snap[t] for t in range(len(scenes))])
        samples.append((base,np.asarray(d),np.asarray(dsnap)))
    # Represent each wall bit by a column of a *measured* response matrix.
    # Signed time samples = phase-preserving detector; early-square = intensity.
    report={"openai_complex_ray":check_ray_filter(),"protocol":{"N":N,"K":K,"dt":DT,"wave_speed":C,
            "source_locations":setup[2],"listener_location":setup[3],"frames":len(samples[0][0]),
            "wall_perturbation":"kappa = 0.28 per active boundary face", "noise_fraction":NOISE,
            "noise_definition":"absolute Gaussian sigma = 0.04 * median single-source single-patch response RMS"}}
    v1=samples[0][1][:,:,0].T
    v3=np.concatenate([s[1][:,:,0].T for s in samples],axis=0)
    patch1=samples[0][1].transpose(1,2,0).reshape(-1,K)
    energy1=((samples[0][0][None,:,:] + samples[0][1])**2 - samples[0][0][None,:,:]**2)[:,:,0].T
    # Coherent TWO-source phase cycling is fundamentally different from changing
    # the global phase of one source. Four source relative phases measure the
    # complex cross term after intensity detection (standard interferometry).
    z0=analytic_signal(samples[0][0][:,0,None] + samples[0][1][:,:,0].T)
    z1=analytic_signal(samples[1][0][:,0,None] + samples[1][1][:,:,0].T)
    r0=analytic_signal(samples[0][0][:,0,None])
    r1=analytic_signal(samples[1][0][:,0,None])
    mix=[]
    for phase in (0,np.pi/2,np.pi,3*np.pi/2):
        offset=np.exp(1j*phase)
        val=np.abs(z0+offset*z1)**2 - np.abs(r0+offset*r1)**2
        mix.append(val)
    phasecycle=np.concatenate(mix,axis=0)
    cases={"single_source_signed_trace":v1,"single_source_intensity_trace":energy1,
           "single_source_five_pixel_patch":patch1,"three_source_signed_traces":v3,
           "two_source_four_phase_intensity":phasecycle}
    for name,mat in cases.items():
        rank,spec=orth_rank(mat)
        centered=mat / np.maximum(np.linalg.norm(mat,axis=0,keepdims=True),1e-12)
        gram=centered.T @ centered
        overlap=np.abs(gram-np.eye(K))
        np.fill_diagonal(overlap,0.)
        report[name]={"rank_at_1pct_largest_singular":rank,"singular_values":spec,
                       "largest_column_correlation":float(np.max(overlap)),
                       "smallest_column_norm":float(np.min(np.linalg.norm(mat,axis=0)))}
    # Fixed noise level across protocols = 4% of a *single-source* one-bit
    # RMS scale. Three-source and four-step results use more measurements.
    sigma=NOISE*np.median(np.linalg.norm(v1,axis=0))/np.sqrt(len(v1))
    for name,mat in cases.items():
        report[name]["one_wall_patch_8_way_accuracy_at_fixed_noise"]=one_patch_recognition(mat,sigma=sigma)
        report[name]["query_count"]=3 if name=="three_source_signed_traces" else (4 if name=="two_source_four_phase_intensity" else 1)
    # Important: the spatial FFT requires the ENTIRE cavity field. Unlike the
    # five-pixel patch, this is a virtual full-field instrument, not a needle.
    for mode in ("parabolic", "fixed_angle", "radial"):
        d_snap=samples[0][2]
        features=np.array([spatial_ray_features(d_snap[k],setup[3],mode) for k in range(K)]).T
        rank,spec=orth_rank(features)
        report["virtual_full_field_"+mode]={"rank_at_1pct_largest_singular":rank,
                        "singular_values":spec,"feature_count":int(features.shape[0]),
                        "requires_full_field_snapshot":True}
    # Only one-bit calibration thus far: no claim that a multi-bit inverse works.
    # Explicit phase cycling is a control: changing the global phase of ONE
    # input scales the coherent receiver trace and cannot increase rank.
    report["global_phase_only_rank"] = orth_rank(np.concatenate([v1,-v1,1j*v1,-1j*v1],axis=0))[0]
    report["notes"]=["Each memory is a persistent Robin wall compliance, not lingering wave activity.",
        "Wall response matrix is calibrated one bit at a time; these ranks are linearized addressability, not multi-bit recognition accuracy.",
        "Complex-ray filter is derived for positive-frequency free half-waves. The bounded, damped cavity does not satisfy its theorem assumptions.",
        "Four-phase comparison assumes two independently driven coherent sources with a phase-quadrature controller; it is not Cook-Mertz catalytic computation.",
        "Calibration and classification assume exactly one altered wall patch; arbitrary multi-patch memory has NOT been decoded.",
        "No lens can increase the rank of information already absent from a fixed full waveform."]
    return report


if __name__ == "__main__":
    result=bench()
    path=Path(__file__).with_name("receipt.json")
    path.write_text(json.dumps(result,indent=2))
    print("Ray identity error:",result["openai_complex_ray"]["multiplier_identity_max_abs_error"])
    for key in ("single_source_signed_trace","single_source_intensity_trace","single_source_five_pixel_patch","three_source_signed_traces"):
        print(key, "rank:",result[key]["rank_at_1pct_largest_singular"],"max col corr:",round(result[key]["largest_column_correlation"],4))
    print("Output:",path)
