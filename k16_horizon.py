"""Kavitaatio K=16 finite-propagation experiment.

Reimplements merged experiment.py's Neumann/Robin stencil with Numba
acceleration (not a different physical model). A 96x96, 350-step K=8
regression matches original traces within 3e-15.

K16 switches from eight to sixteen boundary sectors. This code only measures
preselected high-order Boolean mixed differences; it does not claim to decode
all 65536 16-bit memories.
"""
import numpy as np
from numba import njit
from experiment import N, C, DT, CENTER, A, B


@njit(cache=True)
def wave_trace(mask, sectors, source, sensor, bitflags, wall_gain, nsteps, speed=C):
    n = N
    u = np.zeros((n,n)); old = np.zeros((n,n)); new = np.zeros((n,n))
    sy,sx = source
    for row in range(n):
        for col in range(n):
            if mask[row,col]:
                u[row,col] = np.exp(-((col-sx)**2 + (row-sy)**2)/(2*1.55**2))
    out = np.empty(nsteps, dtype=np.float64)
    ry,rx = sensor
    hdt = 0.006*DT/2
    s2 = (speed*DT)**2
    for t in range(nsteps):
        for row in range(1,n-1):
            for col in range(1,n-1):
                if not mask[row,col]:
                    continue
                pos=u[row,col]
                sm=0.
                missing=0
                if mask[row+1,col]: sm += u[row+1,col]-pos
                else: missing+=1
                if mask[row-1,col]: sm += u[row-1,col]-pos
                else: missing+=1
                if mask[row,col+1]: sm += u[row,col+1]-pos
                else: missing+=1
                if mask[row,col-1]: sm += u[row,col-1]-pos
                else: missing+=1
                if bitflags[sectors[row,col]]:
                    sm -= wall_gain*missing*pos
                new[row,col]=(2*pos-(1-hdt)*old[row,col]+s2*sm)/(1+hdt)
        old,u,new = u,new,old
        out[t]=u[ry,rx]
    return out


def setup(K=16, source_number=0):
    yy,xx = np.mgrid[0:N,0:N]
    X=xx-CENTER; Y=yy-CENTER
    inside=(X/A)**2+(Y/B)**2 <= 1
    theta=np.arctan2(Y/B,X/A)
    sectors=(np.floor(((theta+np.pi)/(2*np.pi))*K).astype(np.int32)%K)
    f=np.sqrt(A*A-B*B)
    source=[
        (round(CENTER),round(CENTER-f)),
        (round(CENTER+8),round(CENTER-f)),
        (round(CENTER-8),round(CENTER-f))
    ][source_number]
    receiver=(round(CENTER),round(CENTER+f))
    return inside,sectors,source,receiver


def inclusion_exclusion(K, subset, source=0, stiffness=.28, nsteps=520, speed=C):
    """Exact k-way Boolean mixed-difference: needs 2**k forward simulations."""
    mask,sectors,src,receiver=setup(K,source)
    assert len(subset)==len(set(subset))
    assert all(0<=k<K for k in subset)
    mixed=np.zeros(nsteps)
    peak_state=0.
    for q in range(1<<len(subset)):
        bits=np.zeros(K,dtype=np.bool_)
        for j,k in enumerate(subset):
            bits[k]=((q>>j)&1)==1
        trace=wave_trace(mask,sectors,src,receiver,bits,stiffness,nsteps,speed)
        mixed+=(-1)**(len(subset)-int(bits.sum()))*trace
        peak_state=max(peak_state,float(np.max(np.abs(trace))))
    result={'K':K,'subset':list(subset),'source':source,
        'strength':stiffness,'wave_speed':speed,'nsteps':nsteps,
        'max_ordinary_wave':peak_state,
        'max_high_order':float(np.max(np.abs(mixed)))}
    for threshold in (1e-8,1e-6,1e-4,1e-3,1e-2):
        ix=np.flatnonzero(np.abs(mixed)>threshold)
        result[f'onset_at_{threshold}']=int(ix[0]) if len(ix) else None
    for low,high in ((0,145),(145,245),(245,345),(345,440),(440,520)):
        if low>=nsteps:continue
        seg=mixed[low:min(nsteps,high)]
        result[f'rms_{low}_{min(nsteps,high)}']=float(np.sqrt(np.mean(seg*seg)))
    return result,mixed
