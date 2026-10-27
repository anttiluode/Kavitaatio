"""Kavitaatio: held-out multibit decoding from measured wall interactions.

Degree-d calibration observes ONLY boundary settings with <=d active sectors.
Full enumeration is an explicitly expensive ground-truth benchmark, not the
proposed memory-reading procedure. NumPy only; requires experiment.py.
"""
import json
from pathlib import Path
import numpy as np
import experiment as exp

K=8
BITS=((np.arange(256)[:,None]>>np.arange(K))&1).astype(float)
ORDER=BITS.sum(axis=1).astype(int)
ROOT=Path(__file__).resolve().parent


def truth(stiffness):
    path=ROOT/f'oracle_wall_responses_{stiffness:.3f}.npz'
    if path.exists():
        wave=np.load(path)['wave']
        assert wave.shape==(256,3,100)
        return wave
    geom=exp.geometry()
    wave=np.empty((256,3,100))
    for mask in range(256):
        for source in range(3):
            samples,_=exp.simulate(BITS[mask],source,setup=geom,stiffness=stiffness)
            wave[mask,source]=samples[:,0]
        if mask%64==0:print('oracle evaluation',stiffness,mask,'/256',flush=True)
    np.savez_compressed(path,wave=wave)
    return wave


def fit(wave,degree):
    """Binary Möbius coefficients through degree, no held-out responses."""
    coeff=np.zeros_like(wave)
    for order in range(degree+1):
        for mask in np.flatnonzero(ORDER==order):
            c=wave[mask].copy()
            sub=(mask-1)&mask
            while sub:
                c-=coeff[sub]
                sub=(sub-1)&mask
            if mask:c-=coeff[0]
            coeff[mask]=c
    predicted=coeff.copy()
    for k in range(K):
        for mask in range(256):
            if mask&(1<<k):predicted[mask]+=predicted[mask^(1<<k)]
    assert np.max(np.abs(predicted[ORDER<=degree]-wave[ORDER<=degree]))<1e-12
    return predicted,coeff


def decode(actual,predicted,holdout,sources,sigma=0.,repeats=1,seed=493):
    """Nearest of 256 model *predictions*, not 256 calibrated measurements."""
    candidates=predicted[:,sources,:].reshape(256,-1)
    target=actual[holdout][:,sources,:]
    norms=np.sum(candidates**2,axis=1)
    rng=np.random.default_rng(seed)
    hits=0;bit_errors=0
    for _ in range(repeats):
        noisy=target if sigma==0 else target+rng.normal(0,sigma,target.shape)
        flat=noisy.reshape(len(holdout),-1)
        distance=norms[None]-2*flat@candidates.T
        guesses=np.argmin(distance,axis=1)
        hits+=int((guesses==holdout).sum())
        bit_errors+=int(np.sum(BITS[guesses]!=BITS[holdout]))
    total=len(holdout)*repeats
    return {'exact':hits,'total':total,'exact_pct':round(100*hits/total,2),
            'bit_accuracy_pct':round(100*(1-bit_errors/(K*total)),2)}


def benchmark():
    strong=truth(.28)
    one=np.array([strong[1<<i,0]-strong[0,0] for i in range(K)])
    sigma=float(.04*np.median(np.linalg.norm(one,axis=1)/10))
    results={'protocol':{'bits':K,'sources':3,'samples_per_ping':100,
        'noise_absolute_std':sigma,'noise_repetitions':10,
        'calibration':'only Hamming weight <= degree',
        'test':'every Hamming weight > degree, evaluated by 256-state oracle',
        'warning':'full oracle is for scoring only; requires 256 simulations per source'},
        'strengths':{}}
    for stiffness in (.28,.07):
        wave=strong if stiffness==.28 else truth(stiffness)
        entry={}
        for degree in (1,2,3,4):
            predicted,_=fit(wave,degree)
            held=np.flatnonzero(ORDER>degree)
            error=np.linalg.norm((predicted[held]-wave[held]).reshape(len(held),-1),axis=1)/(
                np.linalg.norm((wave[held]-wave[0]).reshape(len(held),-1),axis=1)+1e-12)
            sensors={}
            for label,sources in (('one',[0]),('two',[0,1]),('three',[0,1,2])):
                sensors[label]={'noiseless':decode(wave,predicted,held,sources),
                    'gaussian_noise':decode(wave,predicted,held,sources,sigma,10)}
            entry[str(degree)]={'calibration_states_per_source':int(np.sum(ORDER<=degree)),
                 'unseen_states':int(len(held)),
                 'median_relative_waveform_error':float(np.median(error)),
                 'decoding':sensors}
            print('wall',stiffness,'degree',degree,'calibration',
                int(np.sum(ORDER<=degree)),'heldout',len(held),
                '3 pings noisy',sensors['three']['gaussian_noise']['exact_pct'],'%',flush=True)
        common=np.flatnonzero(ORDER>=5)
        entry['common_5plus_holdout']={
            str(d):decode(wave,fit(wave,d)[0],common,[0,1,2]) for d in (1,2,3,4)}
        base=wave[0]
        singles=np.array([wave[1<<i]-base for i in range(K)])
        pairs=[]
        for i in range(K):
            for j in range(i+1,K):
                pairs.append(wave[(1<<i)|(1<<j)]-wave[1<<i]-wave[1<<j]+base)
        entry['pair_to_single_rms']=float(np.sqrt(np.mean(np.array(pairs)**2))/
                                          np.sqrt(np.mean(singles**2)))
        results['strengths'][str(stiffness)]=entry
    (ROOT/'wall_interaction_receipt.json').write_text(json.dumps(results,indent=2))
    return results


if __name__=='__main__':
    benchmark()
