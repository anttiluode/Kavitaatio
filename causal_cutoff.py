"""Six-wall causal horizon: when is material interaction first detectable?

Signed inclusion-exclusion of 64 configurations for SIX selected wall
material parameters. This is not a cheap readout or a proof of propagation
bounds, but a numerical test for delayed high-order scattering.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import experiment as exp

DEFAULT_SUBSET=(0,2,3,4,6,7)


def evaluate(stiffness=.28,source=0,subset=DEFAULT_SUBSET,steps=1000):
    assert len(subset)==6 and len(set(subset))==6
    assert all(0<=x<exp.K for x in subset)
    exp.STEPS=steps;exp.START=0;exp.STOP=steps;exp.STRIDE=2
    setup=exp.geometry()
    nread=(steps+1)//2
    interaction=np.zeros(nread)
    baseline=np.zeros(nread)
    maxindividual=0.
    for mask in range(64):
        bits=np.zeros(exp.K)
        for k,index in enumerate(subset):
            if mask & (1<<k):bits[index]=1.
        wave,_=exp.simulate(bits,source,setup=setup,stiffness=stiffness)
        y=wave[:,0]
        interaction+=(-1)**(6-int(bits.sum()))*y
        maxindividual=max(maxindividual,float(np.max(np.abs(y))))
        if mask==0:baseline[:]=y
    windows={}
    for low,high in ((0,145),(145,345),(345,500),(500,700),(700,steps)):
        if low>=steps:continue
        data=interaction[low//2:min(high,steps)//2]
        if len(data):
            windows[f'{low}-{min(high,steps)}']={
                'rms':float(np.sqrt(np.mean(data*data))),
                'peak_abs':float(np.max(np.abs(data)))}
    onset={str(th):(int(2*np.flatnonzero(np.abs(interaction)>th)[0])
                   if np.any(np.abs(interaction)>th) else None)
           for th in (1e-8,1e-4,1e-2)}
    report={'stiffness':stiffness,'source_index':source,'wall_subset':list(subset),
        'wave_steps':steps,'time_sample_spacing_steps':2,
        'six_wall_mixed_response':windows,'onset_step_by_threshold':onset,
        'max_single_state_abs':maxindividual,
        'overall_six_wall_response_peak':float(np.max(np.abs(interaction))),
        'interpretation':'wall changes persist; wave linear in u, nonlinear in the wall parameters',
        'calibration_cost':'2^6=64 material states for this diagnostic coefficient'}
    return report,interaction,baseline


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stiffness',type=float,default=.28)
    parser.add_argument('--source',type=int,default=0,choices=(0,1,2))
    parser.add_argument('--steps',type=int,default=1000)
    args=parser.parse_args()
    report,interaction,baseline=evaluate(args.stiffness,args.source,steps=args.steps)
    root=Path(__file__).resolve().parent
    suffix=f'_{args.stiffness:.2f}_src{args.source}'
    (root/f'six_wall_horizon{suffix}.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(root/f'six_wall_horizon{suffix}.npz',
        interaction=interaction,baseline=baseline,
        time_steps=np.arange(len(interaction))*2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
