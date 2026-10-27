"""Run K=8 vs K=16 causal wall-interaction and wave-speed controls.

Measured subsets are selected in advance, without a 2^16 state catalogue.
All result files are generated locally with deterministic NumPy/Numba.
"""
import json
from pathlib import Path
from k16_horizon import inclusion_exclusion,C
from waypoint_tour import tour

CASES = (
    (8, (0,2,3,4,6,7)),      # Original widely distributed six-wall control
    (16,(0,1,2,3,4)),
    (16,(0,1,2,3,4,5)),
    (16,(0,1,2,3,4,5,6)),
    (16,(0,1,2,3,4,5,6,7)),
    (16,(0,1,2,3,4,5,6,7,8)),
)


def run():
    out={
        'protocol':{'grid':96,'dt':.5,'speed':C,'wall_stiffness':.28,
                    'onset_threshold':1e-4,'window':[245,345],
                    'caution':'Selected subsets: no universal degree theorem or 16-bit decoder.'},
        'cases':[]
    }
    for k,subset in CASES:
        stats,_ = inclusion_exclusion(k,subset,source=0,nsteps=420)
        bound=tour(k,subset)
        stats.update({
            'geometric_lower_bound_length':bound['tour_lower_bound_length'],
            'earliest_arrival_lower_bound_steps':bound['earliest_arrival_lower_bound_steps']})
        out['cases'].append(stats)
        print('K',k,'order',len(subset),'earliest',round(bound['earliest_arrival_lower_bound_steps'],2),
              'measured',stats['onset_at_0.0001'],'window-RMS',stats['rms_245_345'],flush=True)
    out['speed_control']=[]
    for speed in (.5,.65,.8):
        stats,_=inclusion_exclusion(16,tuple(range(7)),speed=speed,nsteps=440)
        onset=stats['onset_at_0.0001']
        out['speed_control'].append({'speed':speed,'onset_at_0.0001':onset,
            'speed_times_onset':speed*onset,'rms_245_345':stats['rms_245_345']})
        print('Speed',speed,'onset',onset,flush=True)
    path=Path(__file__).with_name('k16_study_receipt.json')
    path.write_text(json.dumps(out,indent=2))
    print('Wrote',path)
    return out


if __name__=='__main__':
    run()
