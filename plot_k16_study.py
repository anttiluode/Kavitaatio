"""Optional plots for the K16 geometry-horizon receipt."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).parent
results=json.loads((root/'k16_study_receipt.json').read_text())
rows=[v for v in results['cases'] if v['K']==16]
k=np.array([len(v['subset']) for v in rows])
earliest=np.array([v['earliest_arrival_lower_bound_steps'] for v in rows])
onsets=np.array([v['onset_at_0.0001'] for v in rows])
rms=np.array([v['rms_245_345'] for v in rows])

fig,ax=plt.subplots(figsize=(9,5.4))
ax.plot(k,earliest,marker='o',label='Earliest geometric arrival')
ax.plot(k,onsets,marker='s',label='Measured threshold onset')
ax.axhline(345,ls='--',label='Original deadline')
ax.set(xlabel='Distinct neighboring wall patches',ylabel='Simulation step',xticks=k,
       title='Geometry predicts high-order wall-interaction onset')
ax.legend();fig.tight_layout()
fig.savefig(root/'k16_geometry_horizon.png',dpi=175);plt.close(fig)

fig,ax=plt.subplots(figsize=(9,5.4))
ax.semilogy(k,np.maximum(rms,1e-16),marker='o',label='16-sector cavity')
ax.scatter([6],[results['cases'][0]['rms_245_345']],marker='s',s=110,
           label='8-sector spread-out six-wall control')
ax.set(xlabel='Number of affected wall sectors',xticks=k,
       ylabel='RMS mixed response, steps 245–345',
       title='Finer wall patches allow higher interaction orders before deadline')
ax.legend();fig.tight_layout()
fig.savefig(root/'k16_degree_amplitude.png',dpi=175);plt.close(fig)

speed=results['speed_control']
c=np.array([v['speed'] for v in speed])
time=np.array([v['onset_at_0.0001'] for v in speed])
fig,ax=plt.subplots(figsize=(8.5,5.1))
ax.plot(1/c,time,'o-',label='Measured onset')
ax.plot(1/c,np.mean(c*time)/c,'--',label='Inverse-speed prediction')
ax.set(xlabel='Inverse speed',ylabel='Measured seventh-order onset',
       title='Travel-time control: onset tracks 1 / wave speed')
ax.legend();fig.tight_layout()
fig.savefig(root/'k16_speed_control.png',dpi=175);plt.close(fig)
print('Saved K16 geometry, amplitude and speed-control plots')
