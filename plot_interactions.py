"""Optional Matplotlib figures from the numerical experiments."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent

fig,ax=plt.subplots(figsize=(9,5))
for stiffness in (.28,.07):
    file=root/f'six_wall_horizon_{stiffness:.2f}_src0.npz'
    data=np.load(file)
    wave=data['interaction']
    time=data['time_steps']
    length=21
    rolling=np.sqrt(np.convolve(wave*wave,np.ones(length)/length,'same'))
    ax.semilogy(time,np.maximum(rolling,1e-13),label=f'Wall stiffness {stiffness:.2f}')
ax.axvline(345,linestyle='--',label='Original read window ends')
ax.set(xlim=(0,1000),ylim=(1e-13,10),xlabel='Simulation step',
       ylabel='Rolling RMS of sixth-order material interaction',
       title='High-order wall interaction appears on delayed echo paths')
ax.legend();fig.tight_layout()
fig.savefig(root/'six_wall_horizon.png',dpi=160)
plt.close(fig)

receipt=json.loads((root/'wall_interaction_receipt.json').read_text())['strengths']
fig,ax=plt.subplots(figsize=(9,5))
degrees=np.arange(1,5)
for idx,name in enumerate(('0.28','0.07')):
    row=receipt[name]['common_5plus_holdout']
    values=[100*row[str(d)]['exact']/row[str(d)]['total'] for d in degrees]
    ax.bar(degrees+(idx-.5)*.35,values,width=.34,label=f'Stiffness {name}')
ax.set(xticks=degrees,xticklabels=['Linear (9)','Pairs (37)','Triples (93)','Quadruples (163)'],
       ylim=(0,110),xlabel='Interaction order (wall states calibrated per source)',
       ylabel='Exact memories recovered, SAME 93 unseen 5+ bit patterns (%)',
       title='Calibrated interaction order versus memory recovery')
ax.legend();fig.tight_layout()
fig.savefig(root/'degree_vs_recovery.png',dpi=160)
plt.close(fig)
print('Saved six_wall_horizon.png and degree_vs_recovery.png')
