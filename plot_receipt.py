"""Make static figures from Kavitaatio's deterministic numerical receipt."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from experiment import geometry, simulate, N, K

out=Path(__file__).parent
r=json.loads((out/"receipt.json").read_text())
mask,sector,sources,receiver=geometry()

fig,ax=plt.subplots(figsize=(8,6))
wall=np.full(mask.shape,np.nan)
for a in (0,1):
    for d in (-1,1):
        nei=np.roll(mask,d,axis=a)
        if a==0:nei[0 if d==1 else -1,:]=False
        else:nei[:,0 if d==1 else -1]=False
        wall[mask & ~nei]=sector[mask & ~nei]
ax.imshow(mask,cmap="Greys",alpha=.16,origin="lower")
ax.imshow(np.ma.masked_invalid(wall),cmap="tab10",vmin=0,vmax=9,origin="lower")
for i,(yy,xx) in enumerate(sources):
    ax.scatter(xx,yy,marker="o",s=80,edgecolors="k",label=f"Ping port {i+1}")
ax.scatter(receiver[1],receiver[0],marker="*",s=220,label="Listener")
ax.legend(loc="upper right")
ax.set(xlim=(0,N-1),ylim=(0,N-1),title="Kavitaatio: 8 persistent wall arc addresses (Robin compliance)",xlabel="grid x",ylabel="grid y")
fig.tight_layout();fig.savefig(out/"wall_geometry.png",dpi=150);plt.close(fig)

labels=["1 pt: signed","1 pt: intensity","5 px: signed","3 sources: signed","4 phases: intensity"]
keys=["single_source_signed_trace","single_source_intensity_trace","single_source_five_pixel_patch","three_source_signed_traces","two_source_four_phase_intensity"]
vals=[100*r[k]["one_wall_patch_8_way_accuracy_at_fixed_noise"] for k in keys]
fig,ax=plt.subplots(figsize=(9,5))
bars=ax.bar(labels,vals)
ax.axhline(12.5,linestyle="--",color="gray",label="8-way chance")
ax.set_ylim(0,112)
ax.set_ylabel("Known single-patch address identified (%)")
ax.set_title("Cavity wall address reading: 300 noisy samples per address")
ax.tick_params(axis="x",rotation=18);ax.legend()
for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+2,f"{v:.1f}%",ha="center")
fig.tight_layout();fig.savefig(out/"address_accuracy.png",dpi=150);plt.close(fig)

base,_=simulate(np.zeros(K),0)
k=np.zeros(K);k[0]=1
v,_=simulate(k,0)
fig,ax=plt.subplots(figsize=(8,4))
ax.plot(np.arange(len(base))*2+145,base[:,0],label="Blank wall")
ax.plot(np.arange(len(v))*2+145,v[:,0],label="Wall sector 0 modified")
ax.set(title="Listener time trace: persistent wall alters returning echo",xlabel="simulation step",ylabel="signed displacement")
ax.legend();fig.tight_layout();fig.savefig(out/"echo_trace.png",dpi=150);plt.close(fig)
print("Figures written")
