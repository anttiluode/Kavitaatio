"""Optimal Euclidean waypoint tour through selected cavity-wall addresses.

Dynamic programming uses actual boundary *cells*, not per-sector center points.
Its travel time is an optimistic geometrical lower bound: ignores the true
specular-reflection constraint, diffraction, grid dispersion and pulse tails.
"""
import numpy as np
from numba import njit
from k16_horizon import setup,N,C,DT


@njit(cache=True)
def tour_dp(points, labels, start, finish, k):
    size=1<<k; n=len(points)
    dp=np.full((size,n),1e20)
    for i in range(n):
        y,x=points[i]
        dp[1<<labels[i],i]=((y-start[0])**2+(x-start[1])**2)**.5
    for mask in range(1,size):
        for i in range(n):
            base=dp[mask,i]
            if base>1e19:continue
            for j in range(n):
                bit=1<<labels[j]
                if mask&bit:continue
                ny,nx=points[j]
                oy,ox=points[i]
                v=base+((ny-oy)**2+(nx-ox)**2)**.5
                nxt=mask|bit
                if v<dp[nxt,j]:dp[nxt,j]=v
    best=1e20
    for i in range(n):
        yy,xx=points[i]
        v=dp[-1,i]+((yy-finish[0])**2+(xx-finish[1])**2)**.5
        if v<best:best=v
    return best


def tour(K,subset,source_number=0,speed=C):
    inside,sectors,source,receiver=setup(K,source_number)
    edge=np.zeros((N,N),bool)
    for ax in (0,1):
        for direction in (-1,1):
            near=np.roll(inside,direction,axis=ax)
            if ax==0: near[0 if direction==1 else -1,:]=False
            else: near[:,0 if direction==1 else -1]=False
            edge|=inside&~near
    positions=np.column_stack(np.nonzero(edge))
    labels=sectors[edge]
    choose=np.isin(labels,subset)
    cells=positions[choose].astype(np.float64)
    lut={s:i for i,s in enumerate(subset)}
    compact=np.asarray([lut[x] for x in labels[choose]],np.int64)
    length=tour_dp(cells,compact,np.array(source,float),np.array(receiver,float),len(subset))
    return {
        'K':K,'subset':list(subset),'source':source,'receiver':receiver,
        'tour_lower_bound_length':float(length),
        'earliest_arrival_lower_bound_steps':float(length/(speed*DT)),
        'boundary_cells_in_calculation':int(len(cells))
    }
