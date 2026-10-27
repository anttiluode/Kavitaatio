# K16: when finer wall addresses defeat fixed-order calibration

**27 October 2026 — Kavitaatio's geometry/causality scaling gate.**

## Question

Claude correctly identified a geometric causal cutoff in the K=8 experiment:
high-order Boolean mixed wall responses are not visible until repeated waves
can visit all selected material patches and reach the listener. The tempting
extrapolation was that a **16-bit wall memory could be decoded by calibrating
only orders <=4 (2,517 of 65,536 configurations)**.

This follow-up tests the *necessary premise*, before attempting 2,517 calibration
runs: **Does a degree-four wall-response approximation remain exact through
the same short observation window after we split eight wall arcs into sixteen?**

**Answer: no for this geometry and window.** A seventh-order interaction is
measurable and an eighth-order one appears near the deadline. Smaller adjacent
wall patches can interact in the same amount of time.

## What was held fixed

- Identical 96x96 ellipse, wave timestep 0.5, wave speed 0.65, scalar wave
  stencil, wall stiffness 0.28, one source at the left focus and one signed
  listener at the right focus.
- Same physical room and recording deadline **step 345**.
- The K=16 boundary simply divides the original wall into twice as many
  angular sectors. We use **contiguous low-index sectors** (0,1,...,k-1).
  The K=8 reference uses the original spaced-out subset (0,2,3,4,6,7).
- The observable is the *signed* kth-order Boolean mixed wall response:
  `I_S(t) = sum_(B subset S) (-1)^(|S|-|B|) y_B(t)`.
  This is nonlinearity in **material configuration**, not nonlinear wave
  dynamics at fixed material state, and not intensity interference.
- We threshold `|I_S(t)| > 1e-4` to report an **operational onset**. Changing
  the threshold changes the onset. Gaussian initial pulses have noncompact
  tails, so we do not claim literal exact zeros at all earlier times.

## Independently computed geometric deadline

`waypoint_tour.py` finds an *optimistic minimum Euclidean path* from the
source to at least one actual boundary grid cell in every selected sector,
then to the receiver. It uses dynamic programming over the chosen sectors
and boundary-cell positions, rather than allowing each pair of sectors to
use incompatible independent closest points.

It provides an **earliest-arrival lower bound**, NOT a specular ray tracer.
Real waves can arrive later because of reflection physics, diffraction and
dispersion. A localized source with strictly finite support would allow a
stronger causality statement than the Gaussian pulse used here.

| Cavity | Material interaction | Optimistic earliest arrival (steps) | Measured onset, threshold 1e-4 | RMS, steps 245–345 |
| --- | --- | ---: | ---: | ---: |
| K=8 | Six spread-out sectors | 406.3 | 420 | 7.16e-15 |
| K=16 | Five neighboring sectors | 273.0 | 273 | 0.03173 |
| K=16 | Six neighboring sectors | 287.9 | 291 | 0.01462 |
| K=16 | Seven neighboring sectors | 303.2 | 312 | 0.005056 |
| K=16 | Eight neighboring sectors | 323.8 | 340 | 5.38e-5 |
| K=16 | Nine neighboring sectors | 347.8 | 369 | 8.28e-9 |

The measured seventh-order effect is more than numerical roundoff. Its
100-sample-window RMS is 0.005 versus the preceding K=8 sixth-order
control at ~7e-15. The eighth-order effect is much weaker, so **whether
it matters to real memory decoding depends on sensor noise**. This is
not a noise-robust 16-bit decoder; no full K=16 decoding task was run.

## Independent propagation-speed control

For seven neighboring sectors, changing only the wave speed:

| Wave speed | Onset step | Speed × onset |
| ---: | ---: | ---: |
| 0.50 | 405 | 202.5 |
| 0.65 | 312 | 202.8 |
| 0.80 | 254 | 203.2 |

The near-constant product supports a finite-travel-time mechanism.
It is a numerical *control*, not a theorem about all boundaries.

## The K=8 symmetry correction

The K=8 focus-to-focus, single-pixel geometry is mirror-symmetric about
the ellipse's major (horizontal) axis. Therefore corresponding upper/lower
wall addresses are hard to separate. Breaking that symmetry by moving
the source eight grid cells off-axis changes the *1%-of-largest-singular-value*
single-source response rank from **4/8 to 8/8**, just as Claude observed.
The identical K=16 test gives **6/16 to 12/16**: doubling the number of
wall addresses creates finer near-collinear responses even with an
off-axis ping. These are **thresholded numerical ranks**, not exact
algebraic ranks or state-recovery probabilities.

## Scaling interpretation and what remains open

For a binary K-patch wall, every fixed finite-time response has a unique
Boolean multilinear expansion. Its coefficient for a subset of patches
is an inclusion–exclusion mixed difference. In the discrete wave-step
operator `A(w)=A0+sum_i w_i Ai`, each interaction corresponds to ordered
insertions along paths that encounter those wall supports.

Finite propagation does limit which combinations can affect a listener by
a deadline. **But that bound depends on the distribution and size of the
patches.** Cutting a fixed wall into more pieces can let the same path
touch more distinct patches. A claim of polynomial calibration
`sum_(k<=k*(T,K)) binom(K,k)` requires *a bound on k*(T,K) independent
of K* (or a stronger structural sparsity restriction). This experiment
falsifies the straightforward extrapolation `k*=4 at K=16` for the
original physical geometry.

This result is about causal paths and multiple scattering in a toy
**bounded wave cavity**. It neither follows from, nor tests, the
October 25 OpenAI local-smoothing **free Euclidean wave theorem**.
The prior Kavitaatio OpenAI-inspired directional multiplier remains
a separate spectral representation experiment.

## Reproduce

Requires Python 3.11+, NumPy and Numba:

~~~bash
pip install numpy numba
python -m unittest -v test_k16_horizon
python k16_study.py
~~~

`k16_study.py` writes `k16_study_receipt.json`. It uses at most
`2^9=512` wall configurations for any tested high-order coefficient,
not `2^16=65536`. It doesn't fit a multi-bit decoder and it doesn't
learn wall coefficients. For plots (optional), install Matplotlib.

## Next falsification gates

1. **Grid resolution**: does the step-onset × wave-speed relationship
   persist at 128 or 192 grid cells with fixed physical geometry?
2. **Patch locality**: compare contiguous and widely distributed
   seven-patch subsets **matched for total written arc length**.
3. **Sensor reliability**: use a fixed Gaussian noise floor, optimize
   input port selection at a fixed measurement/energy budget and test
   blind 16-bit states. Avoid choosing a window after seeing labels.
4. **Calibration scaling**: estimate and exploit a **sparse set of
   geometrically reachable subsets**, rather than calibrating every
   subset up to an order guessed in advance.
5. **Alternative boundary shapes**: ellipse, circle, chaotic stadium
   with comparable area and wall perimeter.

References: [OpenAI local smoothing](https://github.com/openai/math/tree/main/preprints/Local-smoothing-for-the-Euclidean-wave-equation-October-25-2026),
[Kavitaatio original cavity code](experiment.py),
[CausalHorizon's finite-event formula](https://github.com/anttiluode/CausalHorizon/blob/main/THEOREM.md).
