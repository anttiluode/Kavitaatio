# Kavitaatio — an etched-wall focus and interrogation laboratory

Research spike, 27 October 2026. This repository is a controlled numerical experiment, **not a proof of Sogge's conjecture**, a physical-neuron demonstration, or a practical SAT/Williams implementation.

The question: **Can an elliptical reflecting cavity retain information as changes in its wall and reveal those changes through coherent, deliberately chosen pings?** And, separately, **does the exact complex-ray directional multiplier appearing in the new OpenAI local-smoothing manuscript help read that information better than ordinary directional filters?**


## October 27 follow-up — wall interaction order and a causal listening horizon

**Short finding:** The single-wall response is not enough to predict multiple written wall bits. Real, measurable interactions between changed wall patches repair some of the failure. At the original observation deadline, wall combinations have a relatively low useful interaction order. Wait longer, and an interaction involving six separate wall changes becomes clearly measurable.

Scripts: [wall_interactions.py](wall_interactions.py), [causal_cutoff.py](causal_cutoff.py), [test_wall_interactions.py](test_wall_interactions.py). The existing \`experiment.py\` now accepts an optional \`stiffness\` argument; its default 0.28 gives identical behavior to the earlier commit.

### Experimental controls

- Eight permanent Robin wall parameters, all written **before** any wave excitation. No learned wall plasticity.
- Same 96 × 96 Neumann/Robin ellipse and signed 100-sample trace at the second focus. One, two or three distinct physical ping positions can be used.
- Define the signed two-wall mixed response \`I_AB = y_AB - y_A - y_B + y_empty\`. This is **non-additivity with respect to the wall/material setting**. It is NOT the ordinary quadratic optical-intensity cross term.
- Degree-\`d\` prediction is the Boolean Möbius expansion using **only** the wall configurations containing at most \`d\` active patches. The unseen configurations containing \`d+1\` or more bits are never used for calibration.
- A separate 256-state forward-model enumeration is used only to provide ground truth for scoring and to analyze interaction coefficients. This is a diagnostic luxury, **not** a proposed 256-shot learning algorithm. The tests check that changing held-out responses cannot change the fitted model.
- Noisy testing uses independent Gaussian detector noise with the same **absolute standard deviation ~0.02549** at both wall stiffnesses. This equals 4% of the median single-patch waveform RMS for the 0.28-stiffness reference. Ten noisy repeats per held-out memory, with deterministic seeds; three physical pings per test memory in the headline numbers.

| Maximum material interaction order fitted | Physical calibration states / source | Truly unseen memories | Original 0.28 wall, 3 pings, noisy | Weaker 0.07 wall, 3 pings, noisy |
| --- | ---: | ---: | ---: | ---: |
| 1 (individual walls) | 9 | 247 | 2.87% | 4.86% |
| 2 (wall pairs) | 37 | 219 | 32.92% | 54.79% |
| 3 (triples included) | 93 | 163 | **85.89%** | **100%** |
| 4 (quadruples included) | 163 | 93 | **100%** | **100%** |

**Caution:** each row has a different unseen test set. For a common **93-memories-with-five-or-more-bits** holdout (three pings, noiseless), recovery with orders 1, 2, 3 and 4 respectively was **0/93, 0/93, 70/93, 93/93** at strength 0.28, and **0/93, 10/93, 93/93, 93/93** at strength 0.07. The model predicts all 256 candidate words, but only scores wall words outside its own calibration set. Order four calibrates 163 of 256 wall words **per physical source**, a 36% calibration saving, not exponential compression.

At 0.28 stiffness, the direct two-wall interaction RMS was **0.475** times the single-wall response RMS. The degree-2 approximation missed many joint states even though it fit all 37 calibrated states exactly. Higher-order scattering is real.

### A separate time-horizon experiment

Using all 64 subsets of wall sectors \`(0,2,3,4,6,7)\`, \`causal_cutoff.py\` calculates a signed **sixth-order** mixed material response from a **1,000-step** record of a single source (index 0). This is an explicit expensive inclusion–exclusion control, not efficient recovery.

| Time steps | Six-wall RMS, spring 0.28 | Six-wall RMS, spring 0.07 |
| --- | ---: | ---: |
| 0–145 | 0 | 0 |
| 145–345 | 5.55e-15 | 5.81e-15 |
| 345–500 | 0.1234 | 0.00178 |
| 500–700 | 1.1152 | 0.2206 |
| 700–1000 | 3.0259 | 2.3970 |

For 0.28 the first \`|I_6(t)| > 1e-8\` is at step **390**. For 0.07 it is at step **412**. These are **threshold-dependent numerical onsets for one geometry and one six-patch subset**, not a universal wave-speed theorem or a mathematical proof of exact zeros. The higher-order interaction only becomes visible after a longer propagation window. *Listening time is part of the observable computational complexity.*

The simulator is still a linear damped wave equation for each **fixed** wall; the dependence on the changed wall coefficients is nonlinear through repeated propagation. This connects conceptually to the ordered perturbed propagator in [CausalHorizon](https://github.com/anttiluode/CausalHorizon/blob/main/THEOREM.md). Here we directly measured finite differences of **persistent material parameters**, not the time-event resolvent or a Cook–Mertz register-reset protocol.

**Scope of new OpenAI wave paper:** The paper's imaginary-ray directional Fourier multiplier was implemented in the previous experiment. Its free-Euclidean-wave local-smoothing theorem does **not** prove these bounded-cavity memory or interaction results. The current follow-up studies *why focusing alone cannot identify wall addresses, and how much causal interaction history a listener needs*. Novel physical effects are demonstrated **within this toy model**; no claim of a new general theorem or physical neural device is made.

### Reproduce

\`\`\`bash
python -m pip install numpy
python -m unittest -v test_wall_interactions
python wall_interactions.py
python causal_cutoff.py --stiffness .28 --source 0
python causal_cutoff.py --stiffness .07 --source 0
# Optional figures:
python -m pip install matplotlib
python plot_interactions.py
\`\`\`

\`wall_interactions.py\` first generates and caches full truth tables in \`oracle_wall_responses_*.npz\` (768 source/wall simulations per stiffness) so all 256 states are scored without leakage. The summary \`wall_interaction_receipt.json\` and time-horizon files are written locally. The simulations are deterministic, and the cache files are deliberately not versioned.

## Two pieces of physics that must not be conflated

- **Remembering wall:** Eight arc sectors modify persistent Robin boundary compliance. Zero compliance gives discrete Neumann reflection. Waves may damp away; the Robin coefficients remain until externally changed. This is an *assumed physical storage variable*, not a learned wall-update rule.
- **Wave readout:** A pulse launched near the first ellipse focus travels and reflects. We measure a signed time trace at the other focus, a five-cell receiving aperture, and several changed source positions. Repeated reflections make the mapping from **joint wall settings** to the final echo generally non-additive even though the wave equation is linear for each fixed wall.

The simulation runs at 96×96 cells, wave speed 0.65, timestep 0.5, 350 steps, damping 0.006, and records 100 evenly spaced frames. The numerical stencil handles missing neighbor faces as zero normal flux (Neumann), with a negative boundary-spring term for active patches. It does **not** repeat the zeroed-exterior Neumann/Dirichlet mismatch of ResonantNeuron V12.

## Measured outcomes

Run `python experiment.py` with NumPy; the script emits `receipt.json` deterministically.

An eight-column response matrix is constructed by toggling each wall sector **one at a time** and subtracting the unwritten-wall response. Singular-value rank is counted at **1% of the largest singular value**—a declared numerical threshold, not a theorem about channel capacity.

| Measurement apparatus | Queries / shots | Numerical response rank (of 8) | One-patch recognition, Gaussian noise |
| --- | ---: | ---: | ---: |
| One source, signed point trace | 1 | 4 | 55.96% |
| One source, squared point intensity | 1 | 3 | 56.83% |
| One source, five local cells | 1 | 6 | 100% |
| Three spatial source positions, one point | 3 | 8 | 100% |
| Two coherent sources, four relative phases, intensity at one point | 4 | 8 | 100% |

Classification uses a known dictionary of eight **single-patch** states, 300 new noisy trials per state, Gaussian noise with standard deviation 4% of the median single-source response RMS and seed 442. Chance accuracy is 12.5%. A rank of six can still distinguish all eight **discrete** codewords. Extra sensors and extra shots are **not equal resource cost**; the comparison identifies the role of spatial and interrogation diversity, not one winner at equal price. Four-phase readout assumes programmable coherent quadrature sources; it is standard phase-stepping interferometry, not catalytic computation.

The **Williams-like idea that cheap single-bit calibration could simply be summed to decode arbitrary combinations fails badly here**. The separate `python test_multibit.py` uses one-bit calibrations, searches every one of 256 proposed multi-patch memories under an additive model, and scores 12 actual fresh multi-bit wall states. Exact recovery is **0/12 for one source and 0/12 for three sources** (seed 555). Repeated scattering changes the forward operator. A full nonlinear forward model or a richer interaction calibration is required.

## The part explicitly taken from OpenAI's October 25 wave manuscript

The wave paper's introduction derives the effect of imaginary displacement of a **free positive half-wave** along a future-pointing light ray. For frequency `xi`, direction `omega`, positive `rho`,

```
H_rho,omega(xi) = exp[-rho * (|xi| - omega . xi)]
                 = exp[-rho * |xi| * (1 - cos(angle))]
```

Near a chosen direction, this has angular width proportional to **(rho × |xi|)^(-1/2)**. `experiment.py` implements this exact multiplier and verifies both algebraic forms agree to 1.7e-15 in double precision. Measured half-amplitude widths at frequencies 0.7 and 1.4 radians/cell have ratio 0.704, near the high-frequency parabolic-scaling expectation 0.707.

We also take **two complete spatial cavity snapshots** and perform three *virtual*, full-field analysis readouts, each evaluating eight filters at the second focal point: the OpenAI-inspired parabolic directional window, a fixed-angle Gaussian directional window, and radial frequency windows. For this one selected source and these parameters:

| Full-field virtual analysis | Rank of 8 wall-address responses |
| --- | ---: |
| Parabolic, OpenAI-inspired | **8** |
| Fixed-angle directional Gaussian | **8** |
| Radial windows | **7** |

This is **not evidence that the new paper's technique beats older optics**: the fixed-angle control equals it, the comparison is not tuned or held out across geometries, and both directional analyses require access to the **entire two-dimensional field** before filtering. They cannot be produced from a lone scalar receiver; a digital filter cannot recover nullspace information. Nor does the OpenAI free-wave smoothing theorem automatically extend to our bounded, damped, multiply reflecting cavity.

The proof's auxiliary Dirichlet/Monge–Ampère construction has **not** been claimed as a simple spatial Poisson focus meter. We use only a mathematically identified multiplier and its scale dependence, with explicit baselines.

## Williams and Cook–Mertz are a separate question

Ryan Williams's *Simulating Time With Square-Root Space* uses algebraically structured register reuse and bounded-degree polynomial cancellation. Root-of-unity phase cycling in that construction **does not** imply that any multi-scattering cavity has bounded polynomial degree, or that rotating the phase of a single transmitter changes its informational rank. This experiment checks the latter as a null: global source phase changes preserve the rank of one-source observations.

Physically varied sources change the observation map and *can* add independent information. Four relative source phases implement regular interferometry, a different mechanism.

## Reproduce

~~~sh
python -m pip install numpy
python experiment.py
python test_multibit.py
~~~

The two experiment scripts use only NumPy and the Python standard library. For the optional figures, install Matplotlib (`python -m pip install -r requirements.txt`) and run `python plot_receipt.py` after `experiment.py`; it writes `wall_geometry.png`, `address_accuracy.png`, and `echo_trace.png`. Output `receipt.json` is generated locally; no claimed results from physical hardware or external data.

### Next gates (not yet passed)

1. **Resource-matched instrumentation:** compare equal total sample/energy budgets across point trace, five-cell aperture, multi-source pings and phase-cycle detectors, reporting worst-case and average error over noise levels.
2. **Joint wall memory:** measure second- and third-order cross-interactions, calibrate an actual nonlinear forward model and test decoding withheld multi-bit states (not just single altered sectors).
3. **Genuinely physical directional interrogation:** change the *incident wave* at the source and keep the listener fixed. Do not mislabel a full-field Fourier transform as a point-sensor measurement.
4. **OpenAI-specific falsification:** compare frequency-dependent parabolic orientation versus fixed-angle, curvelet/steerable and matched-filter controls on held-out geometries and frequencies; seek an advantage **at equal acquisition and feature cost**.
5. **Boundary and convergence tests:** wave energy for Neumann versus Robin, grid refinement, wall segmentation and repeated-reflection accuracy.

References: [OpenAI wave paper](https://github.com/openai/math/tree/main/preprints/Local-smoothing-for-the-Euclidean-wave-equation-October-25-2026), [ResonantNeuron V12 and V19](https://github.com/anttiluode/ResonantNeuron), [CausalHorizon](https://github.com/anttiluode/CausalHorizon).