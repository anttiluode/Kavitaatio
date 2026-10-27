# Kavitaatio — an etched-wall focus and interrogation laboratory

Research spike, 27 October 2026. This repository is a controlled numerical experiment, **not a proof of Sogge's conjecture**, a physical-neuron demonstration, or a practical SAT/Williams implementation.

The question: **Can an elliptical reflecting cavity retain information as changes in its wall and reveal those changes through coherent, deliberately chosen pings?** And, separately, **does the exact complex-ray directional multiplier appearing in the new OpenAI local-smoothing manuscript help read that information better than ordinary directional filters?**

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