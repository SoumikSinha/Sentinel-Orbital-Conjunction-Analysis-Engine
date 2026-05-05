# 🛰️ SENTINEL
### *Orbital Conjunction Analysis Engine — Satellite–Debris Collision Risk Screening System*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Skyfield-SGP4%20Propagation-4B8BBE?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SciPy-Optimization-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  <i>Track everything. Flag what matters. Act before impact.</i>
</p>

---

## 📌 Table of Contents

1. [What is SENTINEL?](#-what-is-sentinel)
2. [Why This Matters](#-why-this-matters)
3. [System Architecture](#-system-architecture)
4. [Background — Orbital Mechanics Primer](#-background--orbital-mechanics-primer)
   - [What is a TLE?](#what-is-a-tle-two-line-element-set)
   - [What is SGP4?](#what-is-sgp4-simplified-general-perturbations-4)
   - [What is a Conjunction?](#what-is-a-conjunction)
5. [Pipeline Deep Dive](#-pipeline-deep-dive)
   - [Stage 1: TLE Loading](#stage-1--tle-loading)
   - [Stage 2: SGP4 Propagation](#stage-2--sgp4-propagation)
   - [Stage 3: Candidate Filtering](#stage-3--candidate-filtering)
   - [Stage 4: Trajectory Interpolation](#stage-4--trajectory-interpolation)
   - [Stage 5: Minimum Distance & TCA Search](#stage-5--minimum-distance--tca-search)
   - [Stage 6: Relative Speed Estimation](#stage-6--relative-speed-estimation)
   - [Stage 7: Risk Flagging & JSON Output](#stage-7--risk-flagging--json-output)
   - [Stage 8: 2D Orbit Visualization](#stage-8--2d-orbit-visualization)
6. [Downloading TLE Datasets](#-downloading-tle-datasets)
7. [Tech Stack](#-tech-stack)
8. [Installation & Setup](#-installation--setup)
9. [Running SENTINEL](#-running-sentinel)
10. [Interpreting Output](#-interpreting-output)
11. [Planned Features](#-planned-features)
    - [AI Threat Scoring Model](#ai-threat-scoring-model)
    - [Interactive 3D Visualizer with Time Slider](#interactive-3d-visualizer-with-time-slider)
12. [Limitations & Known Constraints](#-limitations--known-constraints)
13. [Future Roadmap](#-future-roadmap)
14. [License](#-license)

---

## 🛰️ What is SENTINEL?

**SENTINEL** is an orbital conjunction analysis engine that screens active satellites against catalogued space debris for potential collision risks over a user-defined prediction window (default: 24 hours ahead).

Given a set of satellite and debris **TLE files** (the standard format for describing orbits), SENTINEL:

1. **Propagates** the trajectory of every object forward in time using the SGP4 orbital mechanics model
2. **Filters** the debris population to only the candidates most likely to come close to each target satellite (by altitude proximity, then by spatial distance)
3. **Searches** for the exact **Time of Closest Approach (TCA)** between each satellite–debris pair using a coarse-then-fine two-phase numerical optimization
4. **Flags** any close approach below a configurable distance threshold as a risky conjunction
5. **Outputs** all flagged events to a structured JSON file and optionally to an API endpoint
6. **Plots** 2D orbit projections of the satellite and its nearest debris objects

> Built because space is getting crowded — and a 2000 km warning radius is not as generous as it sounds when objects close at 14 km/s.

---

## 🌍 Why This Matters

| Fact | Figure |
|---|---|
| Tracked objects in Earth orbit | ~27,000+ (as of 2025) |
| Objects large enough to destroy a satellite | ~23,000 |
| Fragments too small to track but lethal | Millions |
| Average relative speed of a collision | ~10–14 km/s |
| Kinetic energy of a 1 kg object at 10 km/s | Equivalent to ~22 kg of TNT |
| Cost of replacing a communications satellite | $150M–$400M |

The **Kessler Syndrome** — a cascade of collisions generating more debris, triggering more collisions — is a realistic existential threat to the orbital environment we depend on for GPS, weather forecasting, internet, and military communication. SENTINEL is a step toward the automated, high-cadence conjunction screening that preventing this scenario requires.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              INPUT: TLE Files (Satellites + Debris)             │
│           active_sats.tle         debris_large.tle              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
         ┌─────────────▼──────────────┐
         │      TLE Parser / Loader    │
         │  (Skyfield EarthSatellite)  │
         └─────────────┬──────────────┘
                       │
         ┌─────────────▼──────────────────────────────────┐
         │  For each target satellite:                      │
         │                                                  │
         │  ┌────────────────────────────────────────────┐ │
         │  │ Stage 1: Candidate Filtering               │ │
         │  │ (Altitude band → Top-N nearest by distance)│ │
         │  └──────────────────┬─────────────────────────┘ │
         │                     │                            │
         │  ┌──────────────────▼─────────────────────────┐ │
         │  │ Stage 2: SGP4 Propagation                  │ │
         │  │ (Fine 1-min steps + Coarse 10-min steps)   │ │
         │  └──────────────────┬─────────────────────────┘ │
         │                     │                            │
         │  ┌──────────────────▼─────────────────────────┐ │
         │  │ Stage 3: Trajectory Interpolation           │ │
         │  │ (scipy interp1d for smooth position arrays) │ │
         │  └──────────────────┬─────────────────────────┘ │
         │                     │                            │
         │  ┌──────────────────▼─────────────────────────┐ │
         │  │ Stage 4: Min Distance + TCA Search          │ │
         │  │ (Coarse argmin → Bounded minimize_scalar)   │ │
         │  └──────────────────┬─────────────────────────┘ │
         │                     │                            │
         │  ┌──────────────────▼─────────────────────────┐ │
         │  │ Stage 5: Risk Flagging (threshold: 2000 km) │ │
         │  └──────────────────┬─────────────────────────┘ │
         └─────────────────────┼──────────────────────────-┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
    │  JSON Output   │  │ API Endpoint │  │  2D Orbit   │
    │ (conjunctions  │  │  (optional) │  │    Plot     │
    │   _YYYYMMDD    │  │             │  │  (PNG file) │
    │   _HHMMSS.json)│  └─────────────┘  └─────────────┘
    └────────────────┘
```

---

## 📡 Background — Orbital Mechanics Primer

Before the pipeline, it helps to understand the three foundational concepts the entire system is built on.

---

### What is a TLE (Two-Line Element Set)?

A **TLE** is the standard data format used by NASA, ESA, NORAD, and every major space agency to describe the orbit of a satellite or debris object. It encodes the object's orbital parameters at a specific point in time (called the **epoch**) in exactly two 80-character lines.

Here is a real example — the International Space Station:

```
ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00001234  00000-0  28765-4 0  9991
2 25544  51.6416 247.4627 0006703  25.4876 334.6511 15.49407864433165
```

**Line 1:**

| Field | Value (example) | Meaning |
|---|---|---|
| Column 1 | `1` | Line number |
| Columns 3–7 | `25544` | **NORAD Catalog Number** — unique ID assigned to every tracked object |
| Column 8 | `U` | **Classification** — U = Unclassified, C = Classified, S = Secret |
| Columns 10–17 | `98067A` | **International Designator** — year of launch (98=1998), launch number (067), piece (A=primary) |
| Columns 19–32 | `24001.50000000` | **Epoch** — the time at which these orbital elements are valid. Format: `YYDDD.DDDDDDDD` — last two digits of year (24=2024), followed by day-of-year with fractional day (001.5 = Jan 1st, noon) |
| Columns 34–43 | `.00001234` | **First derivative of mean motion** (Ballistic Coefficient / 2) — how fast the orbit is decaying due to atmospheric drag, in revolutions per day² |
| Columns 45–52 | `00000-0` | **Second derivative of mean motion** — used for higher-order drag modeling (usually zero) |
| Columns 54–61 | `28765-4` | **BSTAR drag term** — describes atmospheric drag in the SGP4 model. Format: `mantissa × 10^exponent` |
| Column 63 | `0` | **Ephemeris type** — 0 = SGP4/SDP4 |
| Columns 65–68 | `999` | **Element set number** — how many times this TLE has been updated |
| Column 69 | `1` | **Checksum** — sum of all digits mod 10, for error detection |

**Line 2:**

| Field | Value (example) | Meaning |
|---|---|---|
| Column 1 | `2` | Line number |
| Columns 3–7 | `25544` | NORAD Catalog Number (repeated) |
| Columns 9–16 | `51.6416` | **Inclination (i)** in degrees — the tilt of the orbital plane relative to Earth's equator. 0° = equatorial orbit, 90° = polar orbit |
| Columns 18–25 | `247.4627` | **Right Ascension of Ascending Node (RAAN, Ω)** in degrees — the angle in the equatorial plane between the vernal equinox direction and where the satellite crosses the equator going northward. Defines the orbital plane's orientation in space |
| Columns 27–33 | `0006703` | **Eccentricity (e)** — implied decimal: `0.0006703`. Describes the shape of the orbit: 0 = perfect circle, 1 = parabolic escape trajectory |
| Columns 35–42 | `25.4876` | **Argument of Perigee (ω)** in degrees — the angle within the orbital plane from the ascending node to the closest point to Earth (perigee). Defines where in the orbit the object is closest to Earth |
| Columns 44–51 | `334.6511` | **Mean Anomaly (M)** in degrees — the object's position within its orbit at the epoch, expressed as an angle. 0° = perigee, 180° = apogee. "Mean" because it assumes uniform angular motion and must be converted to true position |
| Columns 53–63 | `15.49407864` | **Mean Motion (n)** in revolutions per day — how many complete orbits the object makes per day. ISS: ~15.5 rev/day ≈ 92-minute orbit |
| Columns 64–68 | `43316` | **Revolution number at epoch** — total orbits completed since launch |
| Column 69 | `5` | **Checksum** |

---

### What is SGP4 (Simplified General Perturbations 4)?

**SGP4** is the standard mathematical model used to propagate (predict forward in time) an object's position and velocity given its TLE. The name reflects its lineage: it is the 4th generation of the "Simplified General Perturbations" family of orbit propagators.

**What "propagation" means:** Given an orbit description at time `t₀` (the TLE epoch), compute the object's 3D position and velocity at any future (or past) time `t`.

SGP4 is not a simple equation — it is a set of analytical approximations that account for the main forces perturbing a low-Earth orbit:

| Perturbation | Cause | Effect |
|---|---|---|
| **Keplerian motion** | Earth's gravity (point mass) | Base elliptical orbit |
| **J2 oblateness** | Earth is not a perfect sphere — it bulges at the equator | Causes RAAN and argument of perigee to slowly drift over time |
| **J3, J4 terms** | Higher-order Earth gravity harmonics | Smaller perturbations to the orbit shape |
| **Atmospheric drag** | Thin upper atmosphere (even at 400 km altitude) | Gradually lowers the orbit; captured by BSTAR term |
| **Solar radiation pressure** | Photons from the Sun pushing on the satellite's surface | Small but cumulative effect on high-area/mass-ratio objects |

SGP4 is **analytical** (not numerical integration) — it evaluates closed-form equations rather than integrating Newton's laws step by step. This makes it extremely fast: millions of propagations per second. Accuracy is typically ~1 km for a few days, degrading with TLE age.

SENTINEL uses **Skyfield's `EarthSatellite`** class, which implements SGP4 internally. The output is a position vector in the **TEME (True Equator Mean Equinox)** coordinate frame — a geocentric Cartesian frame (Earth's center is the origin) measured in kilometers along three orthogonal axes X, Y, Z.

---

### What is a Conjunction?

A **conjunction** in orbital mechanics is a close approach between two orbiting objects. It does not necessarily mean a collision — but if the miss distance is small enough, it is treated as a **collision risk event** requiring attention.

Monitoring agencies use a concept called **CDM (Conjunction Data Message)** — a standardized format for reporting close approaches. Commercial and governmental satellite operators receive CDMs and decide whether to perform an **avoidance maneuver**.

SENTINEL's threshold is `2,000 km` — any approach within this range triggers a risk flag. This is conservative; operational conjunction screens typically use much smaller thresholds (~5–10 km for high-confidence events), but the wider net ensures no candidate is missed at the screening stage.

---

## 🔬 Pipeline Deep Dive

---

### Stage 1 — TLE Loading

```python
satellites, debris = load_satellites(sat_file_path, debris_file_path)
```

Skyfield's `load.tle_file()` reads the TLE text file and creates `EarthSatellite` objects — in-memory representations of each orbit, with the SGP4 propagator ready to use.

Each object has an **epoch** — the datetime at which its TLE was accurate. Skyfield handles epoch parsing internally, but the code explicitly enforces **UTC timezone-awareness** on every epoch:

```python
if sat.epoch.utc_datetime().tzinfo is None:
    sat.epoch = sat.epoch.utc_datetime().replace(tzinfo=utc)
```

**Why UTC matters:** Python's `datetime` objects can be "naive" (no timezone) or "aware" (explicit timezone). Mixing naive and aware datetimes causes crashes. All time arithmetic in SENTINEL uses UTC-aware datetimes throughout.

A **global `position_cache` dictionary** stores propagation results keyed by object name + resolution (e.g., `"ISS_fine"`). Repeat propagation requests for the same object return the cached result instantly, avoiding redundant SGP4 computation.

---

### Stage 2 — SGP4 Propagation

```python
traj = propagate_satellite(sat, start_time, hours=24, step_min=1)
```

This is the core computation. For a target satellite, the function:

**Step 1 — Build a list of datetimes:**
```
t₀, t₀ + 1 min, t₀ + 2 min, ..., t₀ + 24 hours
```
For a 24-hour window at 1-minute steps: **1,440 time points**.

**Step 2 — Convert to Skyfield `Time` objects:**
Skyfield uses its own `Time` type (internally, Julian Date) for all propagation. The list of Python `datetime` objects is batch-converted via `ts.from_datetimes()`.

**Step 3 — Propagate using SGP4:**
```python
positions = sat.at(times).position.km.T
```

- `sat.at(times)` calls the SGP4 model for all 1,440 time points simultaneously (vectorized)
- `.position.km` returns a `(3, N)` array — three rows (X, Y, Z coordinates) across N time points
- `.T` transposes it to `(N, 3)` — N rows, each row being one `[X, Y, Z]` position vector in km

**Two resolution levels are propagated:**

| Resolution | Step | Points (24h) | Purpose |
|---|---|---|---|
| **Fine** | 1 minute | 1,440 | Accurate TCA search and distance computation |
| **Coarse** | 10 minutes | 144 | Fast 2D orbit plot visualization |

---

### Stage 3 — Candidate Filtering

With potentially tens of thousands of debris objects in the catalogue, computing a 24-hour trajectory for all of them against every satellite would be computationally prohibitive. Candidate filtering reduces the debris population to a manageable `N_NEAREST = 50` objects per satellite using a two-stage filter:

**Filter A — Altitude Band:**

```python
target_alt = get_altitude(target_sat)
band = next((b for b in ALTITUDE_BANDS if b > target_alt), ALTITUDE_BANDS[-1])
# Keep only debris within ±100 km altitude of the target
if abs(alt - target_alt) < 100:
    candidates.append(obj)
```

Orbital altitude is approximately related to mean motion (TLE Line 2, revolutions/day) via Kepler's Third Law. Objects in dramatically different altitude bands will never come close — they orbit at different speeds and the geometry makes close approach nearly impossible without a massive delta-V event. Filtering by altitude is a fast, physically motivated pre-screen.

**Filter B — Current Spatial Distance:**

Among the altitude-matched debris, compute the 3D distance from the satellite's position *right now* and keep only the 50 nearest:

```python
target_pos = target_sat.at(now).position.km

def dist_to_target(obj):
    pos = obj.at(now).position.km
    return np.linalg.norm(pos - target_pos)

candidates.sort(key=dist_to_target)
return candidates[:n_nearest]
```

**`np.linalg.norm`** computes the **Euclidean norm** (straight-line distance) of a vector. For a position difference vector `Δ = [Δx, Δy, Δz]`:

```
||Δ|| = √(Δx² + Δy² + Δz²)
```

| Symbol | Meaning |
|---|---|
| `||Δ||` | Euclidean norm — the straight-line distance between two points in 3D space |
| `Δx = x₁ − x₂` | Difference in X-coordinates (km) between the satellite and debris positions |
| `Δy = y₁ − y₂` | Difference in Y-coordinates (km) |
| `Δz = z₁ − z₂` | Difference in Z-coordinates (km) |
| `√(...)` | Square root |

The sorted, trimmed candidate list is used for all further per-pair analysis.

---

### Stage 4 — Trajectory Interpolation

```python
pos_interp = interpolate_trajectory(traj_fine, times_new)
```

`scipy.interpolate.interp1d` performs **piecewise linear interpolation** — given a set of known (time, position) data points, it estimates the position at any new time that falls within the range of the known data.

**Why is this needed?**

The fine trajectories of two objects are propagated over the same time grid (every 1 minute). However, when doing the two-phase TCA search (see Stage 5), the optimizer evaluates the distance function at arbitrary fractional time points that don't fall exactly on the 1-minute grid. Interpolation allows smooth distance evaluation at any point within the 24-hour window.

**How interp1d works:**

The time axis is first converted from Skyfield `Time` objects to **Unix timestamps** — numeric seconds since January 1, 1970 UTC:

```python
time_array = np.array([t.timestamp() for t in traj_fine['times'].utc_datetime()])
```

This gives a 1D numeric array of 1,440 values (one per minute). For each spatial axis (X, Y, Z separately), `interp1d` builds a piecewise linear function:

```
For times tₖ and tₖ₊₁ with positions pₖ and pₖ₊₁:

p(t) = pₖ + (t − tₖ) / (tₖ₊₁ − tₖ) × (pₖ₊₁ − pₖ)
```

| Symbol | Meaning |
|---|---|
| `t` | The new time at which we want the position (Unix timestamp) |
| `tₖ` | The known time just before `t` (the left bracket of the interval) |
| `tₖ₊₁` | The known time just after `t` (the right bracket) |
| `pₖ` | The known position at `tₖ` |
| `pₖ₊₁` | The known position at `tₖ₊₁` |
| `(t − tₖ) / (tₖ₊₁ − tₖ)` | The **fractional progress** through the interval — a number between 0 and 1 |
| Result `p(t)` | The linearly interpolated position estimate at time `t` |

This is done independently for X, Y, and Z, then recombined into a `(N, 3)` position array.

---

### Stage 5 — Minimum Distance & TCA Search

This is the mathematical core of SENTINEL. For each satellite–debris pair, finding the exact **Time of Closest Approach (TCA)** and the **minimum miss distance** is a continuous optimization problem.

**What is TCA?**

TCA is the specific moment in time when two orbiting objects are at their closest point. Before TCA, they are approaching. After TCA, they are receding. The distance at TCA is the **minimum miss distance** — the most critical number in conjunction analysis.

**Two-Phase Approach:**

**Phase 1 — Coarse Search (argmin over discrete trajectory):**

```python
dists = distance_between_trajectories(traj1, traj2)
min_idx_coarse = np.argmin(dists)
min_dist_coarse = dists[min_idx_coarse]
```

`distance_between_trajectories` computes the elementwise Euclidean distance between the two trajectories at every time step:

```
dists[k] = ||pos1[k] − pos2[k]||  for k = 0, 1, 2, ..., 1439
```

| Symbol | Meaning |
|---|---|
| `dists[k]` | Distance between satellite and debris at time step `k` (km) |
| `pos1[k]` | 3D position vector `[x, y, z]` of the satellite at time step `k` |
| `pos2[k]` | 3D position vector `[x, y, z]` of the debris at time step `k` |
| `||...||` | Euclidean norm — straight-line distance in 3D |
| `np.argmin(dists)` | Index `k` at which the distance array is smallest — the coarse TCA estimate |

If this minimum distance is already above the threshold (2,000 km), the pair is immediately discarded — no risky conjunction possible, skip the expensive refinement.

**Phase 2 — Fine Refinement (bounded scalar minimization):**

The coarse argmin gives the closest 1-minute interval, but the true TCA could be at any fraction of a second within that interval. `scipy.optimize.minimize_scalar` finds the exact minimum of the distance function within a narrow window around the coarse estimate.

The time axis is parameterized as a **fractional value between 0 and 1** (0 = start of 24-hour window, 1 = end):

```python
tca_guess = (times[min_idx_coarse] − times[0]) / total_days

def dist_func(t_fraction):
    pos1 = pos1_start + t_fraction × (pos1_end − pos1_start)
    pos2 = pos2_start + t_fraction × (pos2_end − pos2_start)
    return ||pos1 − pos2||

result = minimize_scalar(dist_func,
                         bounds=(tca_guess − 0.01, tca_guess + 0.01),
                         method='bounded')
```

**Every symbol defined:**

| Symbol | Meaning |
|---|---|
| `t_fraction` | Normalized time — 0.0 = start of prediction window, 1.0 = end. The optimizer searches for the value that minimizes distance |
| `tca_guess` | The fractional time of the coarse minimum — the starting point for the fine search |
| `bounds=(tca_guess − 0.01, tca_guess + 0.01)` | The optimizer is constrained to a ±0.01 window around the coarse estimate (±0.01 of the 24-hour window = ±14.4 minutes) |
| `method='bounded'` | Brent's method with bounds — a fast, derivative-free 1D minimizer that brackets the minimum and successively narrows the interval using golden-section search and parabolic interpolation |
| `result.x` | The fractional time at which the minimum distance occurs — the refined TCA as a fraction |
| `result.fun` | The minimum distance value found by the optimizer (km) |
| `pos1_start + t_fraction × (pos1_end − pos1_start)` | Linear interpolation of position at fractional time `t_fraction` |

The TCA fraction is then converted back to an absolute datetime:
```python
tca_time = times[0] + tca_fraction × total_days
```

---

### Stage 6 — Relative Speed Estimation

At the TCA, the **relative speed** between the satellite and debris tells you how fast they are closing (or passing). This is critical for risk assessment: a slow-moving approach gives more warning time; a hypervelocity pass gives milliseconds.

Relative speed is estimated using **finite differences** — approximating velocity from the change in position across two consecutive time steps:

```
velocity₁[k] = (pos1[k] − pos1[k−1]) / Δt

velocity₂[k] = (pos2[k] − pos2[k−1]) / Δt

relative_speed = ||velocity₁[k] − velocity₂[k]||
```

**Every symbol defined:**

| Symbol | Meaning |
|---|---|
| `velocity₁[k]` | Approximate velocity vector of object 1 at time step `k` (km/s), derived from the position difference divided by the time difference |
| `pos1[k]` | 3D position of object 1 at step `k` (km) |
| `pos1[k−1]` | 3D position of object 1 at the previous step (km) |
| `Δt` | Time difference between steps `k` and `k−1`, in seconds |
| `velocity₁[k] − velocity₂[k]` | The **relative velocity vector** — how fast object 2 is moving relative to object 1, in 3D |
| `||...||` | Euclidean norm of the relative velocity vector — gives the **relative speed** as a scalar (km/s) |

For context: typical relative speeds in LEO range from near-zero (for co-planar, co-altitude objects) to ~14 km/s (for a head-on polar vs. equatorial orbit collision).

---

### Stage 7 — Risk Flagging & JSON Output

Any satellite–debris pair where `min_dist < MIN_DISTANCE_THRESHOLD (2000 km)` is flagged as a risky conjunction and stored:

```json
{
  "sat1": "ISS (ZARYA)",
  "sat2": "COSMOS 2251 DEB",
  "min_dist": 847.33,
  "tca": "2025-09-13T18:24:11+00:00",
  "rel_speed": 11.72
}
```

| Field | Meaning |
|---|---|
| `sat1` | Name of the active satellite being screened |
| `sat2` | Name of the debris object that poses a risk |
| `min_dist` | Minimum miss distance at TCA, in km |
| `tca` | Time of Closest Approach as an ISO 8601 UTC datetime string |
| `rel_speed` | Relative speed at TCA, in km/s |

All flagged conjunctions are written to a timestamped JSON file: `conjunctions_YYYYMMDD_HHMMSS.json`. An optional REST API endpoint is available (disabled by default via `ENABLE_API = False`).

---

### Stage 8 — 2D Orbit Visualization

For each target satellite, a 2D orbit plot is generated showing the X-Y projection of the trajectory (coarse, 10-minute resolution) for the satellite and up to 3 of its nearest debris candidates. An Earth circle (radius = 6,378.137 km) is drawn at the origin.

The plot is saved as a PNG file: `orbit_plot_{satellite_name}.png`.

> **Note:** This is an X-Y plane projection only — it is a simplified visualization. The full 3D orbital geometry would require a 3D rendering. A full interactive 3D visualizer is planned (see Planned Features).

---

## 🌐 Downloading TLE Datasets

SENTINEL requires two TLE files in the working directory:
- `active_sats.tle` — active, operational satellites
- `debris_large.tle` — catalogued space debris objects

### Option 1 — CelesTrak (Recommended, Free)

CelesTrak (maintained by Dr. T.S. Kelso) is the most widely used public source for TLE data. It provides pre-categorized TLE files updated multiple times daily.

**Step 1 — Visit CelesTrak's catalogue page:**
```
https://celestrak.org/SOCRATES/query.php
```
or the direct GP data portal:
```
https://celestrak.org/GPS/
```

**Step 2 — Download Active Satellites:**

Navigate to: `https://celestrak.org/SOCRATES/`

Or directly download categorized TLE files:

```bash
# Active satellites (active_sats.tle)
curl -o active_sats.tle "https://celestrak.org/SOCRATES/query.php?CATALOG=active&OTYPE=TLE&MAX=10&ORDERBY=MAXPROB&SORT=MaxProbability&Submit=Submit%22"
```

Or manually from the catalogue pages:
```
https://celestrak.org/satcat/tle.php?STATUSCODE=U
```
Save the downloaded `.txt` file as `active_sats.tle` in the SENTINEL directory.

**Step 3 — Download Debris:**

CelesTrak provides curated debris catalogues:
```
https://celestrak.org/SOCRATES/query.php?CATALOG=analyst&OTYPE=TLE
```

Or the full debris set directly:
```bash
# Large debris objects (debris_large.tle)
curl -o debris_large.tle "https://celestrak.org/pub/TLE/catalog.txt"
```

**Quick download links (copy-paste into browser to download directly):**

| Dataset | URL |
|---|---|
| Active Satellites | `https://celestrak.org/SOCRATES/query.php?CATALOG=active&OTYPE=TLE` |
| Full TLE Catalog | `https://celestrak.org/pub/TLE/catalog.txt` |
| Last 30 days launches | `https://celestrak.org/SOCRATES/query.php?CATALOG=tle-new&OTYPE=TLE` |
| Iridium constellation | `https://celestrak.org/SOCRATES/query.php?CATALOG=iridium-33-debris&OTYPE=TLE` |

---

### Option 2 — Space-Track.org (Official US Government Source)

Space-Track is the official source, operated by US Space Command (USSPACECOM). It requires **free registration** but provides the most complete and up-to-date catalogue.

**Step 1 — Register:**
```
https://www.space-track.org/auth/createAccount
```

**Step 2 — Login and navigate to:** `https://www.space-track.org/basicspacedata/`

**Step 3 — Download via API (after login):**
```bash
# Active satellites
curl --cookie-jar cookies.txt \
     --data "identity=YOUR_EMAIL&password=YOUR_PASSWORD" \
     "https://www.space-track.org/ajaxauth/login"

curl --cookie cookies.txt \
     "https://www.space-track.org/basicspacedata/query/class/tle_latest/ORDINAL/1/OBJECT_TYPE/PAYLOAD/format/tle" \
     -o active_sats.tle

curl --cookie cookies.txt \
     "https://www.space-track.org/basicspacedata/query/class/tle_latest/ORDINAL/1/OBJECT_TYPE/DEBRIS/format/tle" \
     -o debris_large.tle
```

---

### File Placement

Once downloaded, place both files in the **same directory as `app.py`**:

```
sentinel/
├── app.py
├── active_sats.tle       ← put here
├── debris_large.tle      ← put here
├── requirements.txt
└── README.md
```

SENTINEL will load them automatically when `main()` runs.

---

> **⚠️ TLE Freshness Warning:** TLE accuracy degrades with age. SGP4 is typically accurate to ~1 km for fresh TLEs (< 1 day old) and can degrade to tens of km for week-old TLEs, especially for low-altitude objects experiencing atmospheric drag. Always use recently downloaded TLE files for analysis.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Orbital Propagation | Skyfield (`EarthSatellite`, SGP4) |
| Numerical Computing | NumPy |
| Interpolation | SciPy `interp1d` |
| Optimization (TCA) | SciPy `minimize_scalar` |
| Visualization | Matplotlib |
| Data Output | JSON, Pandas |
| HTTP (optional API) | Requests |

---

## 📦 Installation & Setup

### Prerequisites

- Python **3.9 or higher** (tested up to Python 3.13)
- `pip`
- Internet connection (for downloading TLE files)

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Sentinel-Orbital-Conjunction-Analysis-Engine.git
cd Sentinel-Orbital-Conjunction-Analysis-Engine
```

---

### Step 2 — Create a Virtual Environment

A **virtual environment** isolates this project's dependencies from the rest of your Python installation, preventing version conflicts.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt.

---

### Step 3 — Install Dependencies

```bash
pip install skyfield numpy scipy matplotlib pandas requests
```

Or with a `requirements.txt`:
```bash
pip install -r requirements.txt
```

<details>
<summary>📋 Full dependency list (click to expand)</summary>

```
skyfield>=1.48
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
pandas>=2.0.0
requests>=2.31.0
```

</details>

---

### Step 4 — Download TLE Files

Follow the [Downloading TLE Datasets](#-downloading-tle-datasets) section above and place both `active_sats.tle` and `debris_large.tle` in the project directory.

---

### Step 5 — Verify Setup

```bash
python -c "from skyfield.api import load; import numpy, scipy, matplotlib; print('All OK')"
```

---

## ▶️ Running SENTINEL

With the virtual environment activated and TLE files in place:

```bash
python app.py
```

SENTINEL will:
1. Print loading confirmation for each satellite and debris object
2. For each of the first 5 satellites, print the number of filtered debris candidates
3. Propagate trajectories and search for conjunctions
4. Print any risky conjunctions found to the terminal
5. Save results to `conjunctions_YYYYMMDD_HHMMSS.json`
6. Save orbit plots to `orbit_plot_{name}.png` files

**To change the analysis date**, edit this line in `main()`:
```python
start_dt = ensure_utc(datetime(2025, 9, 13, 12, 0, 0))
```

**To screen more satellites**, change:
```python
for target_name, target_sat in list(satellites.items())[:5]:  # Change 5 to any number
```

**To adjust the collision risk threshold**, change:
```python
MIN_DISTANCE_THRESHOLD = 2000.0  # km — lower for fewer alerts, higher for more alerts
```

---

## 📊 Interpreting Output

### Terminal Output

```
SENTINEL Propagation & Collision starting...
Loaded satellite: ISS (ZARYA) | Epoch: 2025-09-12T23:59:59+00:00
...
Processing target satellite: ISS (ZARYA)
Filtered to 12 debris candidates for ISS (ZARYA)
Propagated 144 coarse points for ISS (ZARYA)
Risky: ISS (ZARYA) vs Debris COSMOS 1408 DEB, Dist: 1243.67 km, TCA: 2025-09-13 17:43:22+00:00, Speed: 13.41 km/s
```

### JSON Output

```json
{
  "conjunctions": [
    {
      "sat1": "ISS (ZARYA)",
      "sat2": "COSMOS 1408 DEB",
      "min_dist": 1243.67,
      "tca": "2025-09-13T17:43:22+00:00",
      "rel_speed": 13.41
    }
  ]
}
```

### Risk Interpretation Guide

| min_dist | rel_speed | Interpretation |
|---|---|---|
| < 100 km | > 5 km/s | **Critical** — High-priority conjunction, immediate attention required |
| 100–500 km | any | **High** — Well within screening threshold, monitor closely |
| 500–1000 km | > 10 km/s | **Moderate** — Flagged, probability of collision still very low |
| 1000–2000 km | any | **Low** — Within screening net; almost certainly no collision |
| > 2000 km | — | Not flagged — passes the initial filter |

---

## 🚀 Planned Features

### AI Threat Scoring Model

Currently, SENTINEL flags conjunctions based purely on miss distance (< 2,000 km). This is a **necessary condition** for concern but not sufficient on its own — the actual probability of collision depends on many factors that the simple distance threshold ignores.

The planned **AI Threat Scoring Module** would take a flagged conjunction event and output a **percentage probability of collision** and a **risk severity score** by learning from historical conjunction data and mission parameters.

**Planned input features:**

| Feature | Description |
|---|---|
| `min_dist` | Minimum miss distance at TCA (km) |
| `rel_speed` | Relative speed at TCA (km/s) |
| `time_to_tca` | Hours until TCA from now |
| `sat_mass_estimate` | Estimated mass of the satellite (kg) |
| `debris_rcs` | Radar Cross Section of the debris — a proxy for its physical size (m²) |
| `sat_altitude` | Orbital altitude — affects atmospheric density and drag uncertainty |
| `inclination_diff` | Difference in orbital inclination between the two objects |
| `tca_uncertainty` | TLE age in days — older TLEs = larger position uncertainty |
| `covariance_overlap` | Volume of intersection of the two positional uncertainty ellipsoids |

**Planned output:**
```json
{
  "collision_probability_percent": 0.0034,
  "risk_level": "LOW",
  "recommended_action": "Monitor — No maneuver required at current Pc",
  "maneuver_window": "2025-09-13T14:00:00Z to 2025-09-13T16:00:00Z"
}
```

The model would be trained on publicly available historical conjunction data from Space-Track's CDM archive, using a gradient-boosted classifier or a feedforward neural network (given the tabular nature of the input features).

---

### Interactive 3D Visualizer with Time Slider

The current matplotlib 2D orbit plot is a static image — it cannot communicate the *dynamic nature* of a conjunction event. To make the threat tangible, a **web-based interactive 3D visualizer** is planned.

**Concept:**

The conjunction JSON output is fed to a visualization frontend that renders the satellite and debris trajectories in 3D and animates their positions over the 24-hour prediction window.

```
SENTINEL JSON Output  →  Visualizer API  →  3D Web Frontend
                                              (Three.js / CesiumJS)
```

**Planned features of the visualizer:**

- **3D Earth** at the origin, with realistic surface texture
- **Live trajectory paths** for the flagged satellite and debris object — shown as full 24-hour orbital arcs
- **Animated object positions** — the satellite and debris represented as 3D markers that move along their respective arcs
- **Time slider** — drag to any point in the 24-hour window; both objects update their positions in real time. The moment of TCA is visually marked
- **Closest approach highlight** — when the time slider reaches TCA, the scene zooms in, the miss distance is displayed in km, and the approach vector is drawn
- **Multi-conjunction view** — if multiple debris objects threaten the same satellite, all are shown simultaneously on the same scene
- **Collision projection** — if no avoidance maneuver is taken, the system extrapolates and animates what the collision would look like (for demonstration and public communication purposes)

**Data flow:**

```json
// SENTINEL outputs per conjunction event:
{
  "sat1": "ISS",
  "sat2": "COSMOS DEB",
  "tca": "2025-09-13T17:43:22Z",
  "min_dist": 1243.67,
  "trajectory_sat1": [[x0,y0,z0,t0], [x1,y1,z1,t1], ...],  // ← 1440 points
  "trajectory_sat2": [[x0,y0,z0,t0], [x1,y1,z1,t1], ...]
}
```

The trajectory arrays (position + timestamp for each of 1,440 minutes) give the visualizer everything it needs to animate the full 24-hour approach sequence without any further computation.

---

## ⚠️ Limitations & Known Constraints

- **`get_altitude()` is a placeholder**: The current implementation returns a random altitude between 400–800 km. In production, altitude should be derived from the TLE's mean motion using Kepler's Third Law: `a = (μ/n²)^(1/3) − Rₑ`, where `μ` is Earth's gravitational parameter, `n` is mean motion in rad/s, and `Rₑ` is Earth's radius.
- **Linear position interpolation in TCA finder**: The `dist_func` inside `minimize_scalar` uses linear interpolation between trajectory endpoints rather than re-running SGP4. This introduces small errors near TCA for highly curved orbits. Future fix: re-propagate at refined timesteps.
- **Covariance not used**: True collision probability requires **positional uncertainty ellipsoids** (covariance matrices from TLE fitting). SENTINEL uses only the nominal position — giving miss distance, not Pc (probability of collision).
- **No maneuver modeling**: Satellites that have already executed avoidance maneuvers will have inaccurate TLEs until a new TLE is generated. SENTINEL cannot account for recent untracked maneuvers.
- **TLE age sensitivity**: For objects at 300–400 km altitude where atmospheric drag is significant, a TLE more than 24–48 hours old can accumulate position errors of tens of km, making fine TCA estimates unreliable.

---

## 🔮 Future Roadmap

- [ ] Replace placeholder `get_altitude()` with Kepler's Third Law computation
- [ ] Implement positional covariance propagation for true Pc (probability of collision) computation
- [ ] Integrate the planned AI threat scoring model
- [ ] Build the interactive 3D visualizer with time slider (Three.js / CesiumJS frontend)
- [ ] Add Streamlit dashboard for real-time screening dashboard (live TLE download + auto-refresh)
- [ ] Support **SGP4-XP** (extended precision) for deep-space objects (high eccentricity, long periods)
- [ ] Export CDM-format output for compatibility with CCSDS mission operations standards
- [ ] Batch API mode to screen entire active catalogue nightly

---

## 👤 Author

**Soumik Sinha**  
B.Tech CSE, PES University, Bengaluru  
[GitHub](https://github.com/SoumikSinha) · [LinkedIn](https://linkedin.com/in/soumik-sinha-928a21352)

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>The universe is under no obligation to warn us. SENTINEL is.</i><br>
  <b>🛰️ SENTINEL</b>
</p>
