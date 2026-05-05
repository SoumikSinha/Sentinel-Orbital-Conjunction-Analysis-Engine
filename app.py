  # Our SENTINEL Propagation & Collision module (Python 3.13 Compatible)

# Suppress NumPy warnings for Python 3.13
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Step 1: Import libraries (tools)
from skyfield.api import utc, load, EarthSatellite, wgs84
from skyfield.timelib import Time
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid potential issues
import matplotlib.pyplot as plt

# Step 2: Define constants (settings – easy to change)
PREDICTION_HOURS = 24  # 24h ahead
FINE_STEP_MIN = 1  # 1-min steps for analysis
COARSE_STEP_MIN = 10  # 10-min steps for visuals
ALTITUDE_BANDS = [200, 500, 1000]  # Example bands in km (adjust per project)
N_NEAREST = 50  # Check top 50 nearest per object
MIN_DISTANCE_THRESHOLD = 2000.0  # km – if closer, alert

print("SENTINEL Propagation & Collision starting...")

# Global cache for positions (shared across runs)
position_cache = {}

# ------------------ UTC Helper ------------------
def ensure_utc(dt):
    """Ensure datetime is timezone-aware in UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=utc)
    return dt

# ------------------ Satellite and Debris Loading ------------------
def load_satellites(sat_file_path, debris_file_path):
    """Load satellites and debris from TLE files, ensure UTC-aware epochs."""
    satellites = {}
    debris = {}
    
    # Load satellites
    try:
        sat_data = load.tle_file(sat_file_path)
        for sat in sat_data:
            name = str(sat)
            if sat.epoch.utc_datetime().tzinfo is None:
                sat.epoch = sat.epoch.utc_datetime().replace(tzinfo=utc)
            satellites[name] = sat
            print(f"Loaded satellite: {name} | Epoch: {sat.epoch.utc_datetime().isoformat()}")
    except FileNotFoundError:
        print(f"Warning: {sat_file_path} not found. Using empty satellite list.")
    
    # Load debris
    try:
        debris_data = load.tle_file(debris_file_path)
        for obj in debris_data:
            name = str(obj)
            if obj.epoch.utc_datetime().tzinfo is None:
                obj.epoch = obj.epoch.utc_datetime().replace(tzinfo=utc)
            debris[name] = obj
            print(f"Loaded debris: {name} | Epoch: {obj.epoch.utc_datetime().isoformat()}")
    except FileNotFoundError:
        print(f"Warning: {debris_file_path} not found. Using empty debris list.")
    
    return satellites, debris

# ------------------ Propagation ------------------
def propagate_satellite(sat, start_time, hours, step_min, cache_key=None):
    start_time = ensure_utc(start_time)

    if cache_key and cache_key in position_cache:
        print(f"Using cache for {sat.name}")
        return position_cache[cache_key]

    ts = load.timescale()
    end_time = start_time + timedelta(hours=hours)

    datetimes = []
    current = start_time
    while current < end_time:
        datetimes.append(current)
        current += timedelta(minutes=step_min)

    times = ts.from_datetimes(datetimes)
    positions = sat.at(times).position.km.T

    if cache_key:
        position_cache[cache_key] = {'times': times, 'positions': positions}

    return {'times': times, 'positions': positions}

# ------------------ Altitude (placeholder) ------------------
def get_altitude(sat):
    """Rough altitude from mean motion (simplified – use semi-major axis for real)."""
    # Use numpy.random.default_rng() for Python 3.13 compatibility
    rng = np.random.default_rng()
    return rng.uniform(400, 800)  # Replace with real calc

# ------------------ Candidate Filtering ------------------
def filter_candidates(debris, target_sat, n_nearest=N_NEAREST):
    target_alt = get_altitude(target_sat)
    band = next((b for b in ALTITUDE_BANDS if b > target_alt), ALTITUDE_BANDS[-1])
    candidates = []
    seen_names = set()
    for name, obj in debris.items():
        if name in seen_names:
            continue
        seen_names.add(name)
        alt = get_altitude(obj)
        if abs(alt - target_alt) < 100:
            candidates.append(obj)
    
    ts = load.timescale()
    now = ts.from_datetime(datetime.now(utc))
    target_pos = target_sat.at(now).position.km

    def dist_to_target(obj):
        pos = obj.at(now).position.km
        return np.linalg.norm(pos - target_pos)
    
    candidates.sort(key=dist_to_target)
    return candidates[:n_nearest]

# ------------------ Interpolation ------------------
def interpolate_trajectory(traj_fine, times_new):
    """
    Interpolate fine trajectory to new times (e.g., for visuals).
    Converts datetimes to numeric seconds for SciPy.
    """
    time_array = np.array([t.timestamp() for t in traj_fine['times'].utc_datetime()])
    times_new_num = np.array([t.timestamp() for t in times_new])

    pos_interp_x = interp1d(time_array, traj_fine['positions'][:,0], kind='linear')(times_new_num)
    pos_interp_y = interp1d(time_array, traj_fine['positions'][:,1], kind='linear')(times_new_num)
    pos_interp_z = interp1d(time_array, traj_fine['positions'][:,2], kind='linear')(times_new_num)
    return np.column_stack([pos_interp_x, pos_interp_y, pos_interp_z])

def distance_between_trajectories(traj1, traj2):
    """Positions must be same length. Returns array of distances."""
    diff = traj1['positions'] - traj2['positions']
    return np.linalg.norm(diff, axis=1)

def relative_speed_at_time(traj1, traj2, time_idx):
    """Velocity: Position diff / time diff. Approx relative speed."""
    if time_idx == 0 or time_idx == len(traj1['times'])-1:
        return 0
    dt = (traj1['times'][time_idx].utc_datetime() - traj1['times'][time_idx-1].utc_datetime()).total_seconds()
    if dt == 0:
        return 0
    vel1 = (traj1['positions'][time_idx] - traj1['positions'][time_idx-1]) / dt
    vel2 = (traj2['positions'][time_idx] - traj2['positions'][time_idx-1]) / dt
    rel_vel = np.linalg.norm(vel1 - vel2)
    return rel_vel

def find_min_distance_and_tca(traj1, traj2, threshold=MIN_DISTANCE_THRESHOLD):
    """
    Coarse search, then refine around TCA.
    Returns dict: {'min_dist': float, 'tca': datetime, 'rel_speed': float, 'risky': bool}
    """
    dists = distance_between_trajectories(traj1, traj2)
    min_idx_coarse = np.argmin(dists)
    min_dist_coarse = dists[min_idx_coarse]
    
    if min_dist_coarse > threshold:
        return {'min_dist': min_dist_coarse, 'tca': traj1['times'][min_idx_coarse].utc_datetime(), 'rel_speed': 0, 'risky': False}
    
    total_days = traj1['times'][-1] - traj1['times'][0]
    tca_guess = (traj1['times'][min_idx_coarse] - traj1['times'][0]) / total_days
    
    def dist_func(t_fraction):
        t = traj1['times'][0] + t_fraction * total_days
        pos1 = traj1['positions'][0] + t_fraction * (traj1['positions'][-1] - traj1['positions'][0])
        pos2 = traj2['positions'][0] + t_fraction * (traj2['positions'][-1] - traj2['positions'][0])
        return np.linalg.norm(pos1 - pos2)
    
    result = minimize_scalar(dist_func, bounds=(tca_guess-0.01, tca_guess+0.01), method='bounded')
    tca_fraction = result.x
    min_dist = result.fun
    
    tca_time = traj1['times'][0] + tca_fraction * total_days
    tca_dt = tca_time.utc_datetime()
    
    approx_idx = int(tca_fraction * len(traj1['times']))
    rel_speed = relative_speed_at_time(traj1, traj2, max(1, min(approx_idx, len(traj1['times'])-2)))
    
    return {'min_dist': min_dist, 'tca': tca_dt, 'rel_speed': rel_speed, 'risky': min_dist < threshold}

def send_to_api(conjunctions_list):
    """
    Save conjunctions to JSON file. 
    To enable API: set ENABLE_API = True and configure API_ENDPOINT
    """
    ENABLE_API = False  # Set to True when you have an API endpoint
    API_ENDPOINT = 'http://localhost:5000/api/conjunctions'
    
    # Save to JSON file
    output_file = f"conjunctions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(output_file, 'w') as f:
            json.dump({'conjunctions': conjunctions_list}, f, indent=2)
        print(f"✓ Conjunctions saved to {output_file}")
    except Exception as e:
        print(f"✗ Error saving conjunctions: {e}")
        return
    
    # Optional: Send to API if enabled
    if ENABLE_API:
        try:
            data = {'conjunctions': conjunctions_list}
            response = requests.post(API_ENDPOINT, json=data, timeout=10)
            if response.status_code == 200:
                print("✓ Sent to API successfully!")
            else:
                print(f"✗ API Error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"✗ API request failed: {e}")

# ------------------ Visualization ------------------
def plot_orbits(target_sat, candidates, traj_target_coarse, traj_candidates_coarse, start_dt):
    """Plot 2D orbits (x vs y) for target satellite and up to 3 debris objects."""
    plt.figure(figsize=(10, 8))
    
    plt.plot(traj_target_coarse['positions'][:,0], traj_target_coarse['positions'][:,1], 
             label=f"Satellite: {target_sat.name}", linewidth=2, color='blue')
    
    for i, (cand, traj_cand) in enumerate(zip(candidates[:3], traj_candidates_coarse)):
        plt.plot(traj_cand['positions'][:,0], traj_cand['positions'][:,1], 
                 label=f"Debris: {cand.name}", linestyle='--', color=f'C{i+1}')
    
    theta = np.linspace(0, 2*np.pi, 100)
    earth_radius = 6378.137
    plt.plot(earth_radius * np.cos(theta), earth_radius * np.sin(theta), 'k-', label='Earth')
    
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.title(f'Orbits for {target_sat.name} and Debris (2D Projection)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Save instead of show for better compatibility
    output_file = f"orbit_plot_{target_sat.name.replace(' ', '_')}.png"
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()

# ------------------ Main ------------------
def main():
    sat_file = 'active_sats.tle'
    debris_file = 'debris_large.tle'
    satellites, debris = load_satellites(sat_file, debris_file)
    
    if not satellites:
        print("No satellites loaded. Please check your TLE files.")
        return
    
    start_dt = ensure_utc(datetime(2025, 9, 13, 12, 0, 0))
    
    conjunctions = []
    for target_name, target_sat in list(satellites.items())[:5]:
        print(f"\nProcessing target satellite: {target_name}")
        
        candidates = filter_candidates(debris, target_sat)
        print(f"Filtered to {len(candidates)} debris candidates for {target_name}")
        
        if not candidates:
            print(f"No candidates found for {target_name}")
            continue
        
        traj_target_fine = propagate_satellite(target_sat, start_dt, PREDICTION_HOURS, FINE_STEP_MIN, f"{target_name}_fine")
        traj_target_coarse = propagate_satellite(target_sat, start_dt, PREDICTION_HOURS, COARSE_STEP_MIN, f"{target_name}_coarse")
        print(f"Propagated {len(traj_target_coarse['positions'])} coarse points for {target_name}")
        
        traj_candidates_coarse = []
        for cand in candidates[:3]:
            traj_cand_coarse = propagate_satellite(cand, start_dt, PREDICTION_HOURS, COARSE_STEP_MIN, f"{cand.name}_coarse")
            traj_candidates_coarse.append(traj_cand_coarse)
        
        plot_orbits(target_sat, candidates, traj_target_coarse, traj_candidates_coarse, start_dt)
        
        for cand in candidates:
            traj_cand_fine = propagate_satellite(cand, start_dt, PREDICTION_HOURS, FINE_STEP_MIN, f"{cand.name}_fine")
            analysis = find_min_distance_and_tca(traj_target_fine, traj_cand_fine)
            if analysis['risky']:
                conjunctions.append({
                    'sat1': target_name,
                    'sat2': cand.name,
                    'min_dist': analysis['min_dist'],
                    'tca': analysis['tca'].isoformat(),
                    'rel_speed': analysis['rel_speed']
                })
                print(f"Risky: {target_name} vs Debris {cand.name}, Dist: {analysis['min_dist']:.2f} km, "
                      f"TCA: {analysis['tca']}, Speed: {analysis['rel_speed']:.2f} km/s")
    
    if conjunctions:
        print(f"\nFound {len(conjunctions)} risky conjunctions")
        send_to_api(conjunctions)
    else:
        print("\nNo risks found.")

# Run main if file is executed directly
if __name__ == "__main__":
    main()
