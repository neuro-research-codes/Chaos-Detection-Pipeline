import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pynwb import NWBHDF5IO
from pathlib import Path

# === Config ===
nwb_root = "/path-to-data-nwb-files/"
output_npz_path = "/path-to-folder/task_irrelevant_isi_trains_filtered.npz"
debug_log_path = "/path-to-folder/isi_debug_log_task_irrelevant.csv"
min_isi_len = 20

def compute_isis(spike_times):
    return np.diff(np.sort(spike_times))

def compute_non_task_intervals(trials, total_duration):
    trials_sorted = trials.sort_values("start_time")
    gaps = []
    prev_end = 0.0
    for _, row in trials_sorted.iterrows():
        if row["start_time"] > prev_end:
            gaps.append((prev_end, row["start_time"]))
        prev_end = max(prev_end, row["stop_time"])
    if prev_end < total_duration:
        gaps.append((prev_end, total_duration))
    return gaps

isi_dict = {}
debug_log = []
nwb_files = list(Path(nwb_root).rglob("*.nwb"))

print(f"📁 Found {len(nwb_files)} NWB files")

for nwb_path in tqdm(nwb_files, desc="🔍 Processing"):
    session = nwb_path.stem
    try:
        with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
            nwb = io.read()
            units_df = nwb.units.to_dataframe()
            trials_df = nwb.intervals["trials"].to_dataframe()

            if "stim_phase" not in trials_df.columns:
                continue

            task_trials = trials_df[trials_df["stim_phase"].str.lower().isin(["learn", "recog"])]
            total_duration = nwb.acquisition["events"].timestamps[-1]
            non_task_intervals = compute_non_task_intervals(task_trials, total_duration)

            for unit_id in units_df.index:
                spikes = np.array(units_df.at[unit_id, "spike_times"])
                isi_segments = []
                for start, stop in non_task_intervals:
                    segment = spikes[(spikes >= start) & (spikes < stop)]
                    if len(segment) > 1:
                        isi_segments.append(compute_isis(segment))

                if not isi_segments:
                    debug_log.append({
                        "session": session,
                        "unit": unit_id,
                        "isi_len": 0
                    })
                    continue

                isi = np.concatenate(isi_segments)
                if len(isi) >= min_isi_len:
                    key = f"{session}__unit{unit_id}__non_task"
                    isi_dict[key] = isi.astype(np.float32)
                    debug_log.append({
                        "session": session,
                        "unit": unit_id,
                        "isi_len": len(isi)
                    })
    except Exception as e:
        print(f"❌ Failed to load {session}: {e}")

Path(output_npz_path).parent.mkdir(parents=True, exist_ok=True)
np.savez(output_npz_path, **isi_dict)
pd.DataFrame(debug_log).to_csv(debug_log_path, index=False)

print(f"✅ Saved {len(isi_dict)} valid non-task ISI trains (RAW)")
print(f"📝 Debug info: {debug_log_path}")
print(f"💾 Output NPZ: {output_npz_path}")
