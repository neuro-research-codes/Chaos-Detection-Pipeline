import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pynwb import NWBHDF5IO
from pathlib import Path

# === Config ===
nwb_root = "/path-to-data-nwb-files/"
output_npz_path = "/path-to-folder/task_relevant_isi_trains_filtered.npz"
debug_log_path = "/path-to-folder/isi_debug_log.csv"
min_isi_len = 20

def compute_isis(spike_times):
    return np.diff(np.sort(spike_times))

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

            for unit_id in units_df.index:
                spikes = np.array(units_df.at[unit_id, "spike_times"])
                for phase in ["learn", "recog"]:
                    isi_list = []
                    for _, row in trials_df.iterrows():
                        if str(row["stim_phase"]).lower() != phase:
                            continue
                        trial_spikes = spikes[(spikes >= row["start_time"]) & (spikes <= row["stop_time"])]
                        if len(trial_spikes) > 1:
                            isi_list.append(compute_isis(trial_spikes))
                    if len(isi_list) == 0:
                        debug_log.append({
                            "session": session,
                            "unit": unit_id,
                            "phase": phase,
                            "isi_len": 0
                        })
                        continue
                    isi = np.concatenate(isi_list)
                    if len(isi) >= min_isi_len:
                        key = f"{session}__unit{unit_id}__{phase}"
                        isi_dict[key] = isi.astype(np.float32)
                        debug_log.append({
                            "session": session,
                            "unit": unit_id,
                            "phase": phase,
                            "isi_len": len(isi)
                        })

    except Exception as e:
        print(f"❌ Failed to load {session}: {e}")

Path(output_npz_path).parent.mkdir(parents=True, exist_ok=True)
np.savez(output_npz_path, **isi_dict)
pd.DataFrame(debug_log).to_csv(debug_log_path, index=False)

print(f"✅ Saved {len(isi_dict)} valid ISI trains (RAW)")
print(f"📝 Debug info: {debug_log_path}")
print(f"💾 Output NPZ: {output_npz_path}")