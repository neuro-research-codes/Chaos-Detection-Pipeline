import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import itertools
import pandas as pd
import os
import time
from tqdm import tqdm

np.random.seed(4772)

# === Utility Functions ===
def empirical_p_value(real, null, tail="right"):
    null = np.asarray(null)
    if tail == "left":
        p = np.mean(null <= real)
    else:
        p = np.mean(null >= real)
    return p, np.mean(null), np.std(null)

def zscore_corrected(real, null):
    null = np.asarray(null)
    mu, std = np.mean(null), np.std(null)
    z = (real - mu) / std if std > 1e-8 else np.nan
    p = np.mean(null >= real)
    return z, p, mu, std

def phase_randomize(x):
    Xf = np.fft.rfft(x)
    phases = np.random.uniform(0, 2*np.pi, len(Xf))
    phases[0] = 0
    if len(Xf) % 2 == 0:
        phases[-1] = 0
    return np.fft.irfft(np.abs(Xf) * np.exp(1j * phases), n=len(x))

def sample_entropy(x, m, r):
    x = zscore(x)

    def _phi(m):
        x_m = np.array([x[i:i + m] for i in range(len(x) - m + 1)])
        C = np.sum(np.max(np.abs(x_m[:, None] - x_m[None, :]), axis=2) < r, axis=0) - 1
        return np.sum(C) / (len(x_m) * (len(x_m) - 1))

    try:
        num = _phi(m + 1)
        denom = _phi(m)
        if denom <= 0 or num <= 0:
            return np.nan
        return -np.log(num / denom)
    except:
        return np.nan

def sweep_sampen(x, m_vals=[2,3], r_factors=[0.15, 0.2, 0.25]):
    best = -np.inf
    for m, rfac in itertools.product(m_vals, r_factors):
        r = rfac * np.std(x)
        val = sample_entropy(x, m, r)
        if np.isfinite(val) and val > best:
            best = val
    return best

def run_sampen_only(x, n_surrogates=50):
    x = np.array(x)
    x_z = zscore(x)
    result = {}

    best_se = -np.inf
    for m in [2, 3]:
        for rf in [0.15, 0.2, 0.25]:
            r = rf * np.std(x_z)
            try:
                val = sample_entropy(x_z, m, r)
                if np.isfinite(val) and val > best_se:
                    best_se = val
                    result["sampen"] = {"real": val}
            except:
                continue

    result.setdefault("sampen", {"real": np.nan})
    result["sampen"]["null"] = [sweep_sampen(phase_randomize(x)) for _ in range(n_surrogates)]

    real = result["sampen"]["real"]
    null = result["sampen"]["null"]
    p, mu, std = empirical_p_value(real, null, tail="left")
    result["sampen"].update({"p": p, "mean": mu, "std": std})
    return result

def process_npz(npz_path, old_csv, out_csv, surr_dir, n_surrogates=100):
    os.makedirs(surr_dir, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    isi_dict = dict(data.items())

    processed_units = set()
    if os.path.exists(old_csv):
        with open(old_csv, "r") as f:
            for line in f:
                if line.startswith("session"): continue
                parts = line.strip().split(",")
                session = parts[0]
                unit_id = parts[1]
                phase = parts[2]
                key = f"{session}__unit{unit_id}__{phase}"
                processed_units.add(key)

    if not os.path.exists(out_csv):
        with open(out_csv, "w") as f:
            f.write("session,unit,phase,sampen_real,sampen_p,sampen_null_mean,sampen_null_std\n")

    print(f"[INFO] Total units in file: {len(isi_dict)}")

    for unit, isi in tqdm(isi_dict.items(), desc="Processing units"):
        if unit in processed_units:
            print(f"[SKIP] {unit} - already processed")
            continue

        if len(isi) < 100:
            print(f"[SKIP] {unit} - ISI too short ({len(isi)})")
            continue
        if len(isi) > 4000:
            print(f"[SKIP] {unit} - ISI too long ({len(isi)})")
            continue

        print(f"[START] {unit} (ISI length: {len(isi)})")
        start_time = time.time()

        result = run_sampen_only(isi, n_surrogates=n_surrogates)
        vals = result["sampen"]
        parts = unit.split("__")
        session = parts[0]
        unit_id = int(parts[1].replace("unit", ""))
        phase = parts[2] if len(parts) > 2 else "NA"

        row = [session, unit_id, phase, f"{vals['real']:.6f}", f"{vals['p']:.4f}", f"{vals['mean']:.6f}", f"{vals['std']:.6f}"]
        with open(out_csv, "a") as f:
            f.write(",".join(map(str, row)) + "\n")


        np.savez(os.path.join(surr_dir, f"{unit}.npz"),
                 isi=np.array(isi),
                 real=vals["real"],
                 surrogates=np.array(vals["null"]),
                 p=vals["p"],
                 mean=vals["mean"],
                 std=vals["std"]
        )

        elapsed = time.time() - start_time
        print(f"[DONE] {unit} processed in {elapsed:.2f} sec\n")


# === Input 1 ===#
# === Example usage for task relevant -- comment out input 2
process_npz(
    npz_path="/path-to-folder/task_restricted_isi_trains_filtered.npz",
    old_csv="/path-to-folder/filename.csv",
    out_csv="/path-to-folder/filename.csv",
    surr_dir="/path-to-folder/"
)

# === Input 2 ===#
# === Example usage for task irrelevant -- comment out input 1
process_npz(
    npz_path="/path-to-folder/task_irrelevant_isi_trains_filtered.npz",
    old_csv="/path-to-folder/filename.csv",
    out_csv="/path-to-folder/filename.csv",
    surr_dir="/path-to-folder/"
)
