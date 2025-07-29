import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, zscore
from scipy.spatial import distance
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

def sweep_corr_dim(x, m_vals=[2,3,4,5], theiler=10, verbose=False, index=None):
    x = zscore(x)
    best = -np.inf
    for m in m_vals:
        N = len(x) - m + 1
        if N < 2 * theiler:
            continue
        emb = np.array([x[i:i+m] for i in range(N)])
        D = distance.squareform(distance.pdist(emb))
        np.fill_diagonal(D, np.inf)
        for i in range(N):
            D[i, max(0, i-theiler):i+theiler+1] = np.inf
        distances = D[np.isfinite(D)]
        if len(distances) == 0:
            continue
        r_vals = np.logspace(np.log10(np.percentile(distances, 5)),
                             np.log10(np.percentile(distances, 95)), 25)
        log_C = []
        for r in r_vals:
            count = np.sum(distances < r)
            C = count / (N * (N - 1 - 2 * theiler))
            log_C.append(np.log(C + 1e-10))
        log_r = np.log(r_vals)
        fit_start = len(log_r) // 4
        fit_end = 3 * len(log_r) // 4
        slope, *_ = linregress(log_r[fit_start:fit_end], log_C[fit_start:fit_end])
        if slope > best:
            best = slope
    if verbose and index is not None and index % 10 == 0:
        print(f"    [Surrogate {index}] CD slope: {best:.5f}")
    return best

def run_cd_only(x, n_surrogates=50, theiler=10):
    x = np.array(x)
    x_z = zscore(x)
    result = {}

    best_cd = -np.inf
    for m in [2, 3, 4, 5]:
        N = len(x_z) - m + 1
        if N < 2 * theiler:
            continue
        emb = np.array([x_z[i:i+m] for i in range(N)])
        D = distance.squareform(distance.pdist(emb))
        np.fill_diagonal(D, np.inf)
        for i in range(N):
            D[i, max(0, i-theiler):i+theiler+1] = np.inf
        distances = D[np.isfinite(D)]
        if len(distances) == 0:
            continue
        r_vals = np.logspace(np.log10(np.percentile(distances, 5)),
                             np.log10(np.percentile(distances, 95)), 25)
        log_C = []
        for r in r_vals:
            count = np.sum(distances < r)
            C = count / (N * (N - 1 - 2 * theiler))
            log_C.append(np.log(C + 1e-10))
        log_r = np.log(r_vals)
        fit_start = len(log_r) // 4
        fit_end = 3 * len(log_r) // 4
        try:
            slope, *_ = linregress(log_r[fit_start:fit_end], log_C[fit_start:fit_end])
            if slope > best_cd:
                best_cd = slope
                result["corr_dim"] = {"real": slope}
        except:
            continue

    result.setdefault("corr_dim", {"real": np.nan})
    result["corr_dim"]["null"] = [sweep_corr_dim(phase_randomize(x), theiler=theiler, verbose=True, index=i)
                                  for i in range(n_surrogates)]

    real = result["corr_dim"]["real"]
    null = result["corr_dim"]["null"]
    p, mu, std = empirical_p_value(real, null, tail="left")
    result["corr_dim"].update({"p": p, "mean": mu, "std": std})
    return result

def process_npz(npz_path, old_csv, out_csv, surr_dir, n_surrogates=100, theiler=10):
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
            f.write("session,unit,phase,corr_dim_real,corr_dim_p,corr_dim_null_mean,corr_dim_null_std\n")

    print(f"[INFO] Total units in file: {len(isi_dict)}")

    for unit, isi in tqdm(isi_dict.items(), desc="Processing units"):
        if unit in processed_units:
            print(f"[SKIP] {unit} - already processed")
            continue

        if len(isi) < 100:
            print(f"[SKIP] {unit} - ISI too short ({len(isi)})")
            continue

        if len(isi) > 3000:
            print(f"[INFO] {unit} - ISI length capped from {len(isi)} to 3000")
            isi = isi[:3000]

        print(f"[START] {unit} (ISI length: {len(isi)})")
        start_time = time.time()

        result = run_cd_only(isi, n_surrogates=n_surrogates, theiler=theiler)
        vals = result["corr_dim"]
        parts = unit.split("__")
        session = parts[0]
        unit_id = parts[1].replace("unit", "")
        phase = parts[2] if len(parts) > 2 else "NA"

        np.savez(os.path.join(surr_dir, f"{unit}.npz"),
                 isi=isi,
                 null=vals["null"],
                 real=vals["real"],
                 z=vals.get("z", np.nan),
                 p=vals["p"])

        row = [session, unit_id, phase, f"{vals['real']:.6f}", f"{vals['p']:.4f}", f"{vals['mean']:.6f}", f"{vals['std']:.6f}"]
        with open(out_csv, "a") as f:
            f.write(",".join(map(str, row)) + "\n")

        elapsed = time.time() - start_time
        print(f"[DONE] {unit} processed in {elapsed:.2f} sec\n")

# === Input 1 ===#
# === Example usage for task relevant -- comment out input 2.
process_npz(
    npz_path="/path-to-folder/task_restricted_isi_trains_filtered.npz",
    old_csv="/path-to-folder/filename.csv",
    out_csv="/path-to-folder/filename.csv",
    surr_dir="/path-to-folder/",
    theiler=10
)

# === Input 2 ===#
# === Example usage for task irrelevant -- comment out input 1
process_npz(
    npz_path="/path-to-folder/task_irrelevant_isi_trains_filtered.npz",
    old_csv="/path-to-folder/filename.csv",
    out_csv="/path-to-folder/filename.csv",
    surr_dir="/path-to-folder/",
    theiler=10
)
