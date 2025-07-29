import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import os
from tqdm import tqdm

np.random.seed(4772)

# === Utility Functions ===
def zscore_corrected(real, null):
    null = np.asarray(null)
    null = null[np.isfinite(null)]  # Filter out NaNs
    if len(null) == 0 or not np.isfinite(real):
        return np.nan, np.nan, np.nan, np.nan
    mu, std = np.mean(null), np.std(null)
    z = (real - mu) / std if std > 1e-8 else np.nan
    p = np.mean(null <= real)  # <-- corrected tail
    return z, p, mu, std

def phase_randomize(x):
    Xf = np.fft.rfft(x)
    phases = np.random.uniform(0, 2*np.pi, len(Xf))
    phases[0] = 0
    if len(Xf) % 2 == 0:
        phases[-1] = 0
    return np.fft.irfft(np.abs(Xf) * np.exp(1j * phases), n=len(x))

def wolf_lyapunov(x, tau, dim, max_ratio=10):
    x = zscore(x)
    N = len(x) - (dim - 1) * tau
    if N < 50:
        return np.nan
    X = np.array([x[i:N+i] for i in range(0, dim * tau, tau)]).T
    D = squareform(pdist(X))
    np.fill_diagonal(D, np.inf)
    divs = []
    for i in range(len(X) - 1):
        j = np.argmin(D[i])
        if j + 1 >= len(X) or i + 1 >= len(X): continue
        d0 = np.linalg.norm(X[i] - X[j])
        d1 = np.linalg.norm(X[i+1] - X[j+1])
        if 0 < d0 < d1 < max_ratio * d0:
            divs.append(np.log(d1 / d0))
    return np.mean(divs) if divs else np.nan

# === Main Function ===
def process_npz(npz_path, old_csv, out_csv, surr_dir, n_surrogates=100):
    os.makedirs(surr_dir, exist_ok=True)

    # === Write header if needed ===
    if not os.path.exists(out_csv):
        with open(out_csv, "w") as f:
            f.write("session,unit_id,phase,real,mean_null,std_null,z,p\n")

    # === Load processed units ===
    if os.path.exists(old_csv):
        processed_df = pd.read_csv(old_csv)
        processed_units = set(
            processed_df["session"].astype(str) + "_" +
            processed_df["unit_id"].astype(str) + "_" +
            processed_df["phase"].astype(str)
        )
    else:
        processed_units = set()

    data = np.load(npz_path, allow_pickle=True)
    results = []

    for key in tqdm(data.files, desc="Wolf LE (REL) w/ Max-Null Sweep"):
        session = key.split("_")[0]
        unit_id = key.split("_")[1]
        if "Learn" in key or "learn" in key:
            phase = "learn"
        elif "Recog" in key or "recog" in key:
            phase = "recog"
        else:
            phase = "unknown"

        current_key = f"{session}_{unit_id}_{phase}"

        if current_key in processed_units:
            print(f"⏩ Skipping already processed: {current_key}")
            continue

        x = data[key]
        if len(x) < 50:
            print(f"⚠️ Too short: {key}")
            continue

        try:
            best_le = -np.inf
            best_m, best_tau = None, None
            for m in [3, 4, 5, 6]:
                for tau in [1, 2, 3]:
                    le = wolf_lyapunov(x, tau=tau, dim=m)
                    if not np.isnan(le) and le > best_le:
                        best_le, best_m, best_tau = le, m, tau

            null = []
            for _ in range(n_surrogates):
                surr = phase_randomize(x)
                max_surr_le = -np.inf
                for m in [3, 4, 5, 6]:
                    for tau in [1, 2, 3]:
                        le = wolf_lyapunov(surr, tau=tau, dim=m)
                        if not np.isnan(le) and le > max_surr_le:
                            max_surr_le = le
                null.append(max_surr_le)

            z, p, mu, std = zscore_corrected(best_le, null)

            # === STANDARD SAVE BLOCK ===
            vals = {
                "null": np.array(null),
                "real": best_le,
                "z": z,
                "p": p
            }
            np.savez(os.path.join(surr_dir, f"{key}.npz"),
                     isi=np.array(x),
                     null=vals["null"],
                     real=vals["real"],
                     z=vals["z"],
                     p=vals["p"])

            # === CSV Row ===
            row = [session, unit_id, phase, best_le, np.mean(null), np.std(null), z, p]

            with open(out_csv, "a") as f:
                f.write(",".join(map(str, row)) + "\n")

        except Exception as e:
            print(f"❌ Error {key}: {e}")

# === Input 1 ===#
# === Example usage for task relevant -- comment out input 2
if __name__ == "__main__":
    process_npz(
        npz_path="/path-to-folder/task_restricted_isi_trains_filtered.npz",
        old_csv="/path-to-folder/filename.csv",
        out_csv="/path-to-folder/filename.csv",
        surr_dir="/path-to-folder/"
    )

# === Input 2 ===#
# === Example usage for task irrelevant- comment out input 1
if __name__ == "__main__":
    process_npz(
        npz_path="/path-to-folder/task_irrelevant_isi_trains_filtered.npz",
        old_csv="/path-to-folder/filename.csv",
        out_csv="/path-to-folder/filename.csv",
        surr_dir="/path-to-folder/"
    )