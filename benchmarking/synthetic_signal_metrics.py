import numpy as np
from scipy.stats import linregress, zscore
from scipy.spatial import distance
import itertools

np.random.seed(4772)

# === Phase-Randomized Surrogate ===
def phase_randomize(x):
    Xf = np.fft.rfft(x)
    phases = np.random.uniform(0, 2*np.pi, len(Xf))
    phases[0] = 0
    if len(Xf) % 2 == 0:
        phases[-1] = 0
    return np.fft.irfft(np.abs(Xf) * np.exp(1j * phases), n=len(x))

# === Sample Entropy ===
def sample_entropy(x, m, r):
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

def sweep_sampen(x, m_vals=[2, 3], r_factors=[0.15, 0.2, 0.25]):
    x = zscore(x)
    best = -np.inf
    std_x = np.std(x)
    for m, rfac in itertools.product(m_vals, r_factors):
        r = rfac * std_x
        val = sample_entropy(x, m, r)
        if np.isfinite(val) and val > best:
            best = val
    return best

# === Correlation Dimension ===
def sweep_corr_dim(x, m_vals=[2,3,4,5], theiler=10):
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
        try:
            slope, *_ = linregress(log_r[fit_start:fit_end], log_C[fit_start:fit_end])
            if slope > best:
                best = slope
        except:
            continue
    return best

# === Lyapunov Exponent (Wolf’s algorithm) ===
def wolf_lyapunov(x, tau, dim, max_ratio=10):
    x = zscore(x)
    N = len(x) - (dim - 1) * tau
    if N < 50:
        return np.nan
    X = np.array([x[i:N+i] for i in range(0, dim * tau, tau)]).T
    D = distance.squareform(distance.pdist(X))
    np.fill_diagonal(D, np.inf)
    divs = []
    for i in range(len(X) - 1):
        j = np.argmin(D[i])
        if j + 1 >= len(X) or i + 1 >= len(X):
            continue
        d0 = np.linalg.norm(X[i] - X[j])
        d1 = np.linalg.norm(X[i+1] - X[j+1])
        if 0 < d0 < d1 < max_ratio * d0:
            divs.append(np.log(d1 / d0))
    if len(divs) < 5:
        return np.nan
    return np.mean(divs)

def sweep_lyapunov(x, m_vals=[3, 4, 5, 6], tau_vals=[1, 2, 3]):
    best_le = -np.inf
    for m in m_vals:
        for tau in tau_vals:
            le = wolf_lyapunov(x, tau=tau, dim=m)
            if not np.isnan(le) and le > best_le:
                best_le = le
    return best_le

# === Statistics ===
def empirical_p_value(real, null):
    """
    Returns:
      p    = fraction of null values strictly less than real
      mu   = mean of null distribution
      std  = std. dev. of null distribution
    """
    null = np.asarray(null)
    p = np.mean(null < real)
    return p, null.mean(), null.std()

def zscore_corrected(real, null):
    mu, std = np.mean(null), np.std(null)
    z = (real - mu) / std if std > 1e-8 else np.nan
    p, _, _ = empirical_p_value(real, null)
    return z, p, mu, std

# === Run All Chaos Metrics ===
def run_all_metrics(x, n_surrogates=50, verbose=True):
    x = np.asarray(x)

    # SampEn
    real_sampen = sweep_sampen(x)
    null_sampen = [sweep_sampen(phase_randomize(x)) for _ in range(n_surrogates)]
    z_sampen, p_sampen, mu_sampen, std_sampen = zscore_corrected(real_sampen, null_sampen)

    # CorrDim
    real_cd = sweep_corr_dim(x)
    null_cd = [sweep_corr_dim(phase_randomize(x)) for _ in range(n_surrogates)]
    z_cd, p_cd, mu_cd, std_cd = zscore_corrected(real_cd, null_cd)

    # LE
    real_le = sweep_lyapunov(x)
    null_le = [sweep_lyapunov(phase_randomize(x)) for _ in range(n_surrogates)]
    z_le, p_le, mu_le, std_le = zscore_corrected(real_le, null_le)

    if verbose:
        print("\n=== SampEn ===")
        print(f"Real: {real_sampen:.4f}, Null μ: {mu_sampen:.4f}, σ: {std_sampen:.4f}, Z: {z_sampen:.2f}, p: {p_sampen:.4f}")
        print("\n=== CorrDim ===")
        print(f"Real: {real_cd:.4f}, Null μ: {mu_cd:.4f}, σ: {std_cd:.4f}, Z: {z_cd:.2f}, p: {p_cd:.4f}")
        print("\n=== Lyapunov Exponent ===")
        print(f"Real: {real_le:.4f}, Null μ: {mu_le:.4f}, σ: {std_le:.4f}, Z: {z_le:.2f}, p: {p_le:.4f}")

    return {
        "sampen": {"real": real_sampen, "z": z_sampen, "p": p_sampen},
        "cd": {"real": real_cd, "z": z_cd, "p": p_cd},
        "le": {"real": real_le, "z": z_le, "p": p_le}
    }

# === Example Time Series Generators ===
def generate_rossler(N):
    dt = 0.01
    x, y, z = 0.1, 0.0, 0.0
    xs = []
    for _ in range(N):
        dx = -y - z
        dy = x + 0.2 * y
        dz = 0.2 + z * (x - 5.7)
        x += dx * dt
        y += dy * dt
        z += dz * dt
        xs.append(x)
    return np.array(xs)

def generate_logistic(N, r=3.99):
    x = 0.5
    xs = []
    for _ in range(N):
        x = r * x * (1 - x)
        xs.append(x)
    return np.array(xs)

def generate_tent(N, mu=1.999):
    x = 0.1
    xs = []
    for _ in range(N):
        x = mu * x if x < 0.5 else mu * (1 - x)
        xs.append(x)
    return np.array(xs)

# === Run Script ===
if __name__ == "__main__":
    N = 1000
    t = np.linspace(0, 4*np.pi, N)
    sine = np.sin(t)
    noise = np.random.normal(0, 1, N)
    logistic = generate_logistic(N)
    rossler = generate_rossler(N)
    tent = generate_tent(N)

    for name, signal in zip(
        ["White Noise", "Sine Wave", "Logistic Map", "Rossler", "Tent"],
        [noise, sine, logistic, rossler, tent]
    ):
        print(f"\n### {name} ###")
        run_all_metrics(signal, n_surrogates=100)
