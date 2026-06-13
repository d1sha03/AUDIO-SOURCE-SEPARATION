import torch

def build_harmonic_edges(
    n_freq_bins: int,
    sample_rate: int = 44100,
    n_fft: int = 2048,
    max_harmonic: int = 5,
    tolerance_bins: int = 2,
) -> torch.Tensor:
    freq_per_bin = sample_rate / n_fft
    freqs = torch.arange(n_freq_bins, dtype=torch.float32) * freq_per_bin
    source_nodes = []
    target_nodes = []
    for bin_i in range(1, n_freq_bins):
        freq_i = freqs[bin_i].item()
        for harmonic in range(2, max_harmonic + 1):
            target_freq = freq_i * harmonic
            target_bin = round(target_freq / freq_per_bin)
            if target_bin >= n_freq_bins:
                break
            actual_freq = freqs[target_bin].item()
            freq_error_bins = abs(actual_freq - target_freq) / freq_per_bin
            if freq_error_bins <= tolerance_bins:
                source_nodes.append(bin_i)
                target_nodes.append(target_bin)
                source_nodes.append(target_bin)
                target_nodes.append(bin_i)
    return torch.tensor([source_nodes, target_nodes], dtype=torch.long)

def get_edge_stats(edge_index: torch.Tensor, n_freq_bins: int) -> dict:
    num_edges = edge_index.shape[1]
    degree = torch.zeros(n_freq_bins, dtype=torch.long)
    for node in edge_index[0]:
        degree[node] += 1
    nonzero = degree[degree > 0]
    return {
        "num_nodes": n_freq_bins,
        "num_edges": num_edges,
        "avg_degree": degree.float().mean().item(),
        "max_degree": degree.max().item(),
        "min_degree_nonzero": nonzero.min().item() if len(nonzero) > 0 else 0,
        "isolated_nodes": (degree == 0).sum().item(),
    }
