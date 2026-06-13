import torch
import torch.nn as nn
import torch.nn.functional as func

class HarmonicGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        N = node_features.shape[0]
        support = self.linear(node_features)
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        source_features = support[:, source_nodes, :]
        aggregated = torch.zeros_like(support)
        aggregated.index_add_(1, target_nodes, source_features)
        degree = torch.zeros(num_nodes, device=node_features.device)
        degree.index_add_(0, target_nodes, torch.ones(len(target_nodes), device=node_features.device))
        degree = torch.clamp(degree, min=1.0)
        aggregated = aggregated / degree.view(1, num_nodes, 1)
        return support + aggregated + self.bias

class HarmonicGCNBottleneck(nn.Module):
    def __init__(self, channels: int, edge_index: torch.Tensor, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.dropout_prob = dropout
        self.register_buffer("edge_index", edge_index)
        self.gcn1 = HarmonicGraphConv(channels, channels)
        self.gcn2 = HarmonicGraphConv(channels, channels)
        self.bn1 = nn.LayerNorm(channels)
        self.bn2 = nn.LayerNorm(channels)
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, freq_bins, T = x.shape
        residual = x
        h = x.permute(0, 3, 2, 1)
        h = h.reshape(B * T, freq_bins, C)
        h = self.gcn1(h, self.edge_index, freq_bins)
        h = self.bn1(h)
        h = func.relu(h)
        h = func.dropout(h, p=self.dropout_prob, training=self.training)
        h = self.gcn2(h, self.edge_index, freq_bins)
        h = self.bn2(h)
        h = h.reshape(B, T, freq_bins, C)
        h = h.permute(0, 3, 2, 1)
        return residual + self.residual_scale * h

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
