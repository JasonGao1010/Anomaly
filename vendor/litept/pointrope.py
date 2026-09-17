"""The upstream three-axis PointROPE formula without native-kernel constraints."""

import torch


class PointROPE(torch.nn.Module):
    def __init__(self, freq=100.0):
        super().__init__()
        self.freq = freq

    def forward(self, tokens, positions):
        width = tokens.shape[-1] // 3
        if tokens.shape[-1] % 6 or positions.shape != (tokens.shape[0], tokens.shape[2], 3):
            raise ValueError("PointROPE requires three even-dimensional axis subspaces")
        frequency = self.freq ** (-torch.arange(0, width, 2, device=tokens.device).float() / width)
        phase = positions.float()[..., None] * frequency
        phase = torch.cat((phase, phase), -1).flatten(-2).unsqueeze(1)
        first, second = tokens.unflatten(-1, (3, width)).chunk(2, dim=-1)
        rotated = torch.cat((-second, first), -1).flatten(-2)
        return tokens * phase.cos() + rotated * phase.sin()
