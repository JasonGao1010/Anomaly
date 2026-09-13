"""Upstream PointROPE rotation, evaluated directly in PyTorch without lookup caches."""

import torch


class PointROPE(torch.nn.Module):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base, self.F0 = freq, F0

    def forward(self, tokens, positions):
        d = tokens.shape[-1] // 3
        if tokens.shape[-1] % 6 or positions.shape != (tokens.shape[0], tokens.shape[2], 3):
            raise ValueError("PointROPE needs three even-dimensional axis subspaces")
        frequency = self.F0 / self.base ** (torch.arange(0, d, 2, device=tokens.device).float() / d)
        # This is the same phase as the upstream integer-position cosine/sine table.
        phase = positions.float()[..., None] * frequency
        phase = torch.cat((phase, phase), -1).flatten(-2).unsqueeze(1)
        x = tokens.unflatten(-1, (3, d))
        first, second = x.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), -1).flatten(-2)
        return tokens * phase.cos() + rotated * phase.sin()
