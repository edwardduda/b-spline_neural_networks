import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLayer(nn.Module):

    def __init__(self, in_features, out_features,
                 num_control_points, degree, range_min, range_max):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_control_points = num_control_points
        self.degree = degree
        self.range_min = range_min
        self.range_max = range_max

        if num_control_points < 2 * degree + 1:
            raise ValueError(
                f"num_control_points ({num_control_points}) must be "
                f">= 2*degree + 1 ({2 * degree + 1})"
            )

        num_basis = num_control_points - degree
        self.control_points = nn.Parameter(
            torch.randn(in_features, out_features, num_basis) * 0.1
        )
        self.residual_weight = nn.Parameter(
            torch.ones(in_features, out_features) * 0.1
        )

        self.register_buffer('knot_vector', self._build_knot_vector())

    def _build_knot_vector(self):
        p, G = self.degree, self.num_control_points
        num_interior = G - 2 * p - 1
        interior = torch.linspace(0, 1, num_interior + 2)[1:-1]
        return torch.cat([torch.zeros(p + 1), interior, torch.ones(p + 1)])

    def _compute_basis(self, x_scaled):

        eps = torch.finfo(x_scaled.dtype).eps * 100
        x_exp = x_scaled.unsqueeze(-1)

        lower = self.knot_vector[:self.num_control_points].view(1, 1, -1)
        upper = self.knot_vector[1:self.num_control_points + 1].view(1, 1, -1)

        N = ((x_exp >= lower) & (x_exp < upper)).float()

        boundary = (x_scaled == 1.0).float().unsqueeze(-1)
        N[:, :, -1:] += boundary

        current_num = self.num_control_points
        for d in range(1, self.degree + 1):
            new_num = current_num - 1

            ki = self.knot_vector[:new_num].view(1, 1, -1)
            kid = self.knot_vector[d:d + new_num].view(1, 1, -1)
            d1 = (kid - ki).clamp(min=eps)

            ki1 = self.knot_vector[1:new_num + 1].view(1, 1, -1)
            kid1 = self.knot_vector[d + 1:d + new_num + 1].view(1, 1, -1)
            d2 = (kid1 - ki1).clamp(min=eps)

            t1 = ((x_exp - ki) / d1) * N[..., :new_num]
            t2 = ((kid1 - x_exp) / d2) * N[..., 1:new_num + 1]

            N = t1 + t2
            current_num = new_num

        return N

    def forward(self, x):
        """
        Args:
            x: (batch, in_features)

        Returns:
            (batch, out_features)  — φ(x) = w_b·silu(x) + w_s·spline(x)
        """
        x_scaled = (x - self.range_min) / (self.range_max - self.range_min)
        x_scaled = torch.clamp(x_scaled, 0.0, 1.0)

        N = self._compute_basis(x_scaled=x_scaled)

        spline = torch.einsum('bik,iok->bo', N, self.control_points)
        residual = torch.einsum('bi,io->bo', F.silu(x), self.residual_weight)
        return residual + spline
