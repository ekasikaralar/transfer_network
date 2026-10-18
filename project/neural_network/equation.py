import numpy as np
import torch


class Equation(object):
    """Base class for defining PDE-related functions."""

    def __init__(self, eqn_config):

        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)

        # The model is defined on a coarser grid than the splitting.  The
        # nominal plan, the staffing, zeta and Sigma are all estimated at five
        # minute resolution, so the offline networks and the model data change
        # only on that grid, while the splitting step can be refined below it.
        self.num_model_interval = getattr(eqn_config, "num_model_interval",
                                          self.num_time_interval)

    def model_index(self, n):
        """The model grid point that splitting step n falls in."""
        return n * self.num_model_interval // self.num_time_interval

    def f_tf(self, n, x, z):
        """Generator function in the PDE"""
        raise NotImplementedError

    def g_tf(self, x):
        """Terminal condition of the PDE"""
        raise NotImplementedError


class HJB(Equation):
    """The limiting control problem for the transfer network.

    The auxiliary function is

        F(t, x, p) = D(t, x) . p - M(t, x, p) - sum_i c_i x_i,

    where M is the optimal value of a linear program over the feasible set and
    D is the drift the reference policy contributes.  Both require a solve at
    every state visited in training, so both are replaced by the networks
    trained offline and F is evaluated without any optimization.  The terminal
    value is zero, so V(T, x) = 0.
    """

    def __init__(self, eqn_config, drift_nets=None, m_nets=None):
        super(HJB, self).__init__(eqn_config)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        default_mode = getattr(eqn_config, "offline_mode", "separate")
        self.drift_mode = getattr(eqn_config, "drift_mode", default_mode)
        self.m_mode = getattr(eqn_config, "m_mode", default_mode)

        self.drift_nets = self._freeze(drift_nets)
        self.m_nets = self._freeze(m_nets)

    def _freeze(self, nets):
        """The offline networks are fixed inputs, never trained here."""
        if nets is None:
            return None
        if not isinstance(nets, (list, tuple)):
            nets = [nets]
        out = []
        for net in nets:
            net = net.to(self.device)
            net.eval()
            for p in net.parameters():
                p.requires_grad = False
            out.append(net)
        return out

    def _time_feature(self, m, batch):
        """The elapsed fraction of the horizon on the model grid, which is what
        the pooled networks were trained against."""
        return torch.full((batch, 1), m / self.num_model_interval,
                          dtype=torch.float32, device=self.device)

    def drift_term(self, n, x):
        """Dhat(x) at the model grid point containing splitting step n."""
        if self.drift_mode == "nominal":
            return torch.zeros_like(x)
        m = self.model_index(n)
        if self.drift_mode == "separate":
            return self.drift_nets[m](x)
        return self.drift_nets[0](torch.cat([x, self._time_feature(m, x.size(0))], dim=1))

    def m_term(self, n, x, z):
        """Mhat(x, z) at the model grid point containing splitting step n."""
        m = self.model_index(n)
        if self.m_mode == "separate":
            return self.m_nets[m](torch.cat([x, z], dim=1))
        return self.m_nets[0](
            torch.cat([x, z, self._time_feature(m, x.size(0))], dim=1))

    def f_tf(self, n, x, cost, z):
        """Generator function in the PDE."""
        first_term = torch.sum(self.drift_term(n, x) * z, dim=1, keepdim=True)
        second_term = torch.sum(cost * x, dim=1, keepdim=True)
        return first_term - self.m_term(n, x, z) - second_term

    def g_tf(self, x):
        """Terminal condition of the PDE.  The terminal value is zero, so
        V(T, x) = 0 for every state."""
        return torch.zeros((x.size(0), 1), dtype=x.dtype, device=x.device)
