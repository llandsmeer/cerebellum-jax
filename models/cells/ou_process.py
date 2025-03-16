import brainpy.math as bm


class OUProcess:
    def __init__(self, size, I_OU0, tau_OU, sigma_OU):
        """Initialize Ornstein-Uhlenbeck process.

        Args:
            size: Number of processes
            I_OU0: Baseline current (nA)
            tau_OU: Time constant (ms)
            sigma_OU: Standard deviation
        """
        self.size = size
        self.I_OU0 = bm.asarray(I_OU0)
        self.tau_OU = bm.asarray(tau_OU)
        self.sigma_OU = bm.asarray(sigma_OU)

        # State variable
        self.I_OU = bm.Variable(bm.ones(size) * I_OU0)

    def update(self, dt):
        """Update the OU process.

        Args:
            dt: Time step (ms)
        """
        # Generate Gaussian noise
        xi = bm.random.normal(0, 1, self.size)

        # Update OU process using Euler-Maruyama method
        dI_OU = (
            (self.I_OU0 - self.I_OU) / self.tau_OU
            + self.sigma_OU * bm.sqrt(2.0 / self.tau_OU) * xi
        ) * dt

        self.I_OU.value = self.I_OU + dI_OU
        return self.I_OU
