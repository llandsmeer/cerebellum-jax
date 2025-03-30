import brainpy as bp
import brainpy.math as bm


class DeepCerebellarNuclei(bp.dyn.NeuDyn):
    def __init__(self, size, **kwargs):
        """Initialize the Deep Cerebellar Nuclei (DCN) population.

        Args:
            size: Number of cells.
            kwargs: Dictionary containing the following parameters:
                C: Capacitance (pF)
                gL: Leak conductance (microS)
                EL: Resting potential (mV)
                VT: Threshold potential for exponential term (mV)
                DeltaT: Slope factor (mV)
                tauw: Adaptation time constant (ms)
                a: Subthreshold adaptation (microS)
                b: Spike-triggered adaptation increment (nA)
                Vr: Reset potential (mV)
                v_init: Initial membrane potential (mV)
                w_init: Initial adaptation variable
                I_intrinsic: Intrinsic current (nA)
                tauI: PC inhibition time constant (ms)

                # MF parameters
                I_OU0_MF: MF baseline current (nA)
                tau_OU_MF: MF time constant (ms)
                sigma_OU_MF: MF noise intensity
        """
        super().__init__(size=size)

        # Parameters
        self.C = bm.asarray(kwargs["C"])  # capacitance
        self.gL = bm.asarray(kwargs["gL"])  # leak conductance
        self.EL = bm.asarray(kwargs["EL"])  # leak reversal potential
        self.VT = bm.asarray(kwargs["VT"])  # threshold potential
        self.DeltaT = bm.asarray(kwargs["DeltaT"])  # slope factor
        self.Vcut = self.VT + 5 * self.DeltaT  # spike cutoff potential
        self.tauw = bm.asarray(kwargs["tauw"])  # adaptation time constant
        self.a = bm.asarray(kwargs["a"])  # subthreshold adaptation
        self.b = bm.asarray(kwargs["b"])  # spike-triggered adaptation increment
        self.Vr = bm.asarray(kwargs["Vr"])  # reset potential
        self.tauI = bm.asarray(kwargs["tauI"])  # PC inhibition time constant

        # State variables
        self.V = bm.Variable(bm.asarray(kwargs["v_init"]))
        self.w = bm.Variable(bm.asarray(kwargs["w_init"]))
        self.input = bm.Variable(bm.zeros(size))
        self.I_intrinsic = bm.asarray(kwargs["I_intrinsic"])

        # PC inhibition
        self.I_PC = bm.Variable(bm.zeros(self.num))

        # Spike tracking
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
        self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

        # Integration functions
        self.integral_I_PC = bp.odeint(f=self.dI_PC, method="exp_auto")
        self.integral_V = bp.odeint(f=self.dV, method="exp_auto")
        self.integral_w = bp.odeint(f=self.dw, method="exp_auto")

    def dI_PC(self, I_PC, t):
        return (-I_PC) / self.tauI

    def dV(self, V, t, w, I_PC):
        """Membrane potential dynamics"""
        I_total = self.I_intrinsic + self.input.value - I_PC  # nA
        dV = (
            1000  # nA to pA
            * (
                # microS * mV = nA
                self.gL * (self.EL - V)
                # microS * mV = nA
                + self.gL * self.DeltaT * bm.exp((V - self.VT) / self.DeltaT)
                # nA
                + I_total
                # nA
                - w
            )
            / self.C
            # pA / pF = mV/ms
        )
        return dV

    def dw(self, w, t, V):
        """Adaptation current dynamics"""
        dw = (self.a * (V - self.EL) - w) / self.tauw
        return dw

    def update(self):
        t = bp.share["t"]
        dt = bp.share["dt"]

        # Integrate membrane potential and adaptation current
        self.I_PC.value = self.integral_I_PC(self.I_PC, t, dt=dt)
        V = self.integral_V(self.V, t, self.w, self.I_PC, dt=dt)
        self.V = V
        w = self.integral_w(self.w, t, self.V, dt=dt)
        self.w = w

        # Spike detection
        spike = V > self.Vcut
        self.spike.value = spike

        # Update last spike time
        self.t_last_spike.value = bm.where(spike, t, self.t_last_spike)

        # Reset membrane potential and update adaptation for spiking neurons
        self.V.value = bm.where(spike, self.Vr, V)
        self.w.value = bm.where(spike, w + self.b, w)
