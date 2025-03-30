import brainpy as bp
import brainpy.math as bm


class PurkinjeCell(bp.dyn.NeuDyn):
    def __init__(self, size, **kwargs):
        """Initialize the Purkinje cell population.
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
                t_ref: Refractory period (ms)
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

        # State variables
        self.V = bm.Variable(bm.asarray(kwargs["v_init"]))  # membrane potential
        self.w = bm.Variable(bm.asarray(kwargs["w_init"]))  # adaptation variable
        self.I_intrinsic = bm.asarray(kwargs["I_intrinsic"])  # intrinsic current
        self.input = bm.Variable(bm.zeros(size))  # input current

        # Debug variables for membrane potential terms
        self.dbg_leak = bm.Variable(bm.zeros(size))
        self.dbg_exp = bm.Variable(bm.zeros(size))
        self.dbg_current = bm.Variable(bm.zeros(size))
        self.dbg_w = bm.Variable(bm.zeros(size))
        self.dbg_delta_w = bm.Variable(bm.zeros(size))

        # Spike tracking
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
        self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

        # Integration functions
        self.integral_v = bp.odeint(f=self.dv, method="exp_auto")
        self.integral_w = bp.odeint(f=self.dw, method="exp_auto")

    def dv(self, V, t, w):
        """Membrane potential dynamics"""
        I_total = self.I_intrinsic + self.input.value
        dv = (
            1000  # nA to pA
            * (
                # microS * mV = nA
                self.gL * (self.EL - V)
                # microS * mV * 1 = nA
                + self.gL * self.DeltaT * bm.exp((V - self.VT) / self.DeltaT)
                # nA
                + I_total
                # nA
                - w
            )
            / self.C
            # pA / pF = mV/ms
        )
        return dv

    def dw(self, w, t, V):
        """Adaptation current dynamics"""
        #    microS * mV = nA            # nA  # ms
        dw = (self.a * (V - self.EL) - w) / self.tauw
        return dw

    def update(self):
        t = bp.share["t"]
        dt = bp.share["dt"]

        # Integrate membrane potential and adaptation current
        V = self.integral_v(self.V, t, self.w, dt=dt)
        self.V = V
        w = self.integral_w(self.w, t, self.V, dt=dt)
        self.w = w

        # Store individual terms for debugging
        I_total = self.I_intrinsic + self.input.value
        self.dbg_leak.value = self.gL * (self.EL - V)
        self.dbg_exp.value = self.gL * self.DeltaT * bm.exp((V - self.VT) / self.DeltaT)
        self.dbg_current.value = I_total
        self.dbg_w.value = w
        self.dbg_delta_w.value = (self.a * (V - self.EL) - w) / self.tauw

        # Spike detection
        spike = V > self.Vcut
        self.spike.value = spike

        # Update last spike time
        self.t_last_spike.value = bm.where(spike, t, self.t_last_spike)

        # Reset membrane potential and update adaptation for spiking neurons
        self.V.value = bm.where(spike, self.Vr, V)
        self.w.value = bm.where(spike, w + self.b, w)
