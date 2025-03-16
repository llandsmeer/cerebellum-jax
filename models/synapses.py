import brainpy as bp
import brainpy.math as bm
import numpy as np


class RateMeasurement(bp.dyn.SynConn):
    """Synapse for measuring firing rates of neurons."""

    def __init__(self, pre, post, rate_window, name=None):
        """Initialize the rate measurement synapse.

        Args:
            pre: Pre-synaptic neuron group
            post: Post-synaptic neuron group
            rate_window: Time window for rate measurement (ms)
            name: Name of the synapse
        """
        super().__init__(pre=pre, post=post, name=name)

        # Parameters
        self.rate_window = bm.asarray(rate_window)

        # State variables
        self.recent_rate = bm.Variable(bm.zeros(post.num))

        # Connections
        self.conn = bp.conn.OneToOneConn(pre.num, post.num)

    def update(self):
        # Get pre-synaptic spikes
        pre_spike = self.pre.spike

        # Update recent rate
        self.recent_rate.value = bm.where(
            pre_spike[self.conn.pre_ids],
            self.recent_rate + 1.0 / self.rate_window,
            self.recent_rate,
        )

        # Decay recent rate
        dt = bp.share["dt"]
        self.recent_rate.value = self.recent_rate * (1.0 - dt / self.rate_window)


class NoiseToPC(bp.dyn.SynConn):
    """Synapse connecting noise inputs to Purkinje cells."""

    def __init__(self, pre, post, conn, weights=None, name=None):
        """Initialize the noise to PC synapse.

        Args:
            pre: Pre-synaptic neuron group (noise source)
            post: Post-synaptic neuron group (Purkinje cells)
            conn: Connection pattern
            weights: Synaptic weights
            name: Name of the synapse
        """
        super().__init__(pre=pre, post=post, name=name)

        # Connections
        self.conn = conn

        # Parameters
        if weights is None:
            # Initialize weights with random values that sum to 5 for each post-synaptic neuron
            alpha = 2.0  # concentration parameter > 1 favors center of simplex
            raw_weights = np.zeros((post.num, pre.num))
            for i in range(post.num):
                pre_ids = self.conn.pre_ids(i)
                if len(pre_ids) > 0:
                    raw_weights[i, pre_ids] = (
                        np.random.dirichlet(alpha * np.ones(len(pre_ids))) * 5.0
                    )
            self.weights = bm.Variable(bm.asarray(raw_weights))
        else:
            self.weights = bm.Variable(bm.asarray(weights))

    def update(self):
        # Get pre-synaptic currents (from noise sources)
        pre_I = self.pre.I_OU

        # Update post-synaptic noise current
        post_I_noise = bm.zeros(self.post.num)

        for i in range(self.post.num):
            pre_ids = self.conn.pre_ids(i)
            if len(pre_ids) > 0:
                post_I_noise = post_I_noise.at[i].set(
                    bm.sum(self.weights[i, pre_ids] * pre_I[pre_ids]) / len(pre_ids)
                )

        self.post.I_Noise = post_I_noise


class PCToDCN(bp.dyn.SynConn):
    """Synapse connecting Purkinje cells to Deep Cerebellar Nuclei."""

    def __init__(self, pre, post, conn, delay=10.0, name=None):
        """Initialize the PC to DCN synapse.

        Args:
            pre: Pre-synaptic neuron group (Purkinje cells)
            post: Post-synaptic neuron group (DCN cells)
            conn: Connection pattern
            delay: Synaptic delay (ms)
            name: Name of the synapse
        """
        super().__init__(pre=pre, post=post, name=name)

        # Connections
        self.conn = conn

        # Parameters
        self.delay = delay

        # Create a spike queue for delayed spikes
        self.spike_queue = bp.dyn.SpikeQueue(delay, pre.spike.shape)

    def update(self):
        # Push current spikes to the queue
        self.spike_queue.push(self.pre.spike)

        # Get delayed spikes
        delayed_spikes = self.spike_queue.pop()

        # Update DCN inhibition based on PC spikes
        for i in range(self.post.num):
            pre_ids = self.conn.pre_ids(i)
            if len(pre_ids) > 0 and bm.any(delayed_spikes[pre_ids]):
                self.post.update_pc_inhibition(
                    bm.ones(self.post.num, dtype=bool).at[i].set(True)
                )


class DCNToIO(bp.dyn.SynConn):
    """Synapse connecting Deep Cerebellar Nuclei to Inferior Olive."""

    def __init__(
        self, pre, post, conn, weight=1.0, delay=50.0, shunting=True, name=None
    ):
        """Initialize the DCN to IO synapse.

        Args:
            pre: Pre-synaptic neuron group (DCN cells)
            post: Post-synaptic neuron group (IO cells)
            conn: Connection pattern
            weight: Synaptic weight
            delay: Synaptic delay (ms)
            shunting: Whether to include shunting inhibition
            name: Name of the synapse
        """
        super().__init__(pre=pre, post=post, name=name)

        # Connections
        self.conn = conn

        # Parameters
        self.weight = bm.asarray(weight)
        self.delay = delay
        self.shunting = shunting

        # Create a spike queue for delayed spikes
        self.spike_queue = bp.dyn.SpikeQueue(delay, pre.spike.shape)

    def update(self):
        # Push current spikes to the queue
        self.spike_queue.push(self.pre.spike)

        # Get delayed spikes
        delayed_spikes = self.spike_queue.pop()

        # Update IO currents based on DCN spikes
        for i in range(self.post.num):
            pre_ids = self.conn.pre_ids(i)
            if len(pre_ids) > 0:
                # Count number of active pre-synaptic neurons
                active_pre = bm.sum(delayed_spikes[pre_ids])
                if active_pre > 0:
                    # Update IO current
                    self.post.I_IO_DCN = (
                        self.post.I_IO_DCN + self.weight * active_pre / len(pre_ids)
                    )

                    # Apply shunting inhibition if enabled
                    if self.shunting:
                        self.post.g_c = self.post.g_c - 0.001 * active_pre / len(
                            pre_ids
                        )


class IOToPC(bp.dyn.SynConn):
    """Synapse connecting Inferior Olive to Purkinje cells."""

    def __init__(self, pre, post, conn, weight=1.0, delay=15.0, name=None):
        """Initialize the IO to PC synapse.

        Args:
            pre: Pre-synaptic neuron group (IO cells)
            post: Post-synaptic neuron group (Purkinje cells)
            conn: Connection pattern
            weight: Synaptic weight (nA)
            delay: Synaptic delay (ms)
            name: Name of the synapse
        """
        super().__init__(pre=pre, post=post, name=name)

        # Connections
        self.conn = conn

        # Parameters
        self.weight = bm.asarray(weight)
        self.delay = delay

        # Create a spike queue for delayed spikes
        self.spike_queue = bp.dyn.SpikeQueue(delay, pre.spike.shape)

    def update(self):
        # Push current spikes to the queue
        self.spike_queue.push(self.pre.spike)

        # Get delayed spikes
        delayed_spikes = self.spike_queue.pop()

        # Update PC adaptation based on IO spikes
        climbing_fiber_spikes = bm.zeros(self.post.num, dtype=bool)

        for i in range(self.post.num):
            pre_ids = self.conn.pre_ids(i)
            if len(pre_ids) > 0 and bm.any(delayed_spikes[pre_ids]):
                # Increase adaptation current
                self.post.w.value = self.post.w.value.at[i].set(
                    self.post.w[i] + self.weight
                )

                # Mark for PF weight update
                climbing_fiber_spikes = climbing_fiber_spikes.at[i].set(True)

        # Update PF weights based on climbing fiber input
        if bm.any(climbing_fiber_spikes):
            self.post.update_pf_weights(climbing_fiber_spikes)


class BCMSynapse(bp.dyn.SynConn):
    """Implementation of the BCM (Bienenstock-Cooper-Munro) plasticity rule."""

    def __init__(self, pre, post, conn, tau_thresh=15.0, name=None):
        """Initialize the BCM synapse.

        Args:
            pre: Pre-synaptic neuron group
            post: Post-synaptic neuron group
            conn: Connection pattern
            tau_thresh: Time constant for threshold (ms)
            name: Name of the synapse
        """
        super().__init__(pre=pre, post=post, name=name)

        # Connections
        self.conn = conn

        # Parameters
        self.tau_thresh = bm.asarray(tau_thresh)

        # State variables
        self.thresh_M = bm.Variable(
            bm.ones(post.num) * 60.0
        )  # Moving threshold baseline (Hz)
        self.delta_weight_BCM = bm.Variable(bm.zeros(post.num))
        self.phi_BCM = bm.Variable(bm.zeros(post.num))

    def update(self):
        dt = bp.share["dt"]

        # Get pre and post-synaptic rates
        pre_rate = self.pre.recent_rate
        post_rate = self.post.recent_rate

        # Update BCM variables for each post-synaptic neuron
        for i in range(self.post.num):
            pre_ids = self.conn.pre_ids(i)
            if len(pre_ids) > 0:
                # Calculate phi (BCM modification function)
                phi = bm.tanh(
                    post_rate[i] * (post_rate[i] - self.thresh_M[i]) / self.thresh_M[i]
                )

                # Update phi_BCM
                self.phi_BCM.value = self.phi_BCM.value.at[i].set(
                    self.phi_BCM[i] + dt * (phi - self.phi_BCM[i]) / self.tau_thresh
                )

                # Update delta_weight_BCM
                self.delta_weight_BCM.value = self.delta_weight_BCM.value.at[i].set(
                    5.0 * self.phi_BCM[i] * bm.mean(pre_rate[pre_ids])
                )

                # Update threshold
                self.thresh_M.value = self.thresh_M.value.at[i].set(
                    self.thresh_M[i]
                    + dt * ((post_rate[i] ** 2) - self.thresh_M[i] / self.tau_thresh)
                )


def create_one_to_one_conn(pre_size, post_size):
    """Create a one-to-one connection pattern."""
    return bp.conn.OneToOneConn(pre_size, post_size)


def create_all_to_all_conn(pre_size, post_size):
    """Create an all-to-all connection pattern."""
    return bp.conn.AllToAllConn(pre_size, post_size)


def create_random_conn(pre_size, post_size, prob):
    """Create a random connection pattern with given probability."""
    return bp.conn.RandomConn(pre_size, post_size, prob)


def create_fixed_prob_conn(pre_size, post_size, prob):
    """Create a fixed probability connection pattern."""
    conn_mat = np.random.random((post_size, pre_size)) < prob
    return bp.conn.MatConn(conn_mat)


def create_fixed_pre_conn(pre_size, post_size, n_pre):
    """Create a connection where each post-synaptic neuron receives from n_pre pre-synaptic neurons."""
    conn_mat = np.zeros((post_size, pre_size), dtype=bool)
    for i in range(post_size):
        conn_mat[i, np.random.choice(pre_size, n_pre, replace=False)] = True
    return bp.conn.MatConn(conn_mat)


def create_fixed_post_conn(pre_size, post_size, n_post):
    """Create a connection where each pre-synaptic neuron connects to n_post post-synaptic neurons."""
    conn_mat = np.zeros((post_size, pre_size), dtype=bool)
    for j in range(pre_size):
        conn_mat[np.random.choice(post_size, n_post, replace=False), j] = True
    return bp.conn.MatConn(conn_mat)


def create_io_sources_conn(n_cells_io, n_io_project):
    """Create IO to IO connections based on the provided function."""
    # Generate random connections
    sources = np.concatenate(
        [np.full(n_io_project, io_num) for io_num in range(n_cells_io)]
    )
    targets = np.concatenate(
        [
            np.random.choice(
                np.setdiff1d(range(n_cells_io), [io_num]),
                size=n_io_project,
                replace=False,
            )
            for io_num in range(n_cells_io)
        ]
    )

    # Ensure reciprocal connections
    reciprocal_sources = np.concatenate(
        [targets[i * n_io_project : (i + 1) * n_io_project] for i in range(n_cells_io)]
    )
    reciprocal_targets = np.concatenate(
        [sources[i * n_io_project : (i + 1) * n_io_project] for i in range(n_cells_io)]
    )

    sources = np.concatenate([sources, reciprocal_sources])
    targets = np.concatenate([targets, reciprocal_targets])

    # Create connection matrix
    conn_mat = np.zeros((n_cells_io, n_cells_io), dtype=bool)
    conn_mat[targets, sources] = True

    return bp.conn.MatConn(conn_mat)


def create_noise_pc_conn(n_noise, n_cells_pc):
    """Create Noise to PC connections based on the provided function."""
    # Each PC receives input from all noise sources
    conn_mat = np.ones((n_cells_pc, n_noise), dtype=bool)

    # Initialize weights
    weights = np.zeros((n_cells_pc, n_noise))
    for i in range(n_cells_pc):
        w = np.random.random(n_noise) + 0.2
        weights[i] = w / w.sum() * 5.0  # Normalize to sum to 5

    return bp.conn.MatConn(conn_mat), weights


def create_pc_dcn_conn(n_cells_pc, n_cells_dcn, n_pc_dcn_converge, n_pc_dcn_project):
    """Create PC to DCN connections based on the provided function."""
    sources = [pc_num for pc_num in range(n_cells_pc) for _ in range(n_pc_dcn_project)]
    targets = np.concatenate(
        [
            np.random.choice(n_pc_dcn_converge, size=n_pc_dcn_project, replace=False)
            for _ in range(n_cells_pc)
        ]
    )

    # Create connection matrix
    conn_mat = np.zeros((n_cells_dcn, n_cells_pc), dtype=bool)
    conn_mat[targets, sources] = True

    return bp.conn.MatConn(conn_mat)


def create_io_pc_conn(n_cells_io, n_cells_pc, io_conn_ratio):
    """Create IO to PC connections based on the provided function."""
    sources = []
    targets = []

    if int(n_cells_io / io_conn_ratio) >= n_cells_pc:
        source_indices = np.random.choice(
            range(int(n_cells_io / io_conn_ratio)), n_cells_pc, replace=False
        )
    else:
        source_indices = np.random.choice(
            range(n_cells_io), int(n_cells_io / io_conn_ratio), replace=False
        )

    for target_index in range(n_cells_pc):
        if target_index < int(n_cells_io / io_conn_ratio):
            source_index = source_indices[target_index]
        else:
            source_index = np.random.choice(source_indices)
        sources.append(source_index)
        targets.append(target_index)

    # Create connection matrix
    conn_mat = np.zeros((n_cells_pc, n_cells_io), dtype=bool)
    conn_mat[targets, sources] = True

    return bp.conn.MatConn(conn_mat)
