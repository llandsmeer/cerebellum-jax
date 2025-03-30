import numpy as np
import brainpy.math as bm


def generate_pf_pc_connectivity(
    num_pf, num_pc, n_pf_per_pc=5, weight_scale=5.0, dirichlet_alpha=2.0
):
    """
    Generates PF-PC connectivity and weights.
    Each PC receives input from exactly n_pf_per_pc unique PF bundles.
    Weights resemble a scaled symmetric Dirichlet distribution.

    Args:
        num_pf: Total number of PF bundles.
        num_pc: Total number of Purkinje cells.
        n_pf_per_pc: Number of PF inputs per PC.
        weight_scale: Total sum weights are normalized to.
        dirichlet_alpha: Concentration parameter for Dirichlet distribution.

    Returns:
        tuple: (pre_ids, post_ids, weights_matrix)
            pre_ids: Array of presynaptic PF indices for each connection.
            post_ids: Array of postsynaptic PC indices for each connection.
            weights_matrix: Full (num_pc, num_pf) matrix, only connected weights are non-zero.
    """
    pre_ids = []
    post_ids = []
    # Initialize weight matrix (sparse in effect, dense for simplicity here)
    weights_matrix = np.zeros((num_pc, num_pf), dtype=np.float32)

    if num_pf < n_pf_per_pc:
        raise ValueError(
            f"num_pf ({num_pf}) cannot be less than n_pf_per_pc ({n_pf_per_pc})"
        )

    for pc_idx in range(num_pc):
        # Choose n_pf_per_pc unique PFs for this PC
        connected_pfs = np.random.choice(num_pf, n_pf_per_pc, replace=False)

        # Generate weights for these connections using Dirichlet
        # Ensure alpha is applied correctly for the number of connections
        dirichlet_params = dirichlet_alpha * np.ones(n_pf_per_pc)
        conn_weights = np.random.dirichlet(dirichlet_params) * weight_scale

        # Store connections and weights
        for i, pf_idx in enumerate(connected_pfs):
            pre_ids.append(pf_idx)
            post_ids.append(pc_idx)
            weights_matrix[pc_idx, pf_idx] = conn_weights[i]

    print(
        f"PF->PC: Generated {len(pre_ids)} connections. Each PC <- {n_pf_per_pc} PFs."
    )
    return (
        np.array(pre_ids, dtype=np.int32),
        np.array(post_ids, dtype=np.int32),
        bm.asarray(weights_matrix),
    )


def generate_pc_cn_connectivity(num_pc, num_cn, n_cn_per_pc=16):
    """
    Generates PC-CN connectivity.
    Each PC projects to exactly n_cn_per_pc unique CN cells.

    Args:
        num_pc: Total number of Purkinje cells.
        num_cn: Total number of Cerebellar Nuclei cells.
        n_cn_per_pc: Number of CN targets per PC.

    Returns:
        tuple: (pre_ids, post_ids)
            pre_ids: Array of presynaptic PC indices for each connection.
            post_ids: Array of postsynaptic CN indices for each connection.
    """
    pre_ids = []
    post_ids = []

    if num_cn < n_cn_per_pc:
        raise ValueError(
            f"num_cn ({num_cn}) cannot be less than n_cn_per_pc ({n_cn_per_pc})"
        )

    for pc_idx in range(num_pc):
        # Choose n_cn_per_pc unique CN targets for this PC
        connected_cns = np.random.choice(num_cn, n_cn_per_pc, replace=False)

        # Store connections
        for cn_idx in connected_cns:
            pre_ids.append(pc_idx)
            post_ids.append(cn_idx)

    print(
        f"PC->CN: Generated {len(pre_ids)} connections. Each PC -> {n_cn_per_pc} CNs."
    )
    return np.array(pre_ids, dtype=np.int32), np.array(post_ids, dtype=np.int32)


def generate_cn_io_connectivity(num_cn, num_io, n_io_per_cn=10):
    """
    Generates CN-IO connectivity.
    Each CN projects to exactly n_io_per_cn unique IO cells.

    Args:
        num_cn: Total number of Cerebellar Nuclei cells.
        num_io: Total number of Inferior Olive cells.
        n_io_per_cn: Number of IO targets per CN.

    Returns:
        tuple: (pre_ids, post_ids)
            pre_ids: Array of presynaptic CN indices for each connection.
            post_ids: Array of postsynaptic IO indices for each connection.
    """
    pre_ids = []
    post_ids = []

    if num_io < n_io_per_cn:
        raise ValueError(
            f"num_io ({num_io}) cannot be less than n_io_per_cn ({n_io_per_cn})"
        )

    for cn_idx in range(num_cn):
        # Choose n_io_per_cn unique IO targets for this CN
        connected_ios = np.random.choice(num_io, n_io_per_cn, replace=False)

        # Store connections
        for io_idx in connected_ios:
            pre_ids.append(cn_idx)
            post_ids.append(io_idx)

    print(
        f"CN->IO: Generated {len(pre_ids)} connections. Each CN -> {n_io_per_cn} IOs."
    )
    return np.array(pre_ids, dtype=np.int32), np.array(post_ids, dtype=np.int32)


def generate_io_pc_connectivity(num_io, num_pc, n_io_projecting_ratio=0.5):
    """
    Generates IO-PC connectivity.
    Each PC receives input from exactly 1 unique projecting IO cell.
    Projecting IOs are the first half of the IO population.

    Args:
        num_io: Total number of Inferior Olive cells.
        num_pc: Total number of Purkinje cells.
        n_io_projecting_ratio: Fraction of IO cells that project.

    Returns:
        tuple: (pre_ids, post_ids)
            pre_ids: Array of presynaptic IO indices for each connection.
            post_ids: Array of postsynaptic PC indices for each connection.
    """
    pre_ids = []
    post_ids = []

    num_io_projecting = int(num_io * n_io_projecting_ratio)
    if num_io_projecting == 0:
        raise ValueError("No projecting IO neurons found.")

    projecting_io_indices = np.arange(num_io_projecting)

    # Assign each PC to one randomly chosen projecting IO neuron
    pc_to_io_map = np.random.choice(projecting_io_indices, size=num_pc, replace=True)

    for pc_idx in range(num_pc):
        io_idx = pc_to_io_map[pc_idx]
        pre_ids.append(io_idx)
        post_ids.append(pc_idx)

    print(
        f"IO->PC: Generated {len(pre_ids)} connections. Each PC <- 1 projecting IO ({num_io_projecting} total)."
    )
    return np.array(pre_ids, dtype=np.int32), np.array(post_ids, dtype=np.int32)


# Note: IO-IO Gap Junction connectivity is handled within IONetwork currently.
# If needed, a function could be added here, but it requires details on
# how to implement the "connects to all other neurons" rule described.
