"""
The combined galaxy potential used throughout Phoenix: an NFW halo + a
Miyamoto-Nagai disk + a Plummer bulge.

`total_potential_raw` takes the seven physical parameters positionally, which is
the signature `compute_poisson_penalty` expects (potential_fn(x, y, z, *params)).
It lives here so the disk+halo+bulge sum is defined once and imported wherever it
is needed (optimization pipeline, observables, ...) instead of being repeated.
"""
from phoenix.potentials.potentials import (
    nfw_potential,
    plummer_potential,
    miyamoto_nagai_potential,
)


def total_potential_raw(x, y, z, M_halo, a_halo, M_disk, a_disk, b_disk, M_bulge, a_bulge):
    """Combined NFW halo + Miyamoto-Nagai disk + Plummer bulge potential.

    Parameters are passed positionally (not via a dict) so this can be handed
    directly to routines that call ``potential_fn(x, y, z, *params)``.
    """
    return (nfw_potential(x, y, z, M_halo, a_halo) +
            miyamoto_nagai_potential(x, y, z, M_disk, a_disk, b_disk) +
            plummer_potential(x, y, z, M_bulge, a_bulge))
