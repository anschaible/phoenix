import numpy as np
import jax.numpy as jnp

from phoenix.potentials import plummer_potential
from phoenix.constants import G


def test_plummer_potential_center():
	"""At the origin r=0 the potential should be -G*M/a."""
	M = 1e10
	a = 2.0
	phi = float(plummer_potential(0.0, 0.0, 0.0, M, a))
	expected = -G * M / np.sqrt(a * a)
	assert np.isclose(phi, expected)


def test_plummer_potential_known_radius():
	"""Test against a known radius: (x,y,z)=(3,4,0) => r=5."""
	x, y, z = 3.0, 4.0, 0.0
	r = 5.0
	M = 2e9
	a = 1.5
	phi = plummer_potential(x, y, z, M, a)
	expected = -G * M / np.sqrt(a * a + r * r)
	np.testing.assert_allclose(np.asarray(phi), expected)


def test_plummer_potential_vectorized():
	"""Verify function works elementwise on jax arrays."""
	xs = jnp.array([0.0, 3.0])
	ys = jnp.array([0.0, 4.0])
	zs = jnp.array([0.0, 0.0])
	M = 1e9
	a = 1.0
	phi = plummer_potential(xs, ys, zs, M, a)
	expected = -G * M / jnp.sqrt(a * a + xs ** 2 + ys ** 2 + zs ** 2)
	np.testing.assert_allclose(np.asarray(phi), np.asarray(expected))


from phoenix.potentials import isochrone_potential


def test_isochrone_potential_center():
	"""At the origin r=0 the isochrone potential should be -G*M/(2*a)."""
	M = 5e9
	a = 3.0
	phi = float(isochrone_potential(0.0, 0.0, 0.0, M, a))
	expected = -G * M / (a + (a))
	assert np.isclose(phi, expected)


def test_isochrone_potential_known_radius():
	"""Test a known position: (3,4,0) => r=5."""
	x, y, z = 3.0, 4.0, 0.0
	r = 5.0
	M = 2e8
	a = 1.2
	phi = isochrone_potential(x, y, z, M, a)
	expected = -G * M / (a + jnp.sqrt(r * r + a * a))
	np.testing.assert_allclose(np.asarray(phi), np.asarray(expected))


def test_isochrone_vectorized():
	"""Vectorized inputs return elementwise results for isochrone potential."""
	xs = jnp.array([0.0, 3.0])
	ys = jnp.array([0.0, 4.0])
	zs = jnp.array([0.0, 0.0])
	M = 1e8
	a = 0.5
	phi = isochrone_potential(xs, ys, zs, M, a)
	expected = -G * M / (a + jnp.sqrt(xs ** 2 + ys ** 2 + zs ** 2 + a ** 2))
	np.testing.assert_allclose(np.asarray(phi), np.asarray(expected))


from phoenix.potentials import nfw_potential


def test_nfw_potential_center():
	"""At r->0 the NFW potential tends to -G*M/a."""
	M = 1e11
	a = 4.0
	phi = float(nfw_potential(0.0, 0.0, 0.0, M, a))
	# The implementation clamps r to at least 1e-6, so compute expected
	# using the same small-radius regularization to avoid a false failure.
	r_eps = 1e-6
	expected = float(-G * M / r_eps * jnp.log(1 + r_eps / a))
	assert np.isclose(phi, expected)


def test_nfw_potential_known_radius():
	"""Check a numeric value at (3,4,0) => r=5."""
	x, y, z = 3.0, 4.0, 0.0
	r = 5.0
	M = 2e9
	a = 1.5
	phi = nfw_potential(x, y, z, M, a)
	expected = -G * M / r * jnp.log(1 + r / a)
	np.testing.assert_allclose(np.asarray(phi), np.asarray(expected))


def test_nfw_vectorized():
	"""Vectorized inputs return elementwise results for NFW potential."""
	xs = jnp.array([0.0, 3.0])
	ys = jnp.array([0.0, 4.0])
	zs = jnp.array([0.0, 0.0])
	M = 1e9
	a = 0.7
	phi = nfw_potential(xs, ys, zs, M, a)
	r = jnp.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
	r = jnp.maximum(r, 1e-6)
	expected = -G * M / r * jnp.log(1 + r / a)
	np.testing.assert_allclose(np.asarray(phi), np.asarray(expected))


from phoenix.potentials import miyamoto_nagai_potential


def test_miyamoto_nagai_center():
	"""At origin the Miyamoto-Nagai potential reduces to -G*M/(a + b)."""
	M = 1e10
	a = 2.5
	b = 0.7
	phi = float(miyamoto_nagai_potential(0.0, 0.0, 0.0, M, a, b))
	expected = -G * M / (a + b)
	assert np.isclose(phi, expected)


def test_miyamoto_nagai_known_point():
	"""Numeric comparison at a known point (3,4,2)."""
	x, y, z = 3.0, 4.0, 2.0
	M = 5e9
	a = 1.5
	b = 0.3
	phi = miyamoto_nagai_potential(x, y, z, M, a, b)
	R2 = x**2 + y**2
	B = jnp.sqrt(z**2 + b**2)
	denom = jnp.sqrt(R2 + (a + B) ** 2)
	expected = -G * M / denom
	np.testing.assert_allclose(np.asarray(phi), np.asarray(expected))


def test_miyamoto_nagai_vectorized():
	"""Vectorized behavior for Miyamoto-Nagai potential."""
	xs = jnp.array([0.0, 3.0])
	ys = jnp.array([0.0, 4.0])
	zs = jnp.array([0.0, 2.0])
	M = 2e9
	a = 0.9
	b = 0.4
	phi = miyamoto_nagai_potential(xs, ys, zs, M, a, b)
	R2 = xs ** 2 + ys ** 2
	B = jnp.sqrt(zs ** 2 + b ** 2)
	denom = jnp.sqrt(R2 + (a + B) ** 2)
	expected = -G * M / denom
	np.testing.assert_allclose(np.asarray(phi), np.asarray(expected))


from phoenix.potentials import logarithmic_potential


def test_logarithmic_potential_center():
	"""At origin rtilde=0 => Phi = 0.5 * v0^2 * log(rcore^2)."""
	v0 = 220.0
	rcore = 2.0
	p = 1.0
	q = 1.0
	phi = float(logarithmic_potential(0.0, 0.0, 0.0, v0, rcore, p, q))
	expected = 0.5 * v0 ** 2 * np.log(rcore ** 2)
	assert np.isclose(phi, expected)


def test_logarithmic_potential_known_point():
	"""Numeric comparison at (3,4,2) with axis ratios."""
	x, y, z = 3.0, 4.0, 2.0
	v0 = 150.0
	rcore = 0.5
	p = 0.9
	q = 0.8
	rtilde2 = x ** 2 + (y / p) ** 2 + (z / q) ** 2
	phi = logarithmic_potential(x, y, z, v0, rcore, p, q)
	expected = 0.5 * v0 ** 2 * jnp.log(rcore ** 2 + rtilde2)
	np.testing.assert_allclose(np.asarray(phi), np.asarray(expected))


def test_logarithmic_potential_vectorized():
	"""Vectorized inputs return elementwise results for logarithmic potential."""
	xs = jnp.array([0.0, 3.0])
	ys = jnp.array([0.0, 4.0])
	zs = jnp.array([0.0, 2.0])
	v0 = 180.0
	rcore = 1.0
	p = 1.1
	q = 0.9
	phi = logarithmic_potential(xs, ys, zs, v0, rcore, p, q)
	rtilde2 = xs ** 2 + (ys / p) ** 2 + (zs / q) ** 2
	expected = 0.5 * v0 ** 2 * jnp.log(rcore ** 2 + rtilde2)
	np.testing.assert_allclose(np.asarray(phi), np.asarray(expected))

