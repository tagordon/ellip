# algorithms from Carlson 1994 (https://arxiv.org/pdf/math/9409227.pdf)

import jax
from jax import numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

# relative error will be "less in magnitude than r" 
r = 1.0e-15

@jax.jit 
@jnp.vectorize
def rf(x, y, z):

    r"""JAX implementation of Carlson's :math:`R_\mathrm{F}`

    Computed using the algorithm in Carlson, 1994: https://arxiv.org/pdf/math/9409227.pdf

     Args:
       x: arraylike, real valued.
       y: arraylike, real valued.
       z: arraylike, real valued.

     Returns:
       The value of the integral :math:`R_\mathrm{F}`

     Notes:
       ``rf`` does not support complex-valued inputs.
       ``rf`` requires `jax.config.update("jax_enable_x64", True)`
    """
    
    xyz = jnp.array([x, y, z])
    A0 = jnp.sum(xyz) / 3.0
    v = jnp.max(jnp.abs(A0 - xyz))
    Q = (3 * r) ** (-1 / 6) * v

    cond = lambda s: s['f'] * Q > jnp.abs(s['An'])

    def body(s):

        xyz = s['xyz']
        lam = (
            jnp.sqrt(xyz[0]*xyz[1]) 
            + jnp.sqrt(xyz[0]*xyz[2]) 
            + jnp.sqrt(xyz[1]*xyz[2])
        )

        s['An'] = 0.25 * (s['An'] + lam)
        s['xyz'] = 0.25 * (s['xyz'] + lam)
        s['f'] = s['f'] * 0.25

        return s

    s = {'f': 1, 'An':A0, 'xyz':xyz}
    s = jax.lax.while_loop(cond, body, s)

    x = (A0 - x) / s['An'] * s['f']
    y = (A0 - y) / s['An'] * s['f']
    z = -(x + y)
    E2 = x * y - z * z
    E3 = x * y * z

    return (
        1 
        - 0.1 * E2 
        + E3 / 14 
        + E2 * E2 / 24 
        - 3 * E2 * E3 / 44
    ) / jnp.sqrt(s['An'])

@jax.jit
@jnp.vectorize
def rc(x, y):
    r"""JAX implementation of Carlson's :math:`R_\mathrm{C}`

    Computed using the algorithm in Carlson, 1994: https://arxiv.org/pdf/math/9409227.pdf

     Args:
       x: arraylike, real valued.
       y: arraylike, real valued.

     Returns:
       The value of the integral :math:`R_\mathrm{C}`

     Notes:
       ``rc`` does not support complex-valued inputs.
       ``rc`` requires `jax.config.update("jax_enable_x64", True)`
    """

    return jax.lax.cond(
        y > 0,
        rc_posy,
        rc_negy,
        x, y
    )

def rc_negy(x, y):

    return jnp.sqrt(x / (x - y)) * rc_posy(x - y, -y)

def rc_posy(x, y):

    A0 = (x + 2 * y) / 3.0
    Q = (3 * r) ** (-1 / 8) * jnp.abs(A0 - x)

    cond = lambda s: s['f'] * Q > jnp.abs(s['An'])

    def body(s):

        lam = 2 * jnp.sqrt(s['x'] * s['y']) + s['y']
        s['An'] = 0.25 * (s['An'] + lam)
        s['x'] = 0.25 * (s['x'] + lam)
        s['y'] = 0.25 * (s['y'] + lam)
        s['f'] = s['f'] * 0.25
        
        return s

    s = {'f': 1, 'An': A0, 'x': x, 'y': y}
    s = jax.lax.while_loop(cond, body, s)

    E = (y - A0) * s['f'] / s['An']

    return (
        1 
        + 0.3 * E**2 
        + E**3 / 7 
        + 3 * E**4 / 8 
        + 9 * E**5 / 22 
        + 159 * E**6 / 208 
        + 9 * E**7 / 8
    ) / jnp.sqrt(s['An'])

@jax.jit
@jnp.vectorize
def rj(x, y, z, p):
    r"""JAX implementation of Carlson's :math:`R_\mathrm{J}`

    Computed using the algorithm in Carlson, 1994: https://arxiv.org/pdf/math/9409227.pdf

     Args:
       x: arraylike, real valued.
       y: arraylike, real valued.
       z: arraylike, real valued.
       p: arraylike, real valued.

     Returns:
       The value of the integral :math:`R_\mathrm{J}`

     Notes:
       ``rj`` does not support complex-valued inputs.
       ``rj`` requires `jax.config.update("jax_enable_x64", True)`
    """

    return jax.lax.cond(
        p > 0,
        rj_posp,
        rj_negp,
        x, y, z, p
    )

def rj_negp(x, y, z, p):

    q = -p
    xyz = jnp.sort(jnp.array([x, y, z]))
    p = (z - y) * (y - x) / (y + q) + y
    A = x * z + p * q
    return (
        (p - y) * rj_posp(x, y, z, p) 
        - 3 * rf(x, y, z) 
        + 3 * jnp.sqrt(jnp.prod(xyz) / A) * rc(A, p * q)
    ) / (y + q)

def rj_posp(x, y, z, p):

    xyzp = jnp.array([x, y, z, p])
    A0 = (x + y + z + 2 * p) * 0.2
    delta = jnp.prod(p - xyzp[:-1])
    v = jnp.max(jnp.abs(A0 - xyzp))
    Q = (r / 4) ** (-1 / 6) * v

    cond = lambda s: s['f'] * Q > jnp.abs(s['An'])

    def body(s):

        xyzp = s['xyzp']
        lam = (
            jnp.sqrt(xyzp[0]*xyzp[1]) 
            + jnp.sqrt(xyzp[0]*xyzp[2]) 
            + jnp.sqrt(xyzp[1]*xyzp[2])
        )

        s['An'] = 0.25 * (s['An'] + lam)
        d = jnp.prod(jnp.sqrt(xyzp[3]) + jnp.sqrt(xyzp[:-1]))
        e = s['f'] ** 3 * delta / d**2
        s['t'] = s['t'] + s['f'] * rc(1, 1 + e) / d
        s['xyzp'] = 0.25 * (xyzp + lam)
        s['f'] = s['f'] * 0.25

        return s

    s = {'f': 1, 'An': A0, 'xyzp': xyzp, 't': 0}
    s = jax.lax.while_loop(cond, body, s)

    x = (A0 - x) * s['f'] / s['An']
    y = (A0 - y) * s['f'] / s['An']
    z = (A0 - z) * s['f'] / s['An']
    p = -(x + y + z) * 0.5

    E2 = x * y + x * z + y * z - 3 * p * p
    E3 = x * y * z + 2 * E2 * p + 4 * p**3
    E4 = (2 * x * y * z + E2 * p + 3 * p**3) * p
    E5 = x * y * z * p * p

    return s['f'] * (
        1 
        - 3 * E2 / 14 
        + E3 / 6 
        + 9 * E2**2 / 88 
        - 3 * E4 / 22 
        - 9 * E2 * E3 / 52 
        + 3 * E5 / 26
    ) * s['An']**-1.5 + 6 * s['t']

@jax.jit 
@jnp.vectorize
def rd(x, y, z):
    r"""JAX implementation of Carlson's :math:`R_\mathrm{D}`

    Computed using the algorithm in Carlson, 1994: https://arxiv.org/pdf/math/9409227.pdf

     Args:
       x: arraylike, real valued.
       y: arraylike, real valued.
       z: arraylike, real valued.

     Returns:
       The value of the integral :math:`R_\mathrm{D}`

     Notes:
       ``rd`` does not support complex-valued inputs.
       ``rd`` requires `jax.config.update("jax_enable_x64", True)`
    """

    xyz = jnp.array([x, y, z])
    A0 = 0.2 * (x + y + 3 * z)
    v = jnp.max(jnp.abs(A0 - xyz))
    Q = (0.25 * r) ** (-1 / 6) * v

    cond = lambda s: s['f'] * Q > jnp.abs(s['An'])

    def body(s):

        xyz = s['xyz']
        lam = (
            jnp.sqrt(xyz[0]*xyz[1]) 
            + jnp.sqrt(xyz[0]*xyz[2]) 
            + jnp.sqrt(xyz[1]*xyz[2])
        )

        s['An'] = 0.25 * (s['An'] + lam)
        s['t'] = s['t'] + s['f'] / (jnp.sqrt(xyz[2]) * (xyz[2] + lam))
        s['xyz'] = 0.25 * (xyz + lam)
        s['f'] = s['f'] * 0.25

        return s

    s = {'f': 1, 'An': A0, 'xyz': xyz, 't': 0}
    s = jax.lax.while_loop(cond, body, s)

    x = (A0 - x) * s['f'] / s['An']
    y = (A0 - y) * s['f'] / s['An']
    z = -(x + y) / 3

    E2 = x * y - 6 * z * z
    E3 = (3 * x * y - 8 * z * z) * z
    E4 = 3 * (x * y - z * z) * z * z
    E5 = x * y * z**3

    return s['f'] * (
        1 
        - 3 * E2 / 14 
        + E3 / 6 
        + 9 * E2 **2 / 88 
        - 3 * E4 / 22 
        - 9 * E2 * E3 / 52 
        + 3 * E5 / 26
    ) * s['An']**-1.5 + 3 * s['t']