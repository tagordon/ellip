import jax
import jax.numpy as jnp
from bulirsch import *

@jax.jit
@jnp.vectorize
def ellipk(k):

    kc = jnp.sqrt((1 - k) * (1 + k))
    return cel(kc, 1.0, 1.0, 1.0)

@jax.jit
@jnp.vectorize
def ellipe(k):

    kc = jnp.sqrt((1 - k) * (1 + k))
    return cel(kc, 1.0, 1.0, k**2)

@jax.jit
@jnp.vectorize
def ellippi(k, n):

    kc = jnp.sqrt((1 - k) * (1 + k))
    return cel(kc, n + 1, 1.0, 1.0)

@jax.jit
@jnp.vectorize
def ellipfinc(phi, k):

    x = jnp.arctan(phi)
    kc = jnp.sqrt((1 - k) * (1 + k))
    return el1(x, kc)

@jax.jit
@jnp.vectorize
def ellipeinc(phi, k):

    x = jnp.arctan(phi)
    kc = jnp.sqrt((1 - k) * (1 + k))
    return el2(x, kc, 1.0, kc * kc)

@jax.jit
@jnp.vectorize
def ellippiinc(phi, k, n):

    x = jnp.arctan(phi)
    kc = jnp.sqrt((1 - k) * (1 + k))
    return el3(x, kc, p)