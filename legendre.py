import jax
import jax.numpy as jnp
from bulirsch import *

@jax.jit
@jnp.vectorize
def ellipk(k):
    r"""JAX implementation of the complete elliptic integral of the first kind 

    .. math::

        \[K\left(k\right)=\int_{0}^{\pi/2}\frac{\,\mathrm{d}\theta}{\sqrt{1-k^{2}{%
\sin}^{2}\theta}}]

     Args:
       k: arraylike, real valued.

     Returns:
       The value of the complete elliptic integral of the first kind, :math:`K(k)`

     Notes:
       ``ellipk`` does not support complex-valued inputs.
       ``ellipk`` requires `jax.config.update("jax_enable_x64", True)`
    """

    kc = jnp.sqrt((1 - k) * (1 + k))
    return cel(kc, 1.0, 1.0, 1.0)

@jax.jit
@jnp.vectorize
def ellipe(k):
    r"""JAX implementation of the complete elliptic integral of the second kind 

    .. math::

        \[E\left(k\right)=\int_{0}^{\pi/2}\sqrt{1-k^{2}{\sin}^{2}\theta}\,\mathrm{d}%
\theta\]

     Args:
       k: arraylike, real valued.

     Returns:
       The value of the complete elliptic integral of the second kind, :math:`E(k)`

     Notes:
       ``ellipe`` does not support complex-valued inputs.
       ``ellipe`` requires `jax.config.update("jax_enable_x64", True)`
    """

    kc = jnp.sqrt((1 - k) * (1 + k))
    return cel(kc, 1.0, 1.0, k**2)

@jax.jit
@jnp.vectorize
def ellippi(n, k):
    r"""JAX implementation of the complete elliptic integral of the third kind 

    .. math::

        \[\Pi\left(n, k\right)=\int_{0}^{\pi/2}\frac{\,\mathrm{d}\theta}{%
\sqrt{1-k^{2}{\sin}^{2}\theta}(1+n{\sin}^{2}\theta)}]

     Args:
       n: arraylike, real valued.
       k: arraylike, real valued.

     Returns:
       The value of the complete elliptic integral of the third kind, :math:`\Pi(n, k)`

     Notes:
       ``ellippi`` does not support complex-valued inputs.
       ``ellippi`` requires `jax.config.update("jax_enable_x64", True)`
    """

    kc = jnp.sqrt((1 - k) * (1 + k))
    return cel(kc, n + 1, 1.0, 1.0)

@jax.jit
@jnp.vectorize
def ellipfinc(phi, k):
    r"""JAX implementation of the incomplete elliptic integral of the first kind 

    .. math::

        \[F\left(\phi,k\right)=\int_{0}^{\phi}\frac{\,\mathrm{d}\theta}{\sqrt{1-k^{2}{%
\sin}^{2}\theta}}]

     Args:
       phi: arraylike, real valued.
       k: arraylike, real valued.

     Returns:
       The value of the complete elliptic integral of the first kind, :math:`F(\phi, k)`

     Notes:
       ``ellipfinc`` does not support complex-valued inputs.
       ``ellipfinc`` requires `jax.config.update("jax_enable_x64", True)`
    """

    x = jnp.arctan(phi)
    kc = jnp.sqrt((1 - k) * (1 + k))
    return el1(x, kc)

@jax.jit
@jnp.vectorize
def ellipeinc(phi, k):
    r"""JAX implementation of the incomplete elliptic integral of the second kind 

    .. math::

        \[E\left(\phi,k\right)=\int_{0}^{\phi}\sqrt{1-k^{2}{\sin}^{2}\theta}\,\mathrm{d}%
\theta\\]

     Args:
       phi: arraylike, real valued.
       k: arraylike, real valued.

     Returns:
       The value of the complete elliptic integral of the second kind, :math:`E(\phi, k)`

     Notes:
       ``ellipeinc`` does not support complex-valued inputs.
       ``ellipeinc`` requires `jax.config.update("jax_enable_x64", True)`
    """

    x = jnp.arctan(phi)
    kc = jnp.sqrt((1 - k) * (1 + k))
    return el2(x, kc, 1.0, kc * kc)

@jax.jit
@jnp.vectorize
def ellippiinc(phi, k, n):
    r"""JAX implementation of the incomplete elliptic integral of the third kind 

    .. math::

        \[E\left(\phi,k\right)=\int_{0}^{\phi}\sqrt{1-k^{2}{\sin}^{2}\theta}\,\mathrm{d}%
\theta\\]

     Args:
       phi: arraylike, real valued.
       k: arraylike, real valued.
       n: arraylike, real valued.

     Returns:
       The value of the complete elliptic integral of the third kind, :math:`\Pi(\phi, k)'

     Notes:
       ``ellippiinc`` does not support complex-valued inputs.
       ``ellippiinc`` requires `jax.config.update("jax_enable_x64", True)`
    """

    x = jnp.arctan(phi)
    kc = jnp.sqrt((1 - k) * (1 + k))
    return el3(x, kc, p)