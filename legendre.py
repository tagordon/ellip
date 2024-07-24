import jax
import jax.numpy as jnp
from carlson import *

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

    kc2 = 1 - k**2
    return rf(0.0, kc2, 1.0)

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

    kc2 = 1 - k**2
    return kc2 * (rd(0.0, kc2, 1.0) + rd(0.0, 1.0, kc2)) / 3.0

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

    kc2 = 1 - k**2
    return n * rj(0.0, kc2, 1.0, 1 - n) / 3.0 + ellipk(k)

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

    c = 1.0 / jnp.sin(phi)**2
    return rf(c - 1, c - k**2, c)
    

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

    c = 1.0 / jnp.sin(phi)**2
    return rf(c - 1, c - k**2, c) - k**2 * rd(c - 1, c - k**2, c) / 3.0

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

    c = 1.0 / jnp.sin(phi)**2
    return n * rj(c - 1, c - k**2, c, c - n) / 3.0 + ellipfinc(phi, k)