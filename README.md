# ellip

JAX implementations of various elliptic integrals. 

Note: Currently only compatible with forward-mode autodifferentiation. 

## Bulirsch integrals

* $el1(x, k_c)$: incomplete elliptic integral of the first kind
* $el2(x, k_c, a, b)$: incomplete elliptic integral of the second kind 
* $el3(x, k_c, p)$: incomplete elliptic integral of the third kind
* $cel(k_c, p, a, b)$: Generalized complete elliptic integral

## Carlson integrals

* $R_\mathrm{F}(x, y, z)$
* $R_\mathrm{C}(x, y)$
* $R_\mathrm{J}(x, y, z, p)$
* $R_\mathrm{D}(x, y, z)$

## Legendre forms 

* $K(k)$: complete elliptic integral of the first kind
* $E(k)$: complete elliptic integral of the second kind
* $\Pi(n, k)$: complete elliptic integral of the third kind

* $F(\phi, k)$: incomplete elliptic integral of the first kind
* $E(\phi, k)$: incomplete elliptic integral of the second kind
* $\Pi(\phi, k, n)$: incomplete elliptic integral of the third kind

Note: The Legendre forms are computed directly from the Bulirsch integrals using the relations found in [1]. For most use cases, it will be more efficient to use the Bulirsch integrals directly rather than the Legendre forms. 

## References
[1] [Bulirsch, 1969b](https://doi.org/10.1007/BF02165405)

[2] [Carlson, 1994](https://arxiv.org/pdf/math/9409227)
