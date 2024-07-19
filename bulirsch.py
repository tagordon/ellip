import jax
import jax.numpy as jnp

@jax.custom_jvp
@jax.jit
@jnp.vectorize
def el1(x, kc):
    r"""JAX implementation of Bulirsch's el1 integral

    Computed using the algorithm in Bulirsch, 1969b: https://doi.org/10.1007/BF02165405 

    .. math::

        \[\operatorname{el1}\left(x,k_{c}\right)=\int_{0}^{\operatorname{arctan}x}\frac{%
1}{\sqrt{{\cos}^{2}\theta+k_{c}^{2}{\sin}^{2}\theta}}\,\mathrm{d}\theta,\]

     Args:
       x: arraylike, real valued.
       kc: arraylike, real valued.

     Returns:
       The value of the integral el1

     Notes:
       ``el1`` does not support complex-valued inputs.
       ``el1`` requires `jax.config.update("jax_enable_x64", True)`
    """

    D = 15.0
    ca = 10.0**(-D / 2.0)
    cb = 10.0**(-D + 2.0)

    y = jnp.abs(1.0 / x)
    kc = jnp.abs(kc)
    m = 1.0

    def cont():

        s = {
            'e': m * kc, 
            'g': m, 'm': kc + m, 
            'y': -(m * kc / y) + y, 
            'kc': kc, 'l': 0.0
        }
        s['y'] = jax.lax.cond(
            s['y'] == 0, 
            lambda: jnp.sqrt(s['e']) * cb, 
            lambda: s['y']
        )

        def cond_fun(s):
            
            return jnp.abs(s['g'] - s['kc']) > ca * s['g']

        def body_fun(s):
            
            s['kc'] = 2 * jnp.sqrt(s['e'])
            s['l'] = 2 * s['l']
            s['l'] = s['l'] + (s['y'] < 0)

            s['e'] = s['m'] * s['kc']
            s['g'] = s['m']
            s['m'] = s['kc'] + s['m']
            s['y'] = - (s['e'] / s['y']) + s['y']
            s['y'] = jax.lax.cond(
                s['y'] == 0, 
                lambda: jnp.sqrt(s['e']) * cb, 
                lambda: s['y']
            )

            return s

        s = jax.lax.while_loop(cond_fun, body_fun, s)
        s['l'] = s['l'] + (s['y'] < 0)
        s['e'] = (jnp.arctan(s['m'] / s['y']) + jnp.pi * s['l']) / s['m']
        s['e'] = -(2 * (x < 0) - 1) * s['e']

        return s['e']

    e = jax.lax.cond(
        kc == 0, 
        lambda: jnp.log(x + 1 / jnp.cos(jnp.arctan(x))), 
        cont
    )

    return e

@jax.custom_jvp
@jax.jit 
@jnp.vectorize
def el2(x, kc, a, b):
    r"""JAX implementation of Bulirsch's el2 integral

    Computed using the algorithm in Bulirsch, 1969b: https://doi.org/10.1007/BF02165405 

    .. math::

       \[\operatorname{el2}\left(x,k_{c},a,b\right)=\int_{0}^{\operatorname{arctan}x}%
\frac{a+b{\tan}^{2}\theta}{\sqrt{(1+{\tan}^{2}\theta)(1+k_{c}^{2}{\tan}^{2}%
\theta)}}\,\mathrm{d}\theta.\]

     Args:
       x: arraylike, real valued.
       kc: arraylike, real valued.
       a: arraylike, real valued.
       b: arraylike, real valued.

     Returns:
       The value of the integral el2

     Notes:
       ``el2`` does not support complex-valued inputs.
       ``el2`` requires `jax.config.update("jax_enable_x64", True)`
    """

    D = 15.0
    ca = 10**(-D / 2.0)
    cb = 10**(-D + 2.0)

    c = x**2
    dd = c + 1.0
    p = jnp.sqrt((1.0 + kc**2 * c) / dd)
    dd = x / dd
    c = dd * 0.5 / p
    z = a - b
    ik = a
    a = (b + a) * 0.5
    y = jnp.abs(1 / x)
    f = 0.0
    kc = jnp.abs(kc)
    m = 1.0

    def cont():

        s = {
            'l': 0.0,
            'b': ik * kc + b,
            'e': m * kc,
            'g': m * kc / p,
            'dd': f * m * kc / p + dd,
            'f': c,
            'ik': a,
            'p': m * kc / p + p,
            'c': ((f * m * kc / p + dd) / (m * kc / p + p) + c) * 0.5,
            'g': m,
            'm': kc + m,
            'a': ((ik * kc + b) / (kc + m) + a) * 0.5,
            'y': - (m * kc / y) + y,
            'kc': kc
        }
        
        s['y'] = jax.lax.cond(
            s['y'] == 0, 
            lambda: jnp.sqrt(s['e']) * cb, 
            lambda: s['y']
        )

        def cond_fun(s):
            
            return jnp.abs(s['g'] - s['kc']) > ca * s['g']

        def body_fun(s):

            s['kc'] = 2 * jnp.sqrt(s['e'])
            s['l'] = 2 * s['l']
            s['l'] = s['l'] + (s['y'] < 0)

            s['b'] = s['ik'] * s['kc'] + s['b']
            s['e'] = s['m'] * s['kc']
            s['g'] = s['e'] / s['p']
            s['dd'] = s['f'] * s['g'] + s['dd']
            s['f'] = s['c']
            s['ik'] = s['a']
            s['p'] = s['g'] + s['p']
            s['c'] = (s['dd'] / s['p'] + s['c']) * 0.5
            s['g'] = s['m']
            s['m'] = s['kc'] + s['m']
            s['a'] = (s['b'] / s['m'] + s['a']) * 0.5
            s['y'] = - (s['e'] / s['y']) + s['y']
            s['y'] = jax.lax.cond(
                s['y'] == 0, 
                lambda: jnp.sqrt(s['e']) * cb, 
                lambda: s['y']
            )

            return s

        s = jax.lax.while_loop(cond_fun, body_fun, s)
        s['l'] = s['l'] + (s['y'] < 0)
        s['e'] = (jnp.arctan(s['m'] / s['y']) + jnp.pi * s['l']) * s['a'] / s['m']
        # this line is slightly different from the algorithm in gefera (see ellip.f90 line 127)
        # but it matches the numerical integral... 
        s['e'] = -(2 * (x < 0) - 1) * s['e'] + s['c'] * z

        return s['e']

    e = jax.lax.cond(
        kc == 0, 
        lambda: jnp.sin(jnp.arctan(x)), 
        cont
    )

    return e

@jax.custom_jvp
@jax.jit 
@jnp.vectorize
def el3(x, kc, p):
    r"""JAX implementation of Bulirsch's el3 integral

    Computed using the algorithm in Bulirsch, 1969b: https://doi.org/10.1007/BF02165405 

    .. math::

       \[\operatorname{el3}\left(x,k_{c},p\right)=\int_{0}^{\operatorname{arctan}x}%
\frac{\,\mathrm{d}\theta}{({\cos}^{2}\theta+p{\sin}^{2}\theta)\sqrt{{\cos}^{2}%
\theta+k_{c}^{2}{\sin}^{2}\theta}}=\Pi\left(\operatorname{arctan}x,1-p,k\right),\]

     Args:
       x: arraylike, real valued.
       kc: arraylike, real valued.
       p: arraylike, real valued.

     Returns:
       The value of the integral el3

     Notes:
       ``el3`` does not support complex-valued inputs.
       ``el3`` requires `jax.config.update("jax_enable_x64", True)`
    """

    cD = 15
    ca = 10**(-cD / 2.0)
    cb = 10**(-cD + 2.0)
    ND = 10

    S = {
        'am': 0.0,
        'ap': 0.0,
        'c': 0.0,
        'd': 0.0,
        'de': 0.0,
        'f': p * x * x,
        'fa': 0.0,
        'g': 0.0,
        'h': 1.0 + x * x,
        'hh': x * x,
        'p1': 0.0,
        'pm': 0.0,
        'pz': 0.0,
        'q': 0.0,
        'r': jnp.abs(p),
        's': jax.lax.cond(
            kc == 0.0, 
            lambda: ca / (1.0 + jnp.abs(x)), 
            lambda: kc
        ),
        't': 0.0,
        'u': 0.0,
        'w': 0.0,
        'y': 0.0,
        'ye': 0.0,
        'z': jnp.abs(p * x * x),
        'zd': 0.0,
        'l': 0.0,
        'm': 0.0,
        'n': 0.0,
        'bo': 0.0,
        'bk': 0,
        'k': 0.0
    }
    
    S['t'] = S['s'] * S['s']
    S['pm'] = S['t'] * 0.5
    S['e'] = S['hh'] * S['t']
    cond = (S['e'] < 1.0) & (S['z'] < 0.1) & (S['t'] < 1.0) & (S['r'] < 1.0)

    def continue4(S):

        S['p1'] = jax.lax.cond(p == 0.0, lambda: cb / S['hh'], lambda: p)
        S['s'] = jnp.abs(S['s'])
        S['y'] = jnp.abs(x)
        S['g'] = S['p1'] - 1.0
        S['g'] = S['g'] + (S['g'] == 0.0) * cb
        S['f'] = S['p1'] - S['t']
        S['f'] = S['f'] + (S['f'] == 0.0) * cb * S['t']
        S['am'] = 1.0 - S['t']
        S['ap'] = 1.0 + S['e']
        S['r'] = S['p1'] * S['h']
        S['fa'] = S['g'] / (S['f'] * S['p1'])
        S['bo'] = S['fa'] > 0.0
        S['fa'] = jnp.abs(S['fa'])
        S['pz'] = jnp.abs(S['g'] * S['f'])
        S['de'] = jnp.sqrt(S['pz'])
        S['q'] = jnp.sqrt(jnp.abs(S['p1']))
        S['pm'] = jax.lax.cond(S['pm'] > 0.5, lambda: 0.5, lambda: S['pm'])
        S['pm'] = S['p1'] - S['pm']

        def pm_ge_0(S):
            
            S['u'] = jnp.sqrt(S['r'] * S['ap'])
            S['v'] = S['y'] * S['de']
            S['v'] = -(2 * (S['g'] < 0) - 1) * S['v']
            S['d'] = 1.0 / S['q']
            S['c'] = 1.0
            return S

        def pm_leq_0(S):

            S['u'] = jnp.sqrt(S['h'] * S['ap'] * S['pz'])
            S['ye'] = S['y'] * S['q']
            S['v'] = S['am'] * S['ye']
            S['q'] = -S['de'] / S['g']
            S['d'] = -S['am'] / S['de']
            S['c'] = 0.0
            S['pz'] = S['ap'] - S['r']
            
            return S

        def bo_true(S):

            S['r'] = S['v'] / S['u']
            S['z'] = 1.0
            S['k'] = 1.0

            def pm_lt_0(S):
                
                S['h'] = S['y'] * jnp.sqrt(S['h'] / (S['ap'] * S['fa']))
                S['h'] = 1.0 / S['h'] - S['h']
                S['z'] = S['h'] - 2 * S['r']
                S['r'] = 2.0 + S['r'] * S['h']
                S['r'] = S['r'] + (S['r'] == 0.0) * cb
                S['z'] = S['z'] + (S['z'] == 0.0) * S['h'] * cb
                S['r'] = S['r'] / S['z']
                S['z'] = S['r']
                S['w'] = S['pz']
                
                return S

            S = jax.lax.cond(S['pm'] < 0.0, pm_lt_0, lambda S: S, S)
            S['u'] = S['u'] / S['w']
            S['v'] = S['v'] / S['w']

            return S

        def bo_false(S):

            S['t'] = S['u'] + jnp.abs(S['v'])
            S['bk'] = 1

            def p1_lt_0(S):
                S['de'] = S['v'] / S['pz']
                S['ye'] = S['u'] / S['ye']
                S['ye'] = S['ye'] * S['ye']
                S['u'] = S['t'] / S['pz']
                S['v'] = (-S['f'] - S['g'] * S['e']) / S['t']
                S['t'] = S['pz'] * jnp.abs(S['w'])
                S['z'] = (S['hh'] * S['r'] * S['f'] - S['g'] * S['ap'] + S['ye']) / S['t']
                S['ye'] = S['ye'] / S['t']
                return S

            def p1_geq_0(S):
                S['de'] = S['v'] / S['w']
                S['ye'] = 0.0
                S['u'] = (S['e'] + S['p1']) / S['t']
                S['v'] = S['t'] / S['w']
                S['z'] = 1.0

                return S

            def s_gt_1(S):
                S['h'] = S['u']
                S['u'] = S['v']
                S['v'] = S['h']

                return S

            S = jax.lax.cond(S['p1'] < 0.0, p1_lt_0, p1_geq_0, S)
            S = jax.lax.cond(S['s'] > 1.0, s_gt_1, lambda S: S, S)

            return S
            
        
        S = jax.lax.cond(S['pm'] > 0.0, pm_ge_0, pm_leq_0, S)
        S = jax.lax.cond(S['bo'], bo_true, bo_false, S)

        def goto3(S):
    
            S['y'] = S['y'] - S['e'] / S['y']
            S['y'] = S['y'] + (S['y'] == 0.0) * jnp.sqrt(S['e']) * cb
            S['f'] = S['c']
            S['c'] = S['d'] / S['q'] + S['c']
            S['g'] = S['e'] / S['q']
            S['d'] = S['f'] * S['g'] + S['d']
            S['d'] = 2 * S['d']
            S['q'] = S['g'] + S['q']
            S['g'] = S['t']
            S['t'] = S['s'] + S['t']
            S['n'] = 2 * S['n']
            S['m'] = 2 * S['m']

            def bk_true(S):

                S['de'] = S['de'] / S['u']
                S['ye'] = S['ye'] * (S['h'] + 1.0 / S['h']) + S['de'] * (1.0 + S['r'])
                S['de'] = S['de'] * (S['u'] - S['hh'])
                S['bk'] = (jnp.abs(S['ye']) < 1.0) * 1

                return S

            def bk_false(S):

                # this function is a problem because 
                # jax doesn't differentiate the mantissa. 
                # handled with custom vjp. 
                S['z'], S['k'] = jnp.frexp(S['z'])
                S['k'] = 1.0 * S['k']
                S['m'] = S['m'] + S['k']
                
                return S
                
            def bo_true_2(S):
                
                S['m'] = S['m'] + (S['z'] < 0) * S['k']
                S['k'] = jnp.sign(S['r'])
                S['h'] = S['e'] / (S['u'] * S['u'] + S['v'] * S['v'])
                S['u'] = S['u'] * (1.0 + S['h'])
                S['v'] = S['v'] * (1.0 - S['h'])
                
                return S

            def bo_false_2(S):

                S['r'] = S['u'] / S['v']
                S['h'] = S['z'] * S['r']
                S['z'] = S['h'] * S['z']
                S['hh'] = S['e'] / S['v']

                S = jax.lax.cond(S['bk'], bk_true, bk_false, S)

                return S

            S = jax.lax.cond(S['bo'], bo_true_2, bo_false_2, S)
            return S

        def cond_func(S):

            return jnp.abs(S['g'] - S['s']) > ca * S['g']

        def body_func(S):

            def bo_true_3(S):

                S['g'] = (1.0 / S['r'] - S['r']) * 0.5
                S['hh'] = S['u'] + S['v'] * S['g']
                S['h'] = S['g'] * S['u'] - S['v']
                S['hh'] = S['hh'] + (S['hh'] == 0.0) * S['u'] * cb
                S['z'] = S['r'] * S['h']
                S['r'] = S['hh'] / S['h']

                return S

            def bo_false_3(S):

                S['u'] = S['u'] + S['e'] / S['u']
                S['v'] = S['v'] + S['hh']

                return S

            S = jax.lax.cond(S['bo'], bo_true_3, bo_false_3, S)
            S['s'] = jnp.sqrt(S['e'])
            S['s'] = 2 * S['s']
            S['e'] = S['s'] * S['t']
            S['l'] = 2 * S['l']
            S['l'] = S['l'] + (S['y'] < 0.0)

            S = goto3(S)
            return S

        S['y'] = 1.0 / S['y']
        S['e'] = S['s']
        S['n'] = 1.0
        S['t'] = 1.0
        S['m'] = 0.0
        S['l'] = 0.0

        S = goto3(S)
        S = jax.lax.while_loop(cond_func, body_func, S)
        S['l'] = S['l'] + (S['y'] < 0.0) 
        S['e'] = jnp.arctan(S['t'] / S['y']) + jnp.pi * S['l']
        S['e'] = S['e'] * (S['c'] * S['t'] + S['d']) / (S['t'] * (S['t'] + S['q']))

        def bo_true_4(S):
            
            S['h'] = S['v'] / (S['t'] + S['u'])
            S['z'] = 1.0 - S['r'] * S['h']
            S['h'] = S['r'] + S['h']
            S['z'] = S['z'] + cb * (S['z'] == 0.0)
            S['m'] = S['m'] + (S['z'] < 0.0) * jnp.sign(S['h'])
            S['s'] = jnp.arctan(S['h'] / S['z']) + S['m'] * jnp.pi

            return S

        def bo_false_4(S):

            S['s'] = jax.lax.cond(
                S['bk'], 
                lambda: jnp.arcsinh(S['ye']), 
                lambda: jnp.log(S['z']) + S['m'] * jnp.log(2)
            )
            S['s'] = 0.5 * S['s']
            return S

        S = jax.lax.cond(S['bo'], bo_true_4, bo_false_4, S)
        S['e'] = (S['e'] + jnp.sqrt(S['fa']) * S['s']) / S['n']
        S['e'] =  -(2 * (x < 0) - 1) * S['e']
        return S['e']

    def goto1(S):

        rb = jnp.zeros(ND)
        ra = jnp.zeros(ND)
        rr = jnp.zeros(ND)
        for k in range(2, ND + 1):
            rb = rb.at[k].set(0.5 / k)
            ra = ra.at[k].set(1.0 - rb[k])
        zd = 0.5 / (ND + 1.0)
        s = p + S['pm']
        for k in range(2, ND + 1):
            rr = rr.at[k].set(s)
            S['pm'] = S['pm'] * S['t'] * ra[k]
            s = s * p + S['pm']
        u = s * zd
        s = u
        bo = 0
        for k in range(ND, 1, -1):
            u = u + (rr[k] - u) * rb[k]
            bo = not bo
            v = -(2 * bo - 1) * u
            s = s * S['hh'] + v
        s =  -(2 * bo - 1) * s
        u = (u + 1.0) * 0.5
        return (u - s * S['h']) * jnp.sqrt(S['h']) * x + u * jnp.arcsinh(x)

    def continue1(S):

        S['w'] = 1.0 + S['f']
        return jax.lax.cond(S['w'] == 0.0, lambda S: S['e'], continue4, S)
    
    return  jax.lax.cond(cond, goto1, continue1, S)

@jax.custom_jvp
@jax.jit
@jnp.vectorize
def cel(kc, p, a, b):
    r"""JAX implementation of Bulirsch's general complete elliptic integral

    Computed using the algorithm in Bulirsch, 1969b: https://doi.org/10.1007/BF02165405

    .. math::

       \[\operatorname{cel}\left(k_{c},p,a,b\right)=\int_{0}^{\pi/2}\frac{a{\cos}^{2}%
\theta+b{\sin}^{2}\theta}{{\cos}^{2}\theta+p{\sin}^{2}\theta}\frac{\,\mathrm{d%
}\theta}{\sqrt{{\cos}^{2}\theta+k_{c}^{2}{\sin}^{2}\theta}},\]

     Args:
       x: arraylike, real valued.
       kc: arraylike, real valued.
       p: arraylike, real valued.
       a: arraylike, real valued.
       b: arraylike, real valued.

     Returns:
       The value of cel, the Bulirsch general complete elliptic integral

     Notes:
       ``cel`` does not support complex-valued inputs.
       ``cel`` requires `jax.config.update("jax_enable_x64", True)`
    """

    CA = 1.0e-7
    kc = jnp.abs(kc)

    S = {
        'e': kc,
        'm': 1.0,
        'p': p,
        'b': b,
        'f': 0.0,
        'q': 0.0,
        'g': 0.0,
        'a': a,
        'kc': kc,
    }

    def p_gt_0(S):

        S['p'] = jnp.sqrt(S['p'])
        S['b'] = S['b'] / S['p']
        return S

    def p_leq_0(S):
        
        S['f'] = S['kc'] * S['kc']
        S['q'] = 1.0 - S['f']
        S['g'] = 1.0 - S['p']
        S['f'] = S['f'] - S['p']
        S['q'] = (S['b'] - S['a'] * S['p']) * S['q']
        S['p'] = jnp.sqrt(S['f'] / S['g'])
        S['a'] = (S['a'] - S['b']) / S['g']
        S['b'] = -S['q'] / (S['g'] * S['g'] * S['p']) + S['a'] * S['p']
        return S

    S = jax.lax.cond(p > 0.0, p_gt_0, p_leq_0, S)

    def goto1(S):

        S['f'] = S['a']
        S['a'] = S['b'] / S['p'] + S['a']
        S['g'] = S['e'] / S['p']
        S['b'] = S['f'] * S['g'] + S['b']
        S['b'] = S['b'] + S['b']
        S['p'] = S['g'] + S['p']
        S['g'] = S['m']
        S['m'] = S['kc'] + S['m']
        return S

    def cond_func(S):

        return jnp.abs(S['g'] - S['kc']) > S['g'] * CA

    def body_func(S):

        S['kc'] = jnp.sqrt(S['e'])
        S['kc'] = S['kc'] * 2
        S['e'] = S['kc'] * S['m']
        S = goto1(S)
        return S

    S = goto1(S)
    S = jax.lax.while_loop(cond_func, body_func, S)
    return jnp.pi * 0.5 * ((S['a'] * S['m'] + S['b']) / (S['m'] * (S['m'] + S['p'])))

@el1.defjvp
def el1_jvp(primals, tangents):

    x, kc = primals
    x_dot, kc_dot = tangents 

    k = jnp.sqrt(1 - kc**2)
    cosphi = 1 / jnp.sqrt(1 + x**2)
    sinphi = x * cosphi
    dx = 1 / (jnp.sqrt(1 - k**2 * sinphi**2) * (1 + x**2))

    E = el2(x, kc, 1.0, kc**2)
    F = el1(x, kc)
    dk = -(
        (E - kc**2 * F) / (k * kc) 
        - k * sinphi * cosphi / (kc * jnp.sqrt(1 - k * k * sinphi**2))
    ) / k 

    return F, dx * x_dot + dk * kc_dot

@el2.defjvp
def el2_jvp(primals, tangents):

    x, kc, a, b = primals
    x_dot, kc_dot, a_dot, b_dot = tangents 

    k2 = 1 - kc**2
    kc2 = kc * kc
    k = jnp.sqrt(k2)
    cosphi = 1 / jnp.sqrt(1 + x**2)
    sinphi = x * cosphi
    rad = jnp.sqrt(1 - k2 * sinphi**2)
    dx = ((a - b) / k2 * (rad - 1 / rad) + a / rad) / (1 + x**2)

    E = el2(x, kc, 1.0, kc2)
    F = el1(x, kc)
    da = F + (E - F) / k2
    db = (F - E) / k2

    fac = k**2 * sinphi * cosphi / rad
    dk = (b - a) / k2 * ((1 + 1 / kc2) * E - 2 * F - fac / kc2) + (a / kc2) * (E - kc2 * F - fac)
    dk = -kc * dk

    return el2(x, kc, a, b), dx * x_dot + dk * kc_dot + da * a_dot + db * b_dot

@el3.defjvp
def el3_jvp(primals, tangents):

    x, kc, p = primals
    x_dot, kc_dot, p_dot = tangents 

    # note that 1 - p instead of p -1 
    # is intentional here, since the sign of n 
    # is flipped between the def. of 
    # Pi in Bulirsch 1969 compared to the 
    # standard definnition. 
    n = 1 - p
    k2 = 1 - kc**2
    kc2 = kc * kc
    k = jnp.sqrt(k2)
    cosphi = 1 / jnp.sqrt(1 + x**2)
    sinphi = x * cosphi
    rad = jnp.sqrt(1 - k**2 * sinphi**2)
    dx = 1 / (rad * (1 - n * sinphi**2) * (1 + x**2))
    
    E = el2(x, kc, 1.0, kc2)
    F = el1(x, kc)
    Pi = el3(x, kc, p)
    
    fac = k2 * sinphi * cosphi / rad
    dk = -kc * (-E / kc2 + Pi + fac / kc2) / (n - k2)
    
    dn = (
        E + (k2 - n) * F / n 
        + (n - k) * (n + k) * Pi / n 
        + n * cosphi * sinphi * rad / (n * sinphi**2 - 1)
    ) / (2 * (k2 - n) * (n - 1))

    return Pi, dx * x_dot + dk * kc_dot + dn * p_dot
    
@el3.defjvp
def cel_jvp(primals, tangents):  

    kc, p, a, b = primals
    kc_dot, p_dot, a_dot, b_dot = tangents