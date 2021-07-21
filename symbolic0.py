import sympy as sm
import numpy as np
import scipy.integrate as integrate

'''

norm = lambda var, mu, sigm: 1/(sm.sqrt(2*sm.pi)*sigm)*sm.exp(-1/2*((var-mu)/sigm)**2)

# FOR p(x|y) = p(y|x)p(x)/p(y) for p is all Normal:
\mu_{x|y} = (\sigma_{x}^2 * y + \sigma_{y}^2 * \mu_{x})/(\sigma_{x}^2+\sigma_{y}^2)
\sigma_{x|y} = \sigma_{x}^2 + \sigma_{y}^2
f = lambda x: norm(x, 0.2/(1+0.1**2), 0.1/(sm.sqrt(1+0.1**2)))
x = np.linspace(-3, 3, 1000)
y = np.array([f(xx) for xx in x])
plt.plot(x, y)
# END FOR
'''

def P_xz(xs, zs, sigm_x=0.1, sigm_y=0.1, mu_z=0.0, sigm_z=0.1):

    '''
    Calculating P(x0<x<x1, z0<z<z1)
    = \int P(X|y)*P(y, Z) dy

    = \int (\int p(x|y) dx) * (\int p(y|z)p(z) dz) dy
    where p(x|y), p(y|z), p(z) are all normal distributions:
    p(x|y) = \frac{1}{\sqrt(2*\pi*\sigma_{x}^2)}
             * \exp^{-1/2*(\frac{x-y}{\sigma_{x}})^2}
    
    (used in scipy)
    =
        \int_{y=-inf}^{inf} 
            \int_{x=x0}^{x1} p(x|y) dx
            * \int_{z=z0}^{z1}p(y|z)p(z) dz
        dy
    
    = \int (\int p(x|y) * (\int p(y|z)p(z) dz)) dy dx
    (used in sympy)
    = 
       \int_{x=x0}^{x1} 
            \int_{y=-inf}^{inf} 
                 p(x|y) * \int_{z=z0}^{z1}p(y|z)p(z) dz
            dy
        dx

    '''
    x0, x1 = xs
    z0, z1 = zs

    x, y, z = sm.var("x y z")
    norm = lambda var, mu, sigm: 1/(sm.sqrt(2*sm.pi)*sigm)*sm.exp(-1/2*((var-mu)/sigm)**2)
    
    # p(x|y):
    p_x_y = lambda sigm_x: norm(x, y, sigm_x)

    # p(y|z):
    p_y_z = lambda sigm_y: norm(y, z, sigm_y)

    # p(z):
    p_z = lambda sigm_z, mu_z: norm(z, mu_z, sigm_z)

    # p(y, z0<z<z1) = \int_{z=z0}{z1} p(y|z)*p(z) dz:
    p_yz = sm.integrate(p_y_z(sigm_y)*p_z(sigm_z, mu_z), (z, z0, z1))

    # print("p_yz:")
    # print(p_yz)
    # print("p_x_y:")
    # print(p_x_y(sigm_y))

    # p(x, z0<z<z1) = \int_{y=-inf}{inf} p(x|y)*p(y, z0<z<z1) dz:
    p_xz = sm.integrate(p_x_y(sigm_y)*p_yz, (y, -sm.oo, sm.oo))
    
    # P(x0<x<x1, z0<z<z1) = \int_{x=x0}{x1} p(x, z0<z<z1) dx:
    P_xz = sm.integrate(p_xz, (x, x0, x1))

    return(P_xz)


def P_xz_num(xs, zs, sigm_x=0.1, sigm_y=0.1, mu_z=0.0, sigm_z=0.1):
    '''
    Same as P_xz, but numerical.
    '''
    x0, x1 = xs
    z0, z1 = zs
    y0, y1 = (-10, 10)

    norm = lambda x, mu, sigm: (
        1/(np.sqrt(2*np.pi)*sigm)*np.exp(-1/2*((x-mu)/sigm)**2))

    # p(z):
    p_z = lambda z: norm(z, mu_z, sigm_z)

    # p(y|z):
    p_y_z = lambda y, z: norm(y, z, sigm_y)
    
    # p(y, z0<z<z1) = \int_{z=z0}^{z1} p(y|z)*p(z) dz:
    p_yz = lambda y: integrate.quad(lambda z: p_y_z(y, z)*p_z(z), z0, z1)[0]
    
    # p(x|y):
    p_x_y = lambda x, y: norm(x, y, sigm_x) 
    
    # P(x<x0|y):
    P_x_y = lambda y: integrate.quad(lambda x: p_x_y(x, y), x0, x1)[0] 
    
    # P(x0<x<x1, z0<z<z1) = \int_{y=-inf}^{inf} P(x<x0|y)*p(y, z<z0) dy:
    P_xz = integrate.quad(lambda y: P_x_y(y)*p_yz(y), y0, y1)[0]
    return(P_xz)


def test_scipy():
    
    inf = 10

    print("P(x<1/2, Z): ", P_xz_num((-inf, 1/2), (-inf, inf)))
    # , sigm_x=1.0, sigm_y=1.0, sigm_z=1.0

    print("P(x>1/2, Z): ", P_xz_num((1/2, inf), (-inf, inf)))
    print("P(X, Z)", P_xz_num((-inf, inf), (-inf, inf)))
    
    # for with scipy but not with sympy:
    print("P(x<0.5, z<0.5, mu_z=0.5, sigm_y=1.0, sigm_z=1.0): ")
    print(P_xz_num((-inf, 0.5), (-inf, 0.5), mu_z=0.5, sigm_y=1.0, sigm_z=1.0))
    

def test_sympy():
    
    print("P(x<1/2, Z): ", float(P_xz((-sm.oo, 1/2), (-sm.oo, sm.oo))))
    print("P(x>1/2, Z): ", float(P_xz((1/2, sm.oo), (-sm.oo, sm.oo))))
    print("P(X, Z)", float(P_xz((-sm.oo, sm.oo), (-sm.oo, sm.oo))))

    # this will not work due to sympy.factor:multiplicity bug:
    # print(float(P_xz(0.5, 0.5, mu_z=0.5)))
    # print("P(x<0.5, z<0.5, mu_z=0.5): ",
    #       float(P_xz((-sm.oo, 0.5), (-sm.oo, 0.5), mu_z=0.0, sigm_y=1.0, sigm_z=1.0)))


if __name__ == '__main__':
    print("scipy:")
    test_scipy()
    print("\nsympy")
    # test_sympy()
