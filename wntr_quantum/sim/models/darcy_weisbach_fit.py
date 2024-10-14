import matplotlib.pyplot as plt
import numpy as np
from wntr.epanet.util import FlowUnits
from wntr.epanet.util import HydParam
from wntr.epanet.util import from_si


def friction_factor(q, e, s):  # noqa: D417
    """Computes the ground truth for the friction factor.

    Args:
        q = |pipe flow|
        e = pipe roughness  / diameter
        s = viscosity * pipe diameter
    """
    A8 = 4.61841319859066668690e00
    A9 = -8.68588963806503655300e-01

    w = q / s

    # if w >= A1:
    y1 = A8 / pow(w, 0.9)
    y2 = e / 3.7 + y1
    y3 = A9 * np.log(y2)
    f = 1.0 / (y3 * y3)
    return f


def dw_fit(roughness, diameter, plot=False, convert_to_us_unit=False):
    """Fit the dw friction coefficient to a quadratic polynomial.

    Args:
        roughness (float): roughness pf the pipe in meter
        diameter (float): diamter of the pipe in meter
        plot(bool): plot the solution for visual inspection
        convert_to_us_unit(bool): convert to us unit
    """

    def convert_to_USunit(roughness, diameter):
        """Converts roughness and diameter to US units."""
        diameter_us = from_si(FlowUnits.CFS, diameter, HydParam.Length)
        roughness_us = 0.001 * from_si(FlowUnits.CFS, roughness, HydParam.Length)
        return roughness_us, diameter_us

    N = 250
    Q = np.logspace(0, 4, num=N)
    if convert_to_us_unit:
        roughness, diameter = convert_to_USunit(roughness, diameter)
    viscosity = 0.000011
    e = roughness / diameter
    s = viscosity * diameter

    factors = np.zeros(N)
    for iq, q in enumerate(Q):
        factors[iq] = friction_factor(q, e, s)

    res = np.polyfit(1 / Q, factors, 2)

    if plot:
        approx = np.poly1d(res)
        plt.loglog(Q, approx(1 / Q))
        plt.loglog(Q, factors)
        plt.loglog(Q, res[0] * (1 / Q) ** 2 + res[1] * 1 / Q + res[2])
        plt.show()

        plt.semilogx(Q, 1 - np.abs((approx(1 / Q)) / factors))
        plt.show()

        print(res)
    # return np.array(res), np.poly1d(res)(1 / Q), factors, Q
    return np.array(res)


def evaluate_fit(coeffs, flow):
    """Evaluate the fit.

    Args:
        coeffs (_type_): _description_
        flow (_type_): _description_

    Returns:
        _type_: _description_
    """
    return coeffs[0] * (1 / flow) ** 2 + coeffs[1] * 1 / flow + coeffs[2]


if __name__ == "__main__":
    # r = 0.000164
    # d = 0.820210
    # res = dw_fit(roughness=r, diameter=d, plot=True, convert_to_us_unit=False)

    # print(evaluate_fit(res, 1.766))

    # roughness = 0.005
    roughness = 0.5 * 1e-3
    ndiams = 5
    DIAMS = np.arange(5, 20, 3) / 12

    BASELINE = []
    APPROX = []
    for d in DIAMS:
        print(d)
        res, approx, baseline, qval = dw_fit(
            roughness=roughness, diameter=d, plot=False, convert_to_us_unit=False
        )
        BASELINE.append(baseline)
        APPROX.append(approx)

    n = 24

    colors = plt.cm.tab20(np.linspace(0, 1, n))

    i = 0
    for bl, ap in zip(BASELINE, APPROX):
        plt.loglog(qval, bl, "--", c=colors[i])
        plt.loglog(qval, ap, "-", c=colors[i])
        plt.grid(visible=True, which="both")
        # plt.xlim(10, 1000)
        plt.xlabel("Reynold Number")
        plt.ylabel("Friction Factor")
        # plt.yticks([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
        i += 1
    plt.show()
