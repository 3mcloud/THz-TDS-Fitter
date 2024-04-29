import numpy as np
import scipy as scp
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.5, font='serif')


class THzFitter:
    """
    This class defines fitting parameters for THz TDS spectra, and
    its instances are passed to the FitterGUI class.

    Attributes
    ----------
    funcDict : dict
        Dictionary containing fit function definitions and their parameters.
    ydata_ids : list
        List of y-axis labels for plotting. Default to 'RI' for refractive index and
        'Abs' for absorption coefficient.
    xaxis_labels : list
        List of x-axis labels for plotting. Default to 'Frequency [THz]'
    yaxis_labels : list
        List of y-axis labels. Default to 'n' for refractive index and '\alpha'
        for absorption coefficient.
    xunit : str
        Unit of x-axis. Default to 'THz'
    xconv : float
        Conversion factor for x-axis from Hz. Default to 1e12.
    xlims : list
        List of x-axis limits in units defined by 'xunit.' Default to 0-10.

    Methods
    -------
    fit_func(xdata, *params, mode='RI'):
        Calculates the fit for the THz TDS spectral data.

    """
    def __init__(self):
        # Fit function definitions
        self.funcDict = {
            "EpsInf": {"func": lambda omega, p: np.repeat(p[0], len(omega)),
                       "params": [r"$\varepsilon_{\infty,}$"],
                       "units": [""],
                       "plims": [[1, 3]],
                       "values": [2],
                       "optimize": [True],
                       "nfuncs": 1,
                       "maxfuncs": 1},
            "Debye": {"func": lambda omega, p: p[0] / (1 - (1j * (omega / (2 * np.pi * p[1] * self.xconv)))),
                      "params": [r"$\Delta \varepsilon_{\mathrm{Deb},}$", r"$\nu_{\mathrm{Deb},}$"],
                      "units": ["", "THz"],
                      "plims": [[0.0, 0.5], [0, 20]],
                      "values": [0.1, 1],
                      "optimize": [True, True],
                      "nfuncs": 1,
                      "maxfuncs": 10},
            "HN": {
                "func": lambda omega, p: p[0] / (1 - (1j * (omega / (2 * np.pi * p[1] * self.xconv))) ** p[2]) ** p[3],
                "params": [r"$\Delta \varepsilon_{\mathrm{HN},}$", r"$\nu_{\mathrm{HN},}$",
                           r"$\alpha_{\mathrm{HN},}$", r"$\beta_{\mathrm{HN},}$"],
                "units": ["", "THz", "", ""],
                'plims': [[0.0, 0.5], [0, 5], [0, 2], [0, 2]],
                "values": [0.1, 2, 1, 1],
                "optimize": [True, True, True, True],
                "nfuncs": 0,
                "maxfuncs": 10},
            "DHO": {"func": lambda omega, p: ((p[1] * (2 * np.pi * p[0] * self.xconv) ** 2) /
                                              ((2 * np.pi * p[0] * self.xconv) ** 2 - omega ** 2 -
                                               1j * omega * (2 * np.pi * p[2] * self.xconv))),
                    "params": [r"$\nu_{\mathrm{DHO},}$", r"$A_{\mathrm{DHO},}$", r"$\Gamma_{\mathrm{DHO},}$"],
                    "units": ["THz", "", "THz"],
                    "plims": [[0, 10], [0.0, 0.5], [0, 10]],
                    "values": [2, 0.1, 1],
                    "optimize": [True, True, True],
                    "nfuncs": 1,
                    "maxfuncs": 10}}
        self._calcDict = {"RI": lambda xdata, ydata: [
                             np.real(np.sqrt(ydata)),
                             4 * np.pi * np.imag(np.sqrt(ydata)) * xdata * 1e10 / scp.constants.c],
                          "Eps": lambda xdata, ydata: [
                             np.real(ydata),
                             np.imag(ydata)]}
        self._yLabels = {"RI": [r"$n$", r"$\alpha$ [cm$^{-1}$]"],
                         "Eps": [r"$\varepsilon'$", r"$\varepsilon''$"]}
        # Public variables
        self.ydata_ids = ["RI", "Abs"]
        self.xaxis_labels = ["Frequency [THz]"] * 2
        self.yaxis_labels = [r"$n$", r"$\alpha$ [cm$^{-1}$]"]
        self.xunit = "THz"
        self.xconv = 1e12  # THz
        self.xlims = [0, 10]  # THz
        return

    def fit_func(self, xdata, *params, mode='RI'):
        """
        This method calculates theoretical THz TDS spectrum given a list of
        frequencies and a tuple of fit parameters.

        Parameters
        ----------
        xdata : array-like
            The input data for the fit function.
        *params
            Variable-length argument for THz fit parameters
        mode : str, optional
            The basis for the calculation, either refractive index or
            dielectric permittivity. Default is 'RI'.

        Returns
        -------
        tot_ret : array-like
            The calculated THz spectrum, coadding all functions defined funcDict
        func_ret : list
            The calculated THz spectrum of each function defined in funcDict

        """
        funcs = []
        ii = 0
        for v in self.funcDict.values():
            jj = len(v["params"])
            for n in range(v["nfuncs"]):
                funcs.append(v["func"](
                    2 * np.pi * xdata * self.xconv, params[ii:ii + jj]))
                ii += jj
        tot_ret = self._calcDict[mode](xdata, np.sum(funcs, axis=0))
        func_ret = [self._calcDict[mode](xdata, func) for func in funcs]
        self.yaxis_labels = self._yLabels[mode]
        return tot_ret, func_ret
