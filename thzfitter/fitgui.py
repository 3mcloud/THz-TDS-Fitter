import numpy as np
import scipy as scp
import codecs
import os
import datetime as dt
import copy
import math
import warnings
import ipywidgets as wdg
import IPython as IPy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.5, font='serif')
warnings.filterwarnings("error", category=RuntimeWarning)


class FitterGUI:
    """
    This class builds a Jupyter Notebook GUI for human-assisted
    fitting of complex spectra, splitting into its real and imaginary parts.

    Attributes
    ----------
    fitter : object
        Class that defines the fitting function and parameters for the
        GUI's calculations.
    output : ipywidget widget
        ipywidgets.Output() widget that defines the Jupyter GUI. Must be created
        in the executing Jupyter notebook and passed to this class during initialization.

    Methods
    -------
    display()
        Generate iPython GUI when run in a Jupyter notebook.
    """
    def __init__(self, fitter, output=None):
        """
        This method initiates a FitterGUI object.

        Parameters
        ----------
        fitter : object
            Class that defines the fitting function and parameters for the
            GUI's calculations.
        output : ipython widget
            ipywidgets.Output() widget that defines the Jupyter GUI. Must be created 
            in the executing Jupyter notebook and passed to this initialization method.
        """
        # Public variables
        self.fitter = fitter
        self.output = output
        # Private variables
        self._samp_name = ""
        self._numy = len(self.fitter.ydata_ids)
        self._meas_file_names = ["", ""] * self._numy
        self._data_labels = ['Data', 'Fit']
        self._param_nstep = 1000
        self._fit_lims = self.fitter.xlims
        self._plotted_meas_data = False
        self._plotted_spans = False
        self._save_loc = "Results"
        self._meas_xdata = None
        self._meas_ydata = [[None, None]] * self._numy
        self._fit_xdata = None
        self._fit_ydata = [None] * self._numy
        self._lospan = [None] * self._numy
        self._hispan = [None] * self._numy
        self._params = {}
        self._best_params = {}
        self._meas_data_dict = {}
        self._fig = None
        self._ax = None
        self._calc_error = False
        self._err_incl = True
        self._err_labs = [r"$\chi^{2}_{\mathrm{red}} =$", "red_chi_sq"]
        self._fdelim = '_'
        # Initialization
        self._init_params()
        self._init_widgets()
        return

    def _init_params(self):
        self._params.clear()
        for v in self.fitter.funcDict.values():
            self._params.update({r"%s$_{%d}$" % (v["params"][j], i + 1): {
                "min": v["plims"][j][0], "max": v["plims"][j][1],
                "val": v["values"][j], "err": 0,
                "opt": v["optimize"][j], "unit": v["units"][j]}
                for i in range(v["nfuncs"]) for j in range(len(v["params"]))})
        return

    def _init_widgets(self):
        self._init_fupload_widgets()
        self._init_nfunc_widgets()
        self._init_trunc_widgets()
        self._init_param_widgets()
        self._init_fit_widgets()
        self._init_widget_box()
        return

    def _init_fupload_widgets(self):
        self._fupload_widgets = {
            "label": wdg.Label("Data:"),
            "upload": wdg.FileUpload(accept="", multiple=True)}
        self._sample_widgets = {
            "select": wdg.Dropdown(description="Sample:", layout=wdg.Layout(width='20%'))}
        self._fupload_widgets["upload"].observe(self._set_files, names='value')
        return

    def _init_nfunc_widgets(self):
        self._nfunc_widgets = {k: wdg.BoundedIntText(
            min=0, max=v['maxfuncs'], value=v["nfuncs"], description="N_%s" % k,
            layout=wdg.Layout(width='13%')) for k, v in list(self.fitter.funcDict.items())}
        return

    def _init_trunc_widgets(self):
        self._trunc_widgets = {
            "dataMin": wdg.BoundedFloatText(
                min=0, max=100, value=self._fit_lims[0],
                description="Fit Lo [%s]:" % self.fitter.xunit, layout=wdg.Layout(width='15%')),
            "dataMax": wdg.BoundedFloatText(
                min=0, max=100, value=self._fit_lims[1],
                description="Fit Hi [%s]:" % self.fitter.xunit, layout=wdg.Layout(width='15%'))}
        return

    def _init_param_widgets(self):
        self._opt_widgets = {k: wdg.Checkbox(
            value=v["opt"], layout=wdg.Layout(width='15%'), description=" ")
            for k, v in self._params.items()}
        self._slider_widgets = {k: wdg.FloatSlider(
            min=v["min"], max=v["max"], value=v["val"],
            step=((v["max"] - v["min"]) / self._param_nstep),
            layout=wdg.Layout(width='100%'), readout_format='.4g', font_size=24)
            for k, v in self._params.items()}
        self._min_widgets = {k: wdg.BoundedFloatText(
            min=-1e10, max=1e10, value=v["min"], description="Min:", layout=wdg.Layout(width='16%'))
            for k, v in self._params.items()}
        self._max_widgets = {k: wdg.BoundedFloatText(
            min=-1e10, max=1e10, value=v["max"], description="Max:", layout=wdg.Layout(width='16%'))
            for k, v in self._params.items()}
        return

    def _init_fit_widgets(self):
        self._fit_widgets = {
            "label": wdg.Label("Fit:"),
            "optimize": wdg.Button(description="Optimize", layout=wdg.Layout(width='25%')),
            "save": wdg.Button(description="Save Best Fit", layout=wdg.Layout(width='25%')),
            "load": wdg.Button(description="Load Best Fit", layout=wdg.Layout(width='25%')),
            "upload": wdg.FileUpload(accept="", multiple=True)}
        self._fit_widgets["optimize"].on_click(self._optimize_params)
        self._fit_widgets["save"].on_click(self._save_params)
        self._fit_widgets["load"].on_click(self._load_params)
        self._fit_widgets["upload"].observe(self._upload_params, names='value')
        return

    def _init_widget_box(self):
        self._slider_widget_box = wdg.interactive(self._plot_fits, **self._slider_widgets)
        self._min_widget_box = wdg.interactive(self._set_mins, **self._min_widgets)
        self._max_widget_box = wdg.interactive(self._set_maxs, **self._max_widgets)
        self._opt_widget_box = wdg.interactive(self._set_opts, **self._opt_widgets)
        self._param_widget_box = wdg.VBox([
            wdg.HBox([op, sl, mn, mx], layout=wdg.Layout(width='100%', overflow='auto', padding='0'))
            for (op, sl, mn, mx) in
            zip(self._opt_widget_box.children, self._slider_widget_box.children,
                self._min_widget_box.children, self._max_widget_box.children)])
        self._fupload_widget_box = wdg.HBox(list(self._fupload_widgets.values()))
        self._sample_widget_box = wdg.interactive(self._set_samples, **self._sample_widgets)
        self._trunc_widget_box = wdg.interactive(self._set_truncs, **self._trunc_widgets)
        self._nfunc_widget_box = wdg.interactive(self._set_nfuncs, **self._nfunc_widgets)
        self._button_widget_box = wdg.HBox(list(self._fit_widgets.values()))
        self._widget_box = wdg.VBox([
            wdg.HBox(list(self._fupload_widget_box.children) + list(self._sample_widget_box.children) +
                     list(self._trunc_widget_box.children)),
            wdg.HBox(list(self._nfunc_widget_box.children)),
            self._param_widget_box, self._button_widget_box])
        return

    def _set_files(self, change):
        fconts = {v["name"]: codecs.decode(v.content, encoding='utf-8')
                  for v in self._fupload_widgets["upload"].value}
        self._meas_data_dict.clear()
        for k, v in fconts.items():
            data = _load_data(k, v)
            tags = k.replace('-', '_').replace('.', '_').replace(';', '_').split('_')[:-1]
            found_id = False
            for yid in self.fitter.ydata_ids:
                if yid.upper() in k.upper():
                    fid = '_'.join([t for t in tags if t.upper() != yid.upper()])
                    if fid in self._meas_data_dict.keys():
                        self._meas_data_dict[fid].update({yid: data, "%sfile" % yid: k})
                    else:
                        self._meas_data_dict.update({fid: {yid: data, "%sfile" % yid: k}})
                    found_id = True
                    break
            if not found_id:
                fid = '.'.join(k.split('.')[:-1])
                self._meas_data_dict.update({fid: {"Comb": data, "Combfile": k}})
        self._sample_widgets["select"].options = self._meas_data_dict.keys()
        return

    def _set_samples(self, **kwargs):
        new_samp_name = self._sample_widgets["select"].value
        if new_samp_name is not None:
            if new_samp_name != self._samp_name:
                for ax in self._ax:
                    for ln in ax.lines + ax.collections + ax.patches:
                        ln.remove()
                self._plotted_meas_data = False
                self._err_incl = True
            self._samp_name = new_samp_name
        if self._samp_name == "":
            return
        meas_data = self._meas_data_dict[self._samp_name]
        self._meas_xdata = None
        for k, v in meas_data.items():
            if k in self.fitter.ydata_ids:
                ii = self.fitter.ydata_ids.index(k)
                if self._meas_xdata is None:
                    self._meas_xdata = v[0]
                    self._meas_ydata[ii] = self._sort_ydata(v[1:])
                else:
                    self._meas_ydata[ii] = [np.interp(
                        self._meas_xdata, v[0], y) for y in self._sort_ydata(v[1:]) if y is not None]
            elif k in ["%sfile" % yid for yid in self.fitter.ydata_ids]:
                ii = self.fitter.ydata_ids.index(k.strip("file"))
                self._meas_file_names[ii] = v
            elif k == "Comb":
                self._meas_xdata = v[0]
                for i, yid in enumerate(self.fitter.ydata_ids):
                    if len(v[1:]) <= 2:  # mean values only
                        self._meas_ydata[i] = [v[i + 1], None]
                    else:  # either repeated measurements or mean + std
                        self._meas_ydata[i] = self._sort_ydata(v[2 * i + 1:2 * i + 2 + 1])
            elif k == "Combfile":
                self._meas_file_names[0] = v
                self._meas_file_names[1:] = [" "] * len(self._meas_file_names[1:])
            else:
                continue
        self._fit_xdata = copy.deepcopy(self._meas_xdata)
        self._fit_lims = [self._fit_xdata[0], self._fit_xdata[-1]]
        self._plot_meas_data()
        self._plot_fits()
        self._trunc_widgets["dataMax"].value = self._fit_lims[1]
        self._trunc_widgets["dataMin"].value = self._fit_lims[0]
        return

    def _set_nfuncs(self, **kwargs):
        if not len(kwargs.values()):
            for k, v in self.fitter.funcDict.items():
                v["nfuncs"] = len([kk for kk in self._params.keys() if v["params"][0] in kk])
                self._nfunc_widgets[k].value = v["nfuncs"]
            return
        new_vals = list(kwargs.values())
        for i, v in enumerate(self.fitter.funcDict.values()):
            v["nfuncs"] = new_vals[i]
        new_params = {}
        for v in self.fitter.funcDict.values():
            for i in range(v["nfuncs"]):
                for j in range(len(v["params"])):
                    key = r"%s$_{%d}$" % (v["params"][j], i + 1)
                    if key in list(self._params.keys()):
                        new_params[key] = {"min": self._params[key]["min"], "max": self._params[key]["max"],
                                           "val": self._params[key]["val"], "err": self._params[key]["err"],
                                           "opt": self._params[key]["opt"], "unit": self._params[key]["unit"]}
                    else:
                        if i != 0:
                            nv = new_params[r"%s$_{%d}$" % (v["params"][j], i)]
                            new_params.update({
                                key: {"min": nv["min"], "max": nv["max"],
                                      "val": nv["val"], "err": nv["err"],
                                      "opt": nv["opt"], "unit": nv["unit"]}})
                        else:
                            new_params.update({
                                key: {"min": v["plims"][j][0], "max": v["plims"][j][1],
                                      "val": v["values"][j] * 1.5, "err": 0,
                                      "opt": v["optimize"][j], "unit": v["units"][j]}})
        self._params = new_params
        self._set_param_widgets()
        return

    def _set_param_widgets(self):
        self._init_param_widgets()
        self._slider_widget_box.children = wdg.interactive(self._plot_fits, **self._slider_widgets).children
        self._min_widget_box.children = wdg.interactive(self._set_mins, **self._min_widgets).children
        self._max_widget_box.children = wdg.interactive(self._set_maxs, **self._max_widgets).children
        self._opt_widget_box.children = wdg.interactive(self._set_opts, **self._opt_widgets).children
        self._param_widget_box.children = [
            wdg.HBox([op, sl, mn, mx]) for (op, sl, mn, mx) in
            zip(self._opt_widget_box.children, self._slider_widget_box.children,
                self._min_widget_box.children, self._max_widget_box.children)]
        return

    def _set_truncs(self, **kwargs):
        if len(kwargs.values()):
            self._fit_lims = list(kwargs.values())
        if self._fit_xdata is None:
            return
        if self._fit_lims[0] < self._fit_xdata[0]:
            if self._fit_lims[0] < self._meas_xdata[0]:
                diff = np.diff(self._fit_xdata)[0]
                self._fit_xdata = np.append(
                    np.arange(self._fit_lims[0], self._fit_xdata[0], diff), self._fit_xdata)
            else:
                diff = np.diff(self._meas_xdata)[0]
                self._fit_xdata = np.arange(self._meas_xdata[0], self._fit_xdata[-1] + diff, diff)
        else:
            if self._fit_lims[0] > self._meas_xdata[0]:
                lo = np.argmin(abs(self._meas_xdata[0] - self._fit_xdata))
            else:
                lo = np.argmin(abs(self._fit_lims[0] - self._fit_xdata))
            self._fit_xdata = self._fit_xdata[lo:]
        if self._fit_lims[1] > self._fit_xdata[-1]:
            if self._fit_lims[1] > self._meas_xdata[-1]:
                diff = np.diff(self._fit_xdata)[-1]
                self._fit_xdata = np.append(
                    self._fit_xdata, np.arange(self._fit_xdata[-1] + diff, self._fit_lims[1] + diff, diff))
            else:
                diff = np.diff(self._meas_xdata)[-1]
                self._fit_xdata = np.arange(self._fit_xdata[0], self._meas_xdata[-1] + diff, diff)
        else:
            if self._fit_lims[1] < self._meas_xdata[-1]:
                hi = np.argmin(abs(self._meas_xdata[-1] - self._fit_xdata))
            else:
                hi = np.argmin(abs(self._fit_lims[1] - self._fit_xdata))
            self._fit_xdata = self._fit_xdata[:hi]
        self._plot_fits()
        self._plot_fit_regions()
        return

    def _set_params(self, vals=None, errs=None, rescale=False):
        for i, v in enumerate(self._params.values()):
            if vals is not None:
                v["val"] = vals[i]
            if errs is not None:
                v["err"] = errs[i]
            if rescale is True:
                newmax = (2 * (v['val'] - v['min'])) + v['min']
                v['max'] = round(newmax, 3 - int(math.floor(math.log10(abs(newmax)))) - 1)
        if vals is not None:
            return vals
        else:
            return [v["val"] for v in self._params.values()]

    def _set_mins(self, **kwargs):
        vals = list(kwargs.values())
        for i, v in enumerate(self._params.values()):
            v["min"] = vals[i]
        for i, v in enumerate(self._slider_widgets.values()):
            v.min = vals[i]
            v.step = (v.max - v.min) / self._param_nstep
        return

    def _set_maxs(self, **kwargs):
        vals = list(kwargs.values())
        for i, v in enumerate(self._params.values()):
            v["max"] = vals[i]
        for i, v in enumerate(self._slider_widgets.values()):
            v.max = vals[i]
            v.step = (v.max - v.min) / self._param_nstep
        return

    def _set_opts(self, **kwargs):
        vals = list(kwargs.values())
        for i, v in enumerate(self._params.values()):
            v["opt"] = vals[i]
        return

    def _run_fits(self, pvals):
        pvals = self._set_params(pvals)
        if self._fit_xdata is not None:
            self._fit_func(self._fit_xdata, *pvals)
            self._calc_err_metric()
        return

    def _fit_func(self, xdata, *params):
        try:
            self._fit_ydata = self.fitter.fit_func(xdata, *params)[0]
            self._calc_error = False
        except RuntimeWarning:
            self._fit_ydata = np.zeros((len(self._fit_ydata), len(xdata)))
            self._calc_error = True
        return np.hstack(self._fit_ydata)

    def _optimize_params(self, button):
        vals, opts, mins, maxs = np.array(
            [[v["val"], v["opt"], v["min"], v["max"]]
             for v in self._params.values()]).T
        for i, opt in enumerate(opts):
            if not opt:
                mins[i] = vals[i] * (1 - 1e-6)
                maxs[i] = vals[i] * (1 + 1e-6)
        lo, hi = [np.argmin(abs(fl - self._meas_xdata)) for fl in self._fit_lims]
        timestamp = str(dt.datetime.now()).split()[1].split('.')[0]
        try:
            if self._err_incl:
                sigma = np.hstack([my[1][lo:hi] for my in self._meas_ydata]) + (1e-6)
            else:
                sigma = None
            pout = scp.optimize.curve_fit(
                self._fit_func, self._meas_xdata[lo:hi], np.hstack([my[0][lo:hi] for my in self._meas_ydata]),
                sigma=sigma, p0=vals, bounds=(mins, maxs))
            self._set_params(pout[0], np.sqrt(np.diag(pout[1])), rescale=True)
            self._set_param_widgets()
            self._plot_fits()
            self._fit_widgets['optimize'].description = (
                    r"Optimize (%s = %.2f)" % (self._err_labs[1], self._err_metric))
        except RuntimeError:
            self._fit_widgets['optimize'].description = "Optimize (failed at %s)" % timestamp
        return

    def _save_params(self, button):
        self._best_params = copy.deepcopy(self._params)
        now = str(dt.datetime.now()).split()
        timestamp = now[1].split('.')[0]
        timestamp_str = now[0].replace('-', '') + self._fdelim + now[1].split('.')[0].replace(':', '')
        title = "***** %s, %s = %.2f *****\n" % (self._err_labs[1], self._samp_name, self._err_metric)
        header = ("| %-45s | %-6s | %-9s | %-9s | %-9s | %-9s | %-8s |\n" %
                  ("Parameter", "Units", "Avg", "Stdev", "Min", "Max", "Opt?"))
        div = "-" * 117 + "\n"
        table_layers = title + div + header + div
        layer = 0
        for k, v in self._params.items():
            entry = ("| %-45s | %-6s | %-9.3E | %-9.3E | %-9.3E | %-9.3E | %-8s |\n" %
                     (k, v["unit"], v["val"], v["err"], v["min"], v["max"], str(v["opt"])))
            layer += 1
            table_layers += (entry + div)
        param_file = os.path.join(self._save_loc, "%s_fitParams_%s.txt" % (self._samp_name, timestamp_str))
        if not os.path.isdir(self._save_loc):
            os.makedirs(self._save_loc)
        with open(param_file, 'w') as f:
            f.write(table_layers)
        self._fit_widgets['save'].description = "Save Best Fit (last at %s)" % timestamp
        return

    def _load_params(self, button):
        if len(self._best_params.keys()) == 0:
            return
        self._params = copy.deepcopy(self._best_params)
        timestamp = str(dt.datetime.now()).split()[1].split('.')[0]
        self._fit_widgets['load'].description = "Load Best Fit (last at %s)" % timestamp
        self._set_nfuncs()
        self._set_param_widgets()
        self._plot_fits()
        return

    def _upload_params(self, button):
        fname = self._fit_widgets["upload"].value[0]["name"]
        fdata = codecs.decode(self._fit_widgets["upload"].value[0].content, encoding='utf-8')
        self._params = _unpack_params(fname, fdata)
        self._best_params = copy.deepcopy(self._params)
        ldate = fname.split(self._fdelim)[-2:]
        date = ldate[0][:4] + '-' + ldate[0][4:6] + '-' + ldate[0][6:8]
        time = ldate[1][:2] + ':' + ldate[1][2:4] + ':' + ldate[1][4:6]
        timestamp = date + ' ' + time
        self._fit_widgets['load'].description = "Load Best Fit (last at %s)" % timestamp
        self._set_params(rescale=True)
        self._set_param_widgets()
        self._set_nfuncs()
        self._plot_fits()
        return

    def _plot_meas_data(self):
        if self._meas_xdata is None or self._plotted_meas_data is True:
            return
        for i, dat in enumerate(self._meas_ydata):
            if dat[0] is not None and dat[1] is not None:
                self._ax[i].plot(self._meas_xdata, dat[0], linewidth=1, color='b')
                self._ax[i].fill_between(self._meas_xdata, dat[0] + dat[1], dat[0] - dat[1],
                                         linewidth=0, color='b', alpha=0.5, label=self._data_labels[0])
            elif dat[0] is not None:
                self._ax[i].plot(self._meas_xdata, dat[0], linewidth=2, color='b',
                                 alpha=0.8, label=self._data_labels[0])
            else:
                continue
            self._lospan[i] = self._ax[i].axvspan(-1e10, self._fit_lims[0], color='k', alpha=0.2, linewidth=0)
            self._hispan[i] = self._ax[i].axvspan(self._fit_lims[1], 1e10, color='k', alpha=0.2, linewidth=0)
            if i == (len(self._meas_ydata) - 1):
                self._ax[i].set_xlabel(self.fitter.xaxis_labels[i])
            self._ax[i].set_ylabel(self.fitter.yaxis_labels[i])
            self._ax[i].set_xlim(np.amin(self._meas_xdata), np.amax(self._meas_xdata))
            self._fig.canvas.draw()
        self._plotted_meas_data = True
        return

    def _plot_fits(self, **kwargs):
        if self._fit_xdata is None:
            return
        pvals = list(kwargs.values())
        if len(pvals) == 0:
            pvals = [v["val"] for v in self._params.values()]
        self._run_fits(pvals)
        for i, dat in enumerate(self._meas_ydata):
            if self._fit_ydata[i] is not None:
                if np.any([self._data_labels[1] in lab for lab in [p.get_label() for p in self._ax[i].lines]]):
                    self._ax[i].lines[-1].set_xdata(self._fit_xdata)
                    self._ax[i].lines[-1].set_ydata(self._fit_ydata[i])
                else:
                    self._ax[i].plot(self._fit_xdata, self._fit_ydata[i], linewidth=1, color='r')
        if self._err_metric is not None:
            for ax in self._ax:
                if not self._calc_error:
                    ax.lines[-1].set_label(
                        r"%s %s %.2f" % (self._data_labels[1], self._err_labs[0], self._err_metric))
                else:
                    ax.lines[-1].set_label(r"%s %s $=$ MATH ERROR" % (self._data_labels[1], self._err_labs[0]))
        self._ax[0].legend(title=self._samp_name, fontsize=12, title_fontsize=12)
        self._scale_yaxis()
        self._fig.canvas.draw()
        return

    def _scale_yaxis(self):
        xlo = np.argmin(abs(self._meas_xdata - self._fit_lims[0]))
        xhi = np.argmin(abs(self._meas_xdata - self._fit_lims[1]))
        if self._fit_lims[0] > self._meas_xdata[0]:
            mlo = np.argmin(abs(self._meas_xdata - self._fit_lims[0]))
            flo = np.argmin(abs(self._fit_xdata - self._fit_lims[0]))
        else:
            mlo = 0
            flo = 0
        if self._fit_lims[1] < self._meas_xdata[-1]:
            mhi = np.argmin(abs(self._meas_xdata - self._fit_lims[1]))
            fhi = np.argmin(abs(self._fit_xdata - self._fit_lims[1]))
        else:
            mhi = len(self._meas_xdata)
            fhi = len(self._fit_xdata)
        for i, meas in enumerate(self._meas_ydata):
            avg = meas[0][mlo:mhi]
            if meas[1] is not None:
                std = meas[1][mlo:mhi]
            else:
                std = 0
            fit = self._fit_ydata[i][flo:fhi]
            yall = [y for y in np.hstack([avg + std, avg - std, fit]) if y is not None]
            ymax = np.nanmax(yall)
            ymin = np.nanmin(yall)
            ydiff = ymax - ymin
            self._ax[i].set_ylim([ymin - 0.03 * ydiff, ymax + 0.03 * ydiff])
        return

    def _scale_xaxis(self):
        if self._fit_lims[0] > self._meas_xdata[0]:
            xlo = self._meas_xdata[0]
        else:
            xlo = self._fit_lims[0]
        if self._fit_lims[1] < self._meas_xdata[-1]:
            xhi = self._meas_xdata[-1]
        else:
            xhi = self._fit_lims[1]
        for i, meas in enumerate(self._meas_ydata):
            self._ax[i].set_xlim([xlo, xhi])
        return

    def _plot_fit_regions(self):
        if self._meas_xdata is None:
            return
        for i, dat in enumerate(self._meas_ydata):
            self._lospan[i].remove()
            self._hispan[i].remove()
            self._lospan[i] = self._ax[i].axvspan(
                self._fit_lims[0], self._meas_xdata[0], color='k', alpha=0.1, linewidth=0)
            self._hispan[i] = self._ax[i].axvspan(
                self._fit_lims[1], self._meas_xdata[-1], color='k', alpha=0.1, linewidth=0)
        self._scale_yaxis()
        self._scale_xaxis()
        self._fig.canvas.draw()
        return

    def _calc_err_metric(self):
        if self._calc_error:
            self._err_metric = 0
            return
        mlo, mhi = [np.argmin(abs(self._meas_xdata - fl)) for fl in self._fit_lims]
        self._err_incl = True
        num_vals = []
        denom_vals = []
        if np.any([self._meas_ydata[i][1] is None for i in range(len(self._meas_ydata))]):
            self._err_incl = False
        for i in range(len(self._fit_ydata)):
            if self._err_incl:
                num = (np.interp(self._meas_xdata[mlo:mhi], self._fit_xdata, self._fit_ydata[i]) -
                       self._meas_ydata[i][0][mlo:mhi]) ** 2 / (self._meas_ydata[i][1][mlo:mhi] + 1e-6) ** 2
                num_vals.append(num)
            else:
                num = (np.interp(self._meas_xdata[mlo:mhi], self._fit_xdata, self._fit_ydata[i]) -
                       self._meas_ydata[i][0][mlo:mhi]) ** 2
                denom = (self._meas_ydata[i][0][mlo:mhi] - np.mean(self._meas_ydata[i][0][mlo:mhi])) ** 2
                num_vals.append(num)
                denom_vals.append(denom)
        if self._err_incl:
            num = np.sum(num_vals)
            denom = np.sum([len(md[0][mlo:mhi]) for md in self._meas_ydata]) - len(list(self._params.keys()))
            self._err_metric = num / denom
            self._err_labs = [r"$\chi^{2}_{\mathrm{red}} =$", "red_chi_sq"]
        else:
            self._err_metric = 1 - (np.sum(num_vals) / np.sum(denom_vals))
            self._err_labs = [r"$R^{2}$", "R_squared"]
        return

    def _sort_ydata(self, data):
        if len(data) == 1:
            self._err_incl = False
            return [data[0], None]
        inds = np.argsort(np.mean(data, axis=1))
        lo_half = data[inds][:len(inds) // 2]
        hi_half = data[inds][len(inds) // 2:]
        if np.mean(lo_half) < 0.3 * np.mean(hi_half):  # data is means + stdevs
            avg = np.mean(hi_half, axis=0)
            stdev = np.sqrt(
                np.var(hi_half, axis=0) +
                np.sum(lo_half ** 2, axis=0) / len(lo_half))
        else:  # data is raw
            avg = np.mean(data, axis=0)
            stdev = np.std(data, axis=0)
        return [avg, stdev]

    def display(self):
        """
        A method to display plots and widgets to the executing Jupyter notebook
        via the 'output' attribute.
        """
        self._fig, self._ax = plt.subplots(self._numy, 1)
        self._fig.set_figheight(3 * self._numy)
        self._fig.set_figwidth(9.5)
        if self.output is not None:
            IPy.display.display(self._widget_box, self.output)
        return


def load_data(fname, fdata=None):
    if "CSV" in fname.upper():
        delim = ','
    elif "TXT" in fname.upper() or "DAT" in fname.upper():
        delim = None
    else:
        delim = None
    if fdata is None:
        with open(fname) as f:
            data_rows = [row.split(delim) for row in f.readlines()]
    else:
        data_rows = [row.split(delim) for row in fdata.split('\r\n')]
    num_entries = scp.stats.mode([len(row) for row in data_rows], keepdims=False)[0]
    all_float = [d for d in data_rows
                 if np.all([val.upper().split('E')[0].replace('.', '').isdigit() for val in d])
                 if len(d) == num_entries]
    return np.array(all_float).astype(float).T


def unpack_params(fname, fdata=None):
    if fdata is None:
        with open(fname) as f:
            data_rows = [row.split('|') for row in f.readlines()]
    else:
        data_rows = [row.split('|') for row in fdata.split('\r\n')]
    ldat = np.array(data_rows[4:-1][::2])
    params = {ldat[i][1].strip(): {
        "unit": ldat[i][2].strip(), "val": float(ldat[i][3]),
        "err": float(ldat[i][4]), "min": float(ldat[i][5]),
        "max": float(ldat[i][6]), "opt": eval(ldat[i][7].strip())}
        for i in range(len(ldat))}
    return params
