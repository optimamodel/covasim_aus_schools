import pandas as pd
import numpy as np
import sciris as sc
import kalepy
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
import itertools
import covasim_aus_schools as cvv
import matplotlib.pyplot as plt


def savefig(fname):
    plt.savefig(fname, bbox_inches="tight", dpi=300, transparent=False)


def parse_age_range(x):
    if "+" in x:
        age_lower = float(x.split("+")[0])
        age_upper = np.inf
    else:
        age_lower = float(x.split("-")[0])
        age_upper = float(x.split("-")[1])
    return age_lower, age_upper


def rolling_average(x, window):
    # Rolling average, right-aligned, with given window length
    # Unlike pd.Series.rolling(window) this function pads the array
    # with zeros at the start
    #
    # array([0.54058639, 1.25067553, 0.78450521, 0.65713343, 1.21422499,
    #        -1.30679364, -0.1119663, -1.64590603, 1.30028759, 1.40006004,
    #        0.45048084, 0.77187634, -1.28375399, 1.13676772])
    # pd.Series(x).rolling(7).mean().values
    # array([       nan,        nan,        nan,        nan,        nan,
    #               nan, 0.43262366, 0.1202676 , 0.12735504, 0.21529144,
    #        0.18576964, 0.12257698, 0.12586836, 0.30425893])
    # rolling(x,7)
    # array([0.07722663, 0.25589456, 0.36796673, 0.46184294, 0.63530365,
    #        0.44861885, 0.43262366, 0.1202676, 0.12735504, 0.21529144,
    #        0.18576964, 0.12257698, 0.12586836, 0.30425893]

    return np.convolve(np.concatenate([np.zeros(window - 1), x]), np.ones(window) / window, mode="valid")


class CalibrationMismatch(Exception):
    """
    Exception raised if the run is not consistent with the data
    """

    pass


def result_df(sim):
    resdict = sim.export_results(for_json=False)
    result_df = pd.DataFrame.from_dict(resdict)
    result_df.index = sim.datevec[0 : len(result_df)]
    result_df.index.name = "date"
    result_df["rescale_vec"] = sim.rescale_vec
    result_df["Date"] = sim.datevec[0 : len(result_df)]  # remove this if it breaks things
    return result_df


def save_csv(sim, fname):
    df = result_df(sim)
    df.to_csv(fname)


def bootstrap_distribution(samples: np.ndarray, reduce_func, n_bootstrap=1000):
    """
    Construct bootstrap distribution of statistic from samples

    This functions runs the `reduce_func` on `n_bootstrap` resampled datasets and thus produces a distribution
    for the statistic being produced by the resampling. For example, the reduce_func could be `np.average` in
    which case, this function would return a possible distribution for the mean of the dataset if different
    samples had been used.

    Args:
        samples: Original samples (these will be resampled for the bootstrap method
        reduce_func: A function that takes an array of samples and returns a scalar (e.g. mean, proportion>threshold, etc.)
        n_bootstrap: Number of bootstrap iterations

    Returns: Sampled values of the reduce_func

    """

    output = []
    for i in range(n_bootstrap):
        bootstrap_sample = np.random.choice(samples, size=len(samples), replace=True)
        output.append(reduce_func(bootstrap_sample))
    return output


### KERNEL DENSITY ESTIMATES
def fit_kde(vals, bandwidth=None):
    vals = sc.promotetoarray(vals).ravel()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        raise Exception("No points remained after removing NaNs")
    if np.all(vals >= 0):
        kde = kalepy.KDE(vals, reflect=[0, None], bandwidth=bandwidth)
    else:
        kde = kalepy.KDE(vals, bandwidth=bandwidth)
    return kde


def plot_pdf(vals, res=1000, bandwidth=None, *args, **kwargs):
    kwargs = sc.dcp(kwargs)
    vals = sc.promotetoarray(vals)
    kde = fit_kde(vals, bandwidth=bandwidth)
    if "x" in kwargs:
        x = kwargs.pop("x")
    elif np.all(vals >= 0):
        x = np.linspace(0, np.nanmax(vals), res)
    else:
        x = np.linspace(np.nanmin(vals), np.nanmax(vals), res)
    return plt.plot(x, kde.pdf(x)[1], *args, **kwargs)


def plot_cdf(vals, res=1000, bandwidth=None, *args, **kwargs):
    vals = sc.promotetoarray(vals)
    kde = fit_kde(vals, bandwidth=bandwidth)
    if np.all(vals >= 0):
        x = np.linspace(0, np.nanmax(vals), res)
    else:
        x = np.linspace(np.nanmin(vals), np.nanmax(vals), res)
    plt.plot(x, kde.cdf(x), *args, **kwargs)


def plot_quantiles(res, color, alpha, quantity, label=None, step=0.05, ax=None, show_median=True, subsample=1):
    """

    Args:
        res:
        color:
        alpha:
        quantity:
        label:
        step: Spacing between shaded levels e.g. 0.05. Can specify an array or list of explicit quantile levels e.g. [0.45, 0.25] to plot 5-95 and 25-75 only
        ax:
        show_median:
        subsample: Use only a portion of the results for plotting (deterministically)

    Returns:

    """

    if ax is None:
        ax = plt.gca()
    vals = []

    seeds = res.seeds[0:len(res):int(1/subsample)]
    for seed in seeds:
        vals.append(res[seed][quantity].values)
    vals = np.array(vals)
    if isinstance(step, np.ndarray) or isinstance(step, list):
        levels = step
    else:
        levels = np.arange(step, 0.50, step)
    x = np.arange(0, vals.shape[1])
    for level in levels[::-1]:
        ax.fill_between(x, y1=np.quantile(vals, 0.5 - level, axis=0), y2=np.quantile(vals, 0.5 + level, axis=0), linewidth=0, alpha=alpha, color=color)
    if label is None:
        label = res.identifier
    if show_median:
        l = ax.plot(x, np.median(vals, axis=0), color=color, label=label)[0]
    else:
        l = None
    return l


### Binomial CI


def binomial_ci(outcomes, alpha=0.05):
    """
    Return mean and CI for the

    Args:
        outcomes: Boolean list/array of outcomes
        alpha:

    Returns: mean, (low, high)

    """

    outcomes = sc.promotetoarray(outcomes)
    mean = np.mean(outcomes)
    low, high = proportion_confint(np.sum(outcomes), len(outcomes), alpha=alpha)
    return mean, (low, high)


## Scenario helper


def run_generator(**kwargs):
    """
    Generate parameters for runs

    Takes in arguments with lists of options. Returns a list of dicts with all combinations of options

    e.g.

    >>> run_generator(a=[1,2,3],b=['x','y','z'])

    would return

    `[{'a':1,'b':'x'},{'a':2,'b':'x'},{'a':3,'b':'x'},{'a':1,'b':'y'},...]`

    """
    arg_names = list(kwargs.keys())
    tmp = itertools.product(*(list(kwargs.values())))
    to_run = [dict([(x, y) for x, y in zip(arg_names, z)]) for z in tmp]
    return to_run


### Cache runs

import functools
import pathlib
import hashlib
import pickle
import sciris as sc


def cachefile(verbose=False, cache_folder=".cache", modules=None):
    """

    Args:
        verbose: If True, print extra outputs
        cache_folder: Specify location for cache files
        modules: Optionally specify a collection of modules with .__version__ attributes to add to the hash

    E.g.
    @cachefile
    def add(x,y):
        return x+y
    """

    cache_folder = pathlib.Path(cache_folder).absolute()
    cache_folder.mkdir(parents=True, exist_ok=True)

    def decorator(fcn):
        @functools.wraps(fcn)
        def wrapper(*args, **kwargs):

            params = [args, kwargs]

            if modules is not None:
                params.extend(x.__version__ for x in modules)
                verbose and print(f"Added {params[2:]} to hash")

            try:  # try generating a hash of the input parameters
                hash = hashlib.sha1(pickle.dumps(tuple(params))).hexdigest()[0:8]
            except pickle.PicklingError:
                verbose and print("Argument is not picklable, will not cache function run")
                return fcn(*args, **kwargs)

            fname = cache_folder / f"{fcn.__name__}_{hash}"

            try:  # try loading previous results
                verbose and print(f"Trying cache file {fname}")
                cache_object = sc.loadobj(fname, die=True)
                return cache_object["result"]
            except (FileNotFoundError, pickle.UnpicklingError):
                verbose and print("Cache file not found, re-running function")
                val = {"result": fcn(*args, **kwargs), "args": (args, kwargs)}
                try:
                    sc.saveobj(fname, val)
                    verbose and print(f"Saved cache file for next run to {fname}")
                except Exception as E:
                    print(f"Cache error: {str(E)}")

                return val["result"]

        return wrapper

    return decorator


## OTHER UTILITIES


def dict_equal(a, b):
    # Quick snippet to check equality of objects
    if pickle.dumps(a) == pickle.dumps(b):
        return True

    if set(a.keys()) != set(b.keys()):
        return False

    for k in a:
        # print(k)
        if isinstance(a[k], np.ndarray):
            if (a[k] == b[k]).all():
                continue
            else:
                # print(k)
                return False
        elif isinstance(a[k], cvv.VictoriaLayer):
            if not dict_equal(a[k].__dict__, b[k].__dict__):
                return False
        elif a[k] != b[k]:
            # print(k)
            if np.isnan(a[k]) and np.isnan(b[k]):
                continue
            # print(a[k])
            # print(b[k])
            print(k)
            return False

    return True
