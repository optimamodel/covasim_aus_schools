import argparse
import concurrent.futures
import threading
import time
from pathlib import Path
import numpy as np

from celery import group
from tqdm import tqdm

import covasim_aus_schools as cvv
from covasim_aus_schools.commonwealth_schools.celery import run_sim, celery
from covasim_aus_schools import Samples

seed_offset = 0
debug_mode = False  # If True, run just one set of parameters and do not use threading
intpop = int(1e5)  # initial population size

result_dir = Path("school_final_results")

# GENERATE THE RUNS
possible_runs = cvv.run_generator(
    n_incursions=[1, 2, 3],
    incursion_layer=["primary_school", "high_school"],
    tracing_algorithm=["no_tracing",
                       "class_quarantine",
                       "class_test_to_stay",
                       "class_quarantine+test_to_stay",
                       "school_test_to_stay",
                       ],
    npis=[0, 0.25, 0.5],
    vaccine_coverage=["0_80_80_80_100", "60_80_80_80_100", "80_80_80_80_100", "0_60_60_80_100", "0_0_0_80_100", "0_80_80_80_80", "0_80_80_80_60", ],  # (5-11; 12-15; 16-17; Community 18+; teachers)
    surveillance=['none', "students", "teachers"],
    tts_compliance=[0, 0.25, 0.5, 0.75, 1],
    cross_classroom=['base', 'double', 'max'],
    symp_prob=[0.11, 0.16, 0.06],
    screening_frequency=[7, 3, 2, 1],
)

baseline = {
    'n_incursions': [1],
    'incursion_layer': ["primary_school", "high_school"],
    'tracing_algorithm': ["no_tracing", "class_test_to_stay"],
    'npis': [0],
    'vaccine_coverage': ["0_80_80_80_100"],  # (5-11; 12-15; 16-17; Community 18+; teachers)
    'surveillance': ['none'],
    'tts_compliance': [1],
    'cross_classroom': ['base'],
    'symp_prob': [0.11],
    'screening_frequency': [2.0]
}

# Work out which release coverages to skip

to_run = []
for kwargs in possible_runs:

    varied = lambda x: kwargs[x] not in baseline[x]  # Return True if the quantity is different to the baseline

    # Work out how many quantities are different to baseline
    n_varied = 0
    for k in kwargs:
        if kwargs[k] not in baseline[k]:
            n_varied += 1

    if n_varied <= 1:
        # If only one quantity is different, accept the run
        to_run.append(kwargs)
    elif n_varied == 2 and varied('n_incursions') and varied('surveillance'):
        # Run combinations of incursion rate and surveillance strategy
        to_run.append(kwargs)
    elif n_varied == 2 and varied('tts_compliance') and varied('surveillance'):
        # Run combinations of TTS compliance and surveillance strategy
        to_run.append(kwargs)
    elif n_varied == 2 and varied('cross_classroom') and varied('tracing_algorithm'):
        # Run combinations of cross-classroom interactions and contact management strategy
        to_run.append(kwargs)
    elif n_varied == 2 and varied('screening_frequency') and kwargs['surveillance'] == 'students':
        # Test screening frequency
        to_run.append(kwargs)


def run_scenario(kwargs):
    seeds = seed_offset + np.arange(args.nruns)
    rng = np.random.default_rng(seed_offset)

    betas = np.around(0.33247 + rng.standard_normal(args.nruns) * 0.02185, decimals=5)

    if not hasattr(thread_local, "pbar"):
        thread_local.pbar = tqdm(total=args.nruns)
    pbar = thread_local.pbar
    description = "-".join([str(x) for x in kwargs.values()])
    pbar.set_description(description)
    pbar.n = 0
    pbar.refresh()
    pbar.unpause()

    fname = description + ".zip"
    if (result_dir / fname).exists():
        return

    # Run simulations using celery
    job = group([run_sim.s(beta, seed, **kwargs) for beta, seed in zip(betas, seeds)])
    result = job.apply_async()
    ready = False

    while not ready:
        time.sleep(1)
        n_ready = sum(int(result.ready()) for result in result.results)
        ready = n_ready == len(seeds)
        pbar.n = n_ready
        if pbar.n == 0:
            pbar.reset(total=len(seeds))
        else:
            pbar.refresh()

    if result.successful():
        outputs = result.join()
        Samples.new(result_dir, outputs, kwargs.keys())
    else:
        pbar.set_description("-".join([str(x) for x in kwargs.values()]) + " ERROR")
        for x in result.results:
            if x.failed():
                with open(result_dir / f"error_{x.identifier}.txt", "w") as log:
                    log.write(str(x.__dict__))

    result.forget()

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nruns", default=8, type=int, help="Number of seeds to run per scenario")
    parser.add_argument("--celery", default=False, type=bool, help="If True, use Celery for parallelization")

    args = parser.parse_args()
    thread_local = threading.local()

    if debug_mode:
        # Use debug mode to run the full sampling over seeds, but without Celery
        run_scenario(to_run[0])

    elif args.celery:
        futures = []
        result_dir.mkdir(parents=True, exist_ok=True)

        with tqdm(total=len(to_run), desc=f"Total progress") as pbar:
            pbar.n = 0
            pbar.refresh()
            pbar.unpause()

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:

                for i, run_args in enumerate(to_run):
                    futures.append(executor.submit(run_scenario, run_args))
                    if i == 0:
                        time.sleep(5)

                while True:

                    done = [x for x in futures if x.done()]

                    for result in done:
                        if result.exception():
                            [x.cancel() for x in futures]
                            celery.control.purge()
                            celery.control.shutdown()
                            raise result.exception()

                    pbar.n = len(done)
                    pbar.refresh()
                    if len(done) == len(futures):
                        break
                    time.sleep(1)

        # Shut down the workers
        celery.control.shutdown()

    else:

        import matplotlib.pyplot as plt
        import pandas as pd
        import sciris as sc

        beta = 0.33522
        seed = 0

        kwargs = {
            "n_incursions": 1,
            "incursion_layer": "high_school",
            "tracing_algorithm": "class_test_to_stay",
            'npis': 0,
            "vaccine_coverage": "0_80_80_80_100",
            "surveillance": "none",
            "tts_compliance": 1,
            "cross_classroom": "base",
            "symp_prob": 0.11,
            "screening_frequency": 2,
        }

        df, summary, sim = run_sim(beta, seed, return_sim=True, **kwargs)

        print(df)

        # Results should match 1-high_school-class_test_to_stay-0-0_80_80_80_100-none-1-base-0.11-1.zip -> seed_0.csv

