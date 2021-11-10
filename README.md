# School outbreak analysis

This repository contains code to reproduce the results in "Work package 2.3: Schools" (Attachment D) as part of the Doherty Institute modelling report to National Cabinet 5th November 2021. 

# Usage

- Run `pip install -e .` from root directory to set up the package
- Use `run_scenarios.py` to run the model. By default, the script is set up to perform a single simulation. Parallelization and bulk execution can be achieved with Celery (running a worker with `celery.py` and separately calling `run_scenarios.py` with `celery=True`) 

The main simulation configuration is in `covasim_aus_schools/commonwealth_schools/main.py`. The synthetic population is generated in `covasim_aus_schools/commonwealth_schools/people.py`