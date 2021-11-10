# Setup directions

- Run `pip install -e .` from root directory
- Use `run_scenarios.py` to run the model. Parallelization can be achieved with Celery (running a worker from `celery.py` and calling `run_scenarios.py` with `celery=True`) 

The main simulation configuration is in `covasim_aus_schools/commonwealth_schools/main.py`