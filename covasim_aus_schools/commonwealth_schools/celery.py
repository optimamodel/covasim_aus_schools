from celery import Celery, Task
from covasim import misc
import sciris as sc
import covasim as cv
from celery.signals import after_setup_task_logger
import logging
from covasim_aus_schools.commonwealth_schools import get_sim
import covasim_aus_schools as cvv
import numpy as np
import covasim.utils as cvu
import pandas as pd

misc.git_info = lambda: None  # Disable this function to increase performance slightly

import os

broker = os.getenv("COVID_REDIS_URL", "redis://127.0.0.1:6379")

# Create celery app
celery = Celery("commonwealth_schools")
celery.conf.broker_url = broker
celery.conf.result_backend = broker
celery.conf.task_default_queue = "commonwealth_schools"
celery.conf.accept_content = ["pickle", "json"]
celery.conf.task_serializer = "pickle"
celery.conf.result_serializer = "pickle"
celery.conf.worker_prefetch_multiplier = 1
celery.conf.task_acks_late = True  # Allow other servers to pick up tasks in case they are faster
# celery.conf.result_extended = True  # Capture the inputs in redis as well
celery.conf.worker_max_tasks_per_child = 5
celery.conf.worker_max_memory_per_child = 3000000

# Quieter tasks
@after_setup_task_logger.connect
def setup_task_logger(logger, *args, **kwargs):
    logger.setLevel(logging.WARNING)


@celery.task()
def run_sim(beta, seed, return_sim=False, **kwargs):
    """

    Args:
        beta

    Returns:
        - Dataframe with simulation output
        - Triggered levels, [(day, level)] specifying when level changes took place

    """

    sim = get_sim(beta, seed, **kwargs)
    sim.run(restore_pars=False)

    # Retrieve dataframe (per-timestep output to save in as the CSV for this run)
    df = cvv.result_df(sim)
    df["seed"] = seed

    # Retrieve summary states (scalar outputs to store in summary.csv)
    summary = sc.dcp(kwargs)
    summary["beta"] = beta
    summary["seed"] = seed

    final_day_quantities = [
        "new_diagnoses_14d_avg",
        "new_diagnoses_14d_avg",
        "new_diagnoses_7d_avg",
        "cum_diagnoses",
        "cum_infections",
        "cum_severe",
        "cum_critical",
        "cum_deaths",
        "cum_school_infections",
        "cum_school_diagnoses",
        "cum_community_infections",
        "cum_community_diagnoses",
        "cum_person_days_lost",
    ]

    for quantity in final_day_quantities:
        summary[quantity] = df.iloc[-1][quantity]

    summary["peak_new_diagnoses_14d_avg"] = np.nanmax(df["new_diagnoses_14d_avg"])
    summary["peak_new_diagnoses_7d_avg"] = np.nanmax(df["new_diagnoses_7d_avg"])
    summary["peak_severe"] = np.nanmax(df["n_severe"])
    summary["peak_critical"] = np.nanmax(df["n_critical"])
    summary["worker_hostname"] = celery.current_task.request.hostname

    # --------------------------
    # SCHOOL OUTPUTS

    # Day of first diagnosis
    if summary["cum_school_diagnoses"] > 0:
        day_first_school_diagnosed = (df["cum_school_diagnoses"]>0).argmax()
    else:
        day_first_school_diagnosed = np.nan

    summary["day_first_school_diagnosed"] = day_first_school_diagnosed

    summary["day_first_school_diagnosed"] = day_first_school_diagnosed

    # How long did it take to reach various diagnosis triggers
    # relative to the first school diagnosis
    for threshold in [5,10,20,50]:
        if np.max(df["cum_school_diagnoses"]) > threshold:
            summary[f"time_to_diag_{threshold}"] = (df["cum_school_diagnoses"]>=threshold).argmax() - day_first_school_diagnosed
        else:
            summary[f"time_to_diag_{threshold}"] = None

        if np.max(df["cum_school_infections"]) > threshold:
            summary[f"time_to_inf_{threshold}"] = (df["cum_school_infections"]>=threshold).argmax() - day_first_school_diagnosed
        else:
            summary[f"time_to_diag_{threshold}"] = None

    if return_sim:
        return df, summary, sim
    else:
        return df, summary


    # check school_5_in_14days using time_to_diag distribution of values
    #
    # # Check if it reached 5 in 14 (or something)
    # # It's None if there wasn't enough time since the first diagnosis
    # if day_first_school_diagnosed > 30:
    #     summary["school_5_in_14days"] = None
    # else:
    #     if df["cum_school_diagnoses"].max() < 5:
    #         summary["school_5_in_14days"] = False
    #     else:
    #         summary["school_5_in_14days"] = (df["cum_school_diagnoses"]>=5).argmax() < (day_first_school_diagnosed+14)

    #
    # def days_to_primorhigh(n_schools, data, x_value):
    #     """
    #     Return the number of days until x cumulative diagnoses or infections for one type of school (primary or high)
    #     Returns as a list where the order matches the order of the schools in sim
    #     If a school never reaches x diagnoses or cases, assigns nan
    #     """
    #     days_to_x = []
    #     for indsch in range(n_schools):
    #         daysx = list(np.where(data.loc[:, indsch] >= x_value))  # get the indices of days > X
    #         if len(daysx[0]) == 0:
    #             days_to_x.extend([np.nan])  # if school doesn't get one case, return na
    #         else:
    #             days_to_x.extend([daysx[0][0]])  # if school does get one case, return the index
    #     return days_to_x
    #
    #
    # def days_to(n_prim, n_high, analyzer_data, x_value, diagnoses_or_infections):
    #     """
    #     Return the number of days until x cumulative diagnoses or infections for every school
    #     Returns as a list where the order matches the order of the schools in sim, where:
    #     Primary schools come first in list, then high schools: can see which it is by using nprim or nhigh
    #     """
    #     if diagnoses_or_infections == "diagnoses":
    #         data_prim = analyzer_data.df_primary_diagnosed
    #         data_high = analyzer_data.df_high_diagnosed
    #     elif diagnoses_or_infections == "infections":
    #         data_prim = analyzer_data.df_primary_infected
    #         data_high = analyzer_data.df_high_infected
    #
    #     days_to_x_prim = days_to_primorhigh(n_prim, data_prim, x_value)
    #     days_to_x_high = days_to_primorhigh(n_high, data_high, x_value)
    #
    #     days_to_x = (days_to_x_prim + days_to_x_high)
    #
    #     return days_to_x
    #
    #
    # def find_first_school(days_to_x, n_prim):
    #     """
    #     Find the first school to reach x cumulative diagnoses or infections and return the index
    #     Return whether it is a high school or a primary school
    #     This will the school 'of interest' for the rest of the simulation, as we consider 1 school per sim
    #     """
    #     if np.isnan(days_to_x).all() == True:
    #         first_school_id = np.nan
    #         first_school_type = "NaN"
    #     else:
    #         first_school_id = np.nanargmin(days_to_x)
    #         if first_school_id < n_prim:
    #             first_school_type = "primary"
    #         elif first_school_id > n_prim:
    #             first_school_type = "high"
    #
    #     return first_school_id, first_school_type
    #
    #
    # sch_data = sim.get_analyzer(cvv.CumulativeCasesBySchool)
    # n_prim, n_high = len(sch_data.df_primary_diagnosed.columns), len(sch_data.df_high_diagnosed.columns)  # number of schools
    #
    # # Get days to one case for all schools, then the ID of the school that reaches one case first. store whether it is primary or high school
    # daysto1_diag = days_to(n_prim, n_high, sch_data, 1, "diagnoses")
    # sch_id_diag, sch_type_diag = find_first_school(daysto1_diag, n_prim)
    # summary["school_type"] = sch_type_diag
    #
    # # Get days to X diagnoses
    # xoptions = [5, 10]
    #
    # xdays = []
    # for xval in range(len(xoptions)):
    #     if not np.isnan(sch_id_diag):
    #         xdays.append(days_to(n_prim, n_high, sch_data, xoptions[xval], "diagnoses")[sch_id_diag])
    #     else:
    #         xdays.append(np.nan)
    #
    #
    # # Add the cumulative diagnoses and infections at each t to df for the school of iterest
    # # If the index for the school of interest is nan, populate with 0 because this is equivalent to all schools having no cases as no school reaches 1
    # # This code isn't very efficient just now; need to revise
    #
    # # subset the data for our target school
    # if sch_type_diag=="primary":
    #     data_export_diag = list(sch_data.df_primary_diagnosed.loc[:,sch_id_diag])
    #     data_export_infe = list(sch_data.df_primary_infected.loc[:, sch_id_diag])
    # elif sch_type_diag=="high":
    #     data_export_diag = list(sch_data.df_high_diagnosed.loc[:,(sch_id_diag-n_prim)])
    #     data_export_infe = list(sch_data.df_high_infected.loc[:, (sch_id_diag - n_prim)])
    #
    #
    # # add:
    # #  - the cumulative diagnoses and infections over time
    # #  - the days to 1 or x diagnoses
    # #  - the number of cumulative infections at 1 or x diagnoses
    # if ~np.isnan(sch_id_diag):
    #     df["cum_diag_firstto1"] = data_export_diag
    #     df["cum_infe_firstto1"] = data_export_infe
    #
    #     dayto1 = daysto1_diag[sch_id_diag]
    #     summary["days_to_1_diag"] = dayto1
    #     if ~np.isnan(dayto1):
    #         summary["infe_at_1_diag"] = data_export_infe[dayto1]
    #     else:
    #         summary["infe_at_1_diag"] = np.nan
    #
    #     for xval in range(len(xoptions)):
    #         new_name_daystodiag = "days_to_" + str(xoptions[xval]) + "_diag"
    #         new_name_infeatdiag = "infe_at_" + str(xoptions[xval]) + "_diag"
    #         daytox = xdays[xval]
    #         summary[new_name_daystodiag] = daytox
    #         if ~np.isnan(daytox):
    #             summary[new_name_infeatdiag] = data_export_infe[daytox]
    #         else:
    #             summary[new_name_infeatdiag] = np.nan
    # else:
    #     df["cum_diag_firstto1"] = 0
    #     df["cum_infe_firstto1"] = 0
    #     summary["days_to_1_diag"] = np.nan
    #     summary["infe_at_1_diag"] = np.nan
    #     for xval in range(len(xoptions)):
    #         new_name_daystodiag = "days_to_" + str(xoptions[xval]) + "_diag"
    #         new_name_infeatdiag = "infe_at_" + str(xoptions[xval]) + "_diag"
    #         summary[new_name_daystodiag] = np.nan
    #         summary[new_name_infeatdiag] = np.nan
    #
    #
    # # Get the days from 1 case until X cases
    # for xval in range(len(xoptions)):
    #     new_name = "days_from_1_to_" + str(xoptions[xval]) + "_diag"
    #     name_x = "days_to_" + str(xoptions[xval]) + "_diag"
    #     summary[new_name] = summary[name_x] - summary["days_to_1_diag"]
    #
    # # Boolean flag: has the school reached 5 diagnoses in 14 days from 1st diagnosis?
    # if ~np.isnan(sch_id_diag):
    #     if (daysto1_diag[sch_id_diag] + 14) > len(df["cum_diag_firstto1"]):
    #         ind1, ind2 = daysto1_diag[sch_id_diag], len(df["cum_diag_firstto1"])
    #         summary["school_5_in_14days"] = any(y >= 5 for y in list(df["cum_diag_firstto1"])[ind1:ind2])
    #     else:
    #         ind1, ind2 = daysto1_diag[sch_id_diag], daysto1_diag[sch_id_diag] + 14
    #         summary["school_5_in_14days"] = any(y >= 5 for y in list(df["cum_diag_firstto1"])[ind1:ind2])


