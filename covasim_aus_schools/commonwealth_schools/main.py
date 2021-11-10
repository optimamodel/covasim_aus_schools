import covasim as cv
import pandas as pd
import covasim_aus_schools as cvv
import covasim.misc as cvm
import sciris as sc
from functools import partial
import logging
import networkx as nx
import covasim.utils as cvu
from .people import get_people, get_layers
import numpy as np
import cykhash
from . import rootdir
from covasim_aus_schools import logger
from covasim_aus_schools.analyzers import add_result

# get the simulation. have removed extra interventions that were loaded. keep blank for now

PCR_sensitivity = 0.87
RAT_sensitivity = 0.773  # Reduced sensitivity, Muhi et al. 2021 The Lancet Regional Health-Western Pacific, 9, 100115

def get_sim(
    beta,
    seed,
    incursion_layer: str = None,
    tracing_algorithm: str = None,
    vaccine_coverage: str = "0_80_80_80_80",
    npis: float = 0, # Fraction reduction in school-related beta values
    surveillance=None,
    n_incursions=1,
    tts_compliance=1, # test-to-stay compliance
    cross_classroom='base',
    symp_prob=0.11,
    screening_frequency=2, # times to screen per week
) -> cv.Sim:
    """
    Get the baseline simulation, based on Victoria

    Args:
        beta: The beta value to use for the simulation
        seed: The seed to use for the simulation run
        people: A pre-generated `People` instance e.g. from `get_victoria_people()`
        n_incursions: Number of simultaneous incursions into the same school
        incursion_layer: Specify the incursion must take place in 'primary_school' or 'high_school'
        response_strategy: Passed to the `Response` intervention

    Returns: A `cv.Sim` ready to run

    """

    people = get_people(seed, int(1e5), cross_classroom=cross_classroom)  # load the seed and the initial population size

    # define additional parameters
    pars = {}
    pars["pop_size"] = len(people)
    pars["beta"] = beta
    pars["rand_seed"] = seed
    pars["verbose"] = 0
    pars["n_imports"] = 0  # Number of imports per day
    pars["pop_infected"] = 0

    # Disable scaling (since we focus on small outbreaks)
    pars["pop_scale"] = 1
    pars["rescale"] = False

    # Imperial College age-specific wild type disease prognosis estimates: https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-16-COVID19-Report-9.pdf
    # Knock et al to calculate age-specific pr{hospitalization and ICU | infection with wild type)
    # Adjusting disease prognosis and pr(hospitalization or ICU | infection) for more severe outcomes with delta, based on a Canadian study (OR = 2.08 being applied to hospitalization, ICU and deaths): https://www.medrxiv.org/content/10.1101/2021.07.05.21260050v3.full.pdf
    # Assuming that in the model, only a percentage of people with severe or critical disease end up in hospital or ICU, according to age-specific pr(hosp|infection) and pr(ICU|infection) from Knock.
    prognoses = dict(
        age_cutoffs=np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90]),  # Age cutoffs (lower limits)
        sus_ORs=np.array([0.413, 0.500, 0.585, 0.708, 1, 1, 1, 1, 1, 1, 1, 1, 1.24, 1.24, 1.47, 1.47, 1.47, 1.47]),  # Values for 0-20 estimated by Nick Golding
        trans_ORs=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        # Odds ratios for relative transmissibility -- no evidence of differences
        comorbidities=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        # Comorbidities by age -- set to 1 by default since already included in disease progression rates
        symp_probs=np.array([0.28, 0.28, 0.20, 0.20, 0.26, 0.26, 0.33, 0.33, 0.40, 0.40, 0.49, 0.49, 0.63, 0.63, 0.69, 0.69, 0.69, 0.69]),
        severe_probs=np.array([0.0005, 0.0005, 0.00096, 0.00165, 0.00720, 0.00720, 0.0208, 0.0208, 0.0343, 0.0343, 0.0765, 0.0765, 0.1328, 0.1328, 0.20655, 0.20655, 0.2457, 0.2457]),
        crit_probs=np.array([0.00003, 0.00003, 0.00005, 0.00008, 0.00036, 0.00036, 0.00104, 0.00104, 0.00216, 0.00216, 0.00933, 0.00933, 0.03639, 0.03639, 0.08923, 0.08923, 0.1742, 0.1742]),
        death_probs=np.array([0.00002, 0.00002, 0.00002, 0.00002, 0.00010, 0.00010, 0.00032, 0.00032, 0.00098, 0.00098, 0.00265, 0.00265, 0.00766, 0.00766, 0.02439, 0.02439, 0.08292, 0.1619]),
        hosp_given_severe=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        ICU_given_critical=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.72, 0.71, 0.10, 0.10]),
    )

    def increase_OR(p_old, ratio):
        # Compute p_new, such that (p_new/(1-p_new))/(p_old/(1-p_old)) = ratio
        x = ratio * p_old / (1 - p_old)
        p_new = x / (1 + x)
        return p_new

    # Adjusting disease prognosis and pr(hospitalization or ICU | infection) for more severe outcomes with delta, based on a Canadian study (OR = 2.08 being applied to hospitalization, ICU and deaths): https://www.medrxiv.org/content/10.1101/2021.07.05.21260050v3.full.pdf
    prognoses["severe_probs"] = increase_OR(prognoses["severe_probs"], 2.08)
    prognoses["crit_probs"] = increase_OR(prognoses["crit_probs"], 2.08)
    prognoses["death_probs"] = increase_OR(prognoses["death_probs"], 2.08)

    prognoses["death_probs"] /= prognoses["crit_probs"]  # Conditional probability of dying, given critical symptoms
    prognoses["crit_probs"] /= prognoses["severe_probs"]  # Conditional probability of symptoms becoming critical, given severe
    prognoses["severe_probs"] /= prognoses["symp_probs"]  # Conditional probability of symptoms becoming severe, given symptomatic
    pars["prognoses"] = prognoses

    # Add the beta values
    layers = get_layers()
    pars["beta_layer"] = {k: v.baseline_beta for k, v in people.contacts.items()}
    pars["quar_factor"] = {k: v["quar_factor"] for k, v in layers.items()}
    pars["iso_factor"] = {k: v["iso_factor"] for k, v in layers.items()}
    pars["asymp_factor"] = 0.5  # add asymptomatic beta

    # Apply NPI reduction in beta
    pars["beta_layer"]['primary_school'] *= (1-npis)
    pars["beta_layer"]['high_school'] *= (1-npis)

    # # Extra keys that Covasim needs but which aren't used
    pars["contacts"] = dict.fromkeys(layers, np.nan)
    pars["dynam_layer"] = dict.fromkeys(layers, False)  # This should get ignored because People doesn't use it

    pars["interventions"] = []

    # ADD BACKGROUND TESTING
    asymp_prob = 10000 / 6.7e6

    testing_intervention = cvv.test_prob_quarantine(
        symp_prob=symp_prob,
        asymp_prob=asymp_prob,
        symp_quar_prob=1,  # Optimistically assume anyone in quarantine will test immediately
        asymp_quar_prob=asymp_prob,
        sensitivity=PCR_sensitivity,
        test_delay=1,  # Poisson distribution mean, noting that the actual test delay has a minimum of 1 day
        quarantine_compliance=1,
        vac_symp_prob=1.0*symp_prob, # Assume no reduction in vaccinated symptomatic test probability for now
        label="normal_testing",
    )
    pars["interventions"].append(testing_intervention)

    ## ADD DAILY SCREENING INTERVENTION
    if surveillance == "teachers":
        surveillance_inds = np.concatenate([people.contacts["primary_school"].teachers, people.contacts["high_school"].teachers])
    elif surveillance == "students":
        surveillance_inds = np.concatenate([people.contacts["primary_school"].students, people.contacts["high_school"].students])
    elif surveillance == "none":
        surveillance_inds = []
    else:
        raise Exception("Unknown daily surveillance")

    surveillance_testing = cvv.test_prob_screening(
        inds=surveillance_inds,
        test_prob=screening_frequency/7,
        sensitivity=RAT_sensitivity,  # Reduced sensitivity, lower bound of Muhi et al. 2021 The Lancet Regional Health-Western Pacific, 9, 100115
        test_delay=0,  # Poisson distribution mean, noting that the actual test delay has a minimum of 1 day
        label="surveillance",
    )
    pars["interventions"].append(surveillance_testing)


    # Add general population tracing intervention
    pars["interventions"].append(
        cvv.SecondRingTracing(
            testing_intervention=testing_intervention,
            trace_probs={k: v["trace_prob"] for k, v in layers.items() if k not in {"primary_school", "high_school"}},
            trace_time={k: v["trace_time"] for k, v in layers.items() if k not in {"primary_school", "high_school"}},
            capacity_levels=np.array((0, 25, 75, 150, 500, np.inf)),  # step function for changes in contact tracing efficacy
            capacity_fraction=np.array((1.0, 0.8, 0.5, 0.3, 0.2, 0.2)),  # fraction of cases traced at different tracing capacity levels
            second_ring_layers=[],  # no second ring tracing in future
            unlimited_capacity_layers=["H", "child_care"],  # case-initiated notifications in these layers only
            label="general_tracing",
        )
    )

    # ADD SCHOOL TESTING INTERVENTION
    if tracing_algorithm == "no_tracing":
        quarantine_policy = "none"
        test_policy = "none"
    elif tracing_algorithm == "class_quarantine":
        quarantine_policy = "class"
        test_policy = "none"
    elif tracing_algorithm == "class_quarantine+test_to_stay":
        quarantine_policy = "class"
        test_policy = "class"
    elif tracing_algorithm == "close_quarantine":
        quarantine_policy = "close"
        test_policy = "none"
    elif tracing_algorithm == "class_test_to_stay":
        quarantine_policy = "none"
        test_policy = "class"
    elif tracing_algorithm == "close_test_to_stay":
        quarantine_policy = "none"
        test_policy = "close"
    elif tracing_algorithm == "test_class+school_one_off":
        quarantine_policy = "none"
        test_policy = "class+school_one_off"
    elif tracing_algorithm == "school_test_to_stay":
        quarantine_policy = "none"
        test_policy = "school"
    elif tracing_algorithm == "unvaccinated_quarantine":
        quarantine_policy = "unvaccinated"
        test_policy = "none"
    elif tracing_algorithm == "unvac_quar_test_to_stay":
        quarantine_policy = "unvaccinated"
        test_policy = "class"
    else:
        raise Exception(f"Unknown tracing algorithm '{tracing_algorithm}")

    test_to_stay = cvv.test_prob_screening(
        inds=[],
        test_prob=1,  # Test daily
        sensitivity=tts_compliance*RAT_sensitivity,
        test_delay=0,
        label="test_to_stay",
    )

    pars["interventions"].append(test_to_stay)

    # Intervention with 1 day dlay
    delayed_antigen = cvv.test_prob_screening(
        inds=[],
        test_prob=1,  # Test daily
        sensitivity=RAT_sensitivity,
        test_delay=1,
        label="delayed_antigen",
    )
    pars["interventions"].append(delayed_antigen)

    # Note that school tracing appears after the testing interventions
    # Therefore, when test-to-stay is turned on with delay 0, it only takes
    # effect at the next timestep. In contrast, `delayed_antigen._test()` is called
    # immediately
    pars["interventions"].append(
        SchoolTracing(
            quarantine_policy=quarantine_policy,
            test_policy=test_policy,
            daily_test_intervention=test_to_stay,
            delayed_antigen_intervention=delayed_antigen,
            label="school_tracing",
        )
    )

    ## ADD VACCINE INTERVENTIONS

    # randomly sample from people to return boolean array of all people indicated true/false for vaccinated/not-vaccinated according to the vax coverage in categories above
    to_vax = baseline_vac_eligible(people, vaccine_coverage)

    logger.debug(f"{sum(to_vax)} people will be vaccinated at baseline")

    # define two interventions; one for Pfizer and one for AZ
    pars["interventions"].append(cvv.TimedVaccinationProgram(vaccine=cvv.Vaccine.pfizer(), sequence=[], num_doses=0, label="pfizer"))
    pars["interventions"].append(cvv.TimedVaccinationProgram(vaccine=cvv.Vaccine.astra_zeneca(), sequence=[], num_doses=0, label="az"))

    # ADD EXTRA OUTPUTS
    pars["analyzers"] = [
        cvv.RollingAverageDiagnoses(),
        SchoolAnalyzer(label='school_analyzer'),
    ]

    pars["start_day"] = 0
    pars["n_days"] = 45

    # Make the base sim
    sim = cv.Sim(
        pars=pars,
        popfile=people,
        load_pop=True,
    )

    # Serial interval
    sim.pars["dur"]["exp2inf"] = dict(dist="lognormal_int", par1=3.71, par2=0.99)  # parameters based on Li et al. (2021). "Viral infection and Transmission..." https://doi.org/10.1101/2021.07.07.21260122

    # Hospital durations
    sim.pars["dur"]["sev2rec"] = dict(dist="lognormal_int", par1=7.8, par2=8.3)  # Duration for people with severe symptoms to recover, 24.7 days total; see Verity et al., https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30243-7/fulltext; 18.1 days = 24.7 onset-to-recovery - 6.6 sym2sev; 6.3 = 0.35 coefficient of variation * 18.1; see also https://doi.org/10.1017/S0950268820001259 (22 days) and https://doi.org/10.3390/ijerph17207560 (3-10 days)

    # ICU survivors - total hospital stay - 18.7, std=16.2.
    # ICU survivors - ICU stay - median=4, IQR=2-14. Assuming lognormal, equivalent mean/SD:
    # In[31]: np.percentile(cv.sample(dist='lognormal_int', par1=11, par2=25, size=100000), [25, 50, 75])
    # Out[31]: array([2., 4., 11.])
    sim.pars["dur"]["crit2rec"] = dict(dist="lognormal_int", par1=11.3, par2=27.9)  # Duration for people with critical symptoms to recover; as above (Verity et al.)

    # Retain previous sev2crit
    sim.pars["dur"]["sev2crit"] = dict(dist="lognormal_int", par1=1.5, par2=2.0)  # Duration from severe symptoms to requiring ICU; average of 1.9 and 1.0; see Chen et al., https://www.sciencedirect.com/science/article/pii/S0163445320301195, 8.5 days total - 6.6 days sym2sev = 1.9 days; see also Wang et al., https://jamanetwork.com/journals/jama/fullarticle/2761044, Table 3, 1 day, IQR 0-3 days; std=2.0 is an estimate

    # ICU non-survivors - 9 (IQR 5-19)
    # In[43]: np.percentile(cv.sample(dist='lognormal_int', par1=15, par2=19, size=100000), [25, 50, 75])
    # Out[43]: array([5., 9., 18.])
    sim.pars["dur"]["crit2die"] = dict(dist="lognormal_int", par1=16.1, par2=22.6)  # Duration from critical symptoms to death, 18.8 days total; see Verity et al., https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30243-7/fulltext; 10.7 = 18.8 onset-to-death - 6.6 sym2sev - 1.5 sev2crit; 4.8 = 0.45 coefficient of variation * 10.7

    sim.initialize()

    # APPLY VACCINATION

    pfizer = sim.get_intervention("pfizer")  # this is necessary because sim.initialize() deep copies the objects, so we need to retrieve the one that is actually contained in the sim
    az = sim.get_intervention("az")

    # load csv of proportion of vaccinated people getting each vaccine by age
    vax_type = pd.read_csv(rootdir / "vax_type_proportions.csv")  # load proportion of people vaxxed receiving az and pf

    # get the indices of the people recieving pfizer and the people recieving az
    #  each element in list corresponds to list of indices of vaccinated with that vaccine for each age cat
    inds_pf, inds_az = vaccine_type_indices(vax_type, people, to_vax)

    # concatenate the lists
    inds_pf = list(np.concatenate(inds_pf).flat)
    inds_az = list(np.concatenate(inds_az).flat)

    # flag an error if the number of people for sum of az and pfizer are not the same as the total number of people to be vaxxed
    if (len(inds_pf) + len(inds_az)) != sum(to_vax):
        raise Exception("Number of people to vaccinate has not been computed correctly")

    # carry out vaccination
    pfizer.vaccinate(sim, inds_pf, t=-180)  # e.g. vaccinating 6 months ago. Note that at the moment there is no waning immunity
    az.vaccinate(sim, inds_az, t=-180)

    ## SEED INITIAL INFECTION

    assert incursion_layer in {'primary_school', 'high_school'}

    # First, select a school to seed
    school_id = int(np.random.choice(len(people.contacts[incursion_layer].student_schools), 1))

    # Then identify students and teachers and randomly seed amongst them
    student_candidates = np.fromiter(people.contacts[incursion_layer].student_schools[school_id], dtype=cv.default_int)
    teacher_candidates = np.fromiter(people.contacts[incursion_layer].teacher_schools[school_id], dtype=cv.default_int)
    candidates = np.concatenate([student_candidates, teacher_candidates])
    seeded = np.random.choice(candidates, n_incursions, replace=False)
    sim.people.infect(seeded, layer="seed_infection")
    sim.pars["pop_infected"] = n_incursions

    return sim


def get_vaccine_indices_students(people, agelow, agehigh, coverage, coverage_index):
    """
    Get the indices for people that should be vaccinated based on their age
    """
    people_age_group = (np.where((people.age >= agelow) & (people.age <= agehigh)))[0]  # indices of people where age is in bounds
    number_tovax = np.int(np.round(np.int(coverage[coverage_index] * len(people_age_group))))  # number of people in this age category that can be vaxxed, depending on desired base coverage
    tovax_age_group = np.random.choice(people_age_group, number_tovax, replace=False)  # randomly sample the people to vaccinate depending on the number to vaccinate
    return tovax_age_group


def get_vaccine_indices_teachers(people, coverage, coverage_index):

    people_teachers_primary = list(people.contacts["primary_school"].teachers)
    people_teachers_high = list(people.contacts["high_school"].teachers)
    people_teachers = people_teachers_primary + people_teachers_high
    number_tovax = np.int(np.round(np.int(coverage[coverage_index] * len(people_teachers))))
    tovax_teachers = np.random.choice(people_teachers, number_tovax, replace=False)  # randomly sample the people to vaccinate depending on the number to vaccinate
    return tovax_teachers


def get_vaccine_indices_community(people, agelow, agehigh, coverage, coverage_index):

    people_age_group = (np.where((people.age >= agelow) & (people.age <= agehigh)))[0]  # indices of people where age is in bounds
    people_teachers_primary = list(people.contacts["primary_school"].teachers)
    people_teachers_high = list(people.contacts["high_school"].teachers)
    people_teachers = people_teachers_primary + people_teachers_high  # indices of teachers

    # get people in community who aren't teachers
    people_community = list(set(people_age_group) - set(people_teachers))

    number_tovax = np.int(np.round(np.int(coverage[coverage_index] * len(people_community))))
    tovax_community = np.random.choice(people_community, number_tovax, replace=False)  # randomly sample the people to vaccinate depending on the number to vaccinate
    return tovax_community


def baseline_vac_eligible(people, vac_base_coverage: str) -> np.array:
    """
    Account for baseline coverage

    Return the a boolean array for every person denoted whether they are vaccinated at baseline
    The vac_base_coverage is parsed into five components

        - Coverage among 5-11
        - Coverage among 12-15
        - Coverage among 16-17
        - Coverage among 18+
        - Teachers

    and specified as a string in that order - e.g. `vac_base_coverage=['0_0_0_80_90']`

    Args:
        people:
        vac_base_coverage: Underscore-separated string specifying vaccine coverage for (5-11; 12-15; 16-17; Community 18+; teachers)

    Returns:
    np
    """

    # string split into proportion
    coverage = [float(x) / 100 for x in vac_base_coverage.split("_")]

    # make object length of people
    p_vaccinated = np.full(len(people), fill_value=np.nan)

    # input the baseline coverages into p_vaccinated for each category

    tovax_5_11 = get_vaccine_indices_students(people, 5, 11, coverage, 0)
    p_vaccinated[tovax_5_11] = coverage[0]

    tovax_12_15 = get_vaccine_indices_students(people, 12, 15, coverage, 1)
    p_vaccinated[tovax_12_15] = coverage[1]

    tovax_16_17 = get_vaccine_indices_students(people, 16, 17, coverage, 2)
    p_vaccinated[tovax_16_17] = coverage[2]

    tovax_community = get_vaccine_indices_community(people, 18, 100, coverage, 3)
    p_vaccinated[tovax_community] = coverage[3]

    tovax_teachers = get_vaccine_indices_teachers(people, coverage, 4)
    p_vaccinated[tovax_teachers] = coverage[4]

    # return as a boolean array
    return ~np.isnan(p_vaccinated)


def vaccine_type_indices(vax_type, people, to_vax):
    """
       Get the indices of people to be vaccinated with Pfizer and AstraZeneca
       Uses the people 'to be vaxxed' as defined by proportions of school age/community being vaxxed
       People to be vaxxed are randomly given Pf or Az

    Inputs:
        vax_type: a .csv with the proportion of vaxxed people receiving AZ or PF by age group
        people: sim.people
        to_vax: boolean array of people to be vaccinated with length len(people)
    """

    inds_pf = []
    inds_az = []

    for vind in range(vax_type.shape[0]):

        inds_age = list(np.where((people.age >= vax_type.age_lo[vind]) & (people.age <= vax_type.age_hi[vind]) & (to_vax == True)))
        np.random.shuffle(inds_age)  # shuffle the indices of people to be vaccinated so it is a random draw

        # print(len(inds_age[0]), "people vaccinated for age ", vax_type.age_lo[vind], "to", vax_type.age_hi[vind])

        # get the number of people vaccinated with pfizer; the remainder will be az. Our list to subset from is total people vaccinated
        num_pf_age = round((len(inds_age) * vax_type.pf_prop_ofvaxxed[vind]))

        inds_pf_age = list(inds_age[:num_pf_age])
        inds_az_age = list(inds_age[(num_pf_age):])

        #  extend the list to add people vaccinated with vaccine of each age
        #  each element in list corresponds to list of indices of vaccinated with that vaccine for each age cat
        inds_pf += inds_pf_age
        inds_az += inds_az_age

    return inds_pf, inds_az



class SchoolTracing(cv.Intervention):
    def __init__(self, quarantine_policy: str, test_policy: str, daily_test_intervention: cvv.test_prob_screening, delayed_antigen_intervention, **kwargs):
        super().__init__(**kwargs)  # Initialize the Intervention object
        self.quarantine = quarantine_policy  # ['unvaccinated','all','none'] - contacts get quarantined
        self.test_policy = test_policy  # ['none','close','school']
        self.daily_test_intervention = daily_test_intervention # Test to stay intervention - this one should have 0 delay because students are testing in the morning before they go to school
        self.delayed_antigen_intervention = delayed_antigen_intervention # This one has 1 delay because when a diagnosis is recorded, screening the whole school would take at least 1 day
        self.index_case = None

        # We want to track days of school lost only due to the person actually being infected (i.e. symptomatic, tests positive, and is sent home)
        # OR because they were quarantined due to being an identified contact at school
        self.notified_via_school = None

        self.date_stop_daily_testing = None  # Track when people should stop doing daily tests

    def initialize(self, sim):
        super().initialize(sim)

        teachers = np.concatenate([sim.people.contacts["primary_school"].teachers, sim.people.contacts["high_school"].teachers])
        students = np.concatenate([sim.people.contacts["primary_school"].students, sim.people.contacts["high_school"].students])
        schools = np.concatenate([students, teachers])
        self.student_lookup = cykhash.Int32Set_from_buffer(students.astype(np.int32))
        self.teacher_lookup = cykhash.Int32Set_from_buffer(teachers.astype(np.int32))
        self.school_lookup = cykhash.Int32Set_from_buffer(schools.astype(np.int32))

        # Track whether the whole school has been screened or not, for one-off testing
        self.primary_screened = dict.fromkeys(range(len(sim.people.contacts["primary_school"].student_schools)), False)
        self.high_screened = dict.fromkeys(range(len(sim.people.contacts["high_school"].student_schools)), False)
        self.notified_via_school = np.full(len(sim.people), fill_value=False, dtype=bool)

        self.date_stop_daily_testing = np.full(len(sim.people), fill_value=np.nan, dtype=cv.default_float)

    def apply(self, sim):
        """
        Trace and notify contacts

        Tracing involves three steps that can independently be overloaded or extended
        by derived classes

        - Select which confirmed cases get interviewed by contact tracers
        - Identify the contacts of the confirmed case
        - Notify those contacts that they have been exposed and need to take some action
        """

        if self.quarantine != "none" or self.test_policy != "none":
            trace_inds = self.select_cases(sim)
            close_contacts = self.get_close_contacts(sim, trace_inds)  # close contacts of today's new school diagnoses
            class_contacts = self.get_class_contacts(sim, trace_inds)  # class contacts of today's new school diagnoses

        # Now we can apply interventions to the school contacts identified at this timestep
        if self.quarantine == "close":
            self.quarantine_contacts(sim, close_contacts)
        elif self.quarantine ==  "class":
            self.quarantine_contacts(sim, class_contacts)
        elif self.quarantine == "unvaccinated":
            self.quarantine_contacts(sim, class_contacts[~sim.people.vaccinated[class_contacts]])
        elif self.quarantine == "none":
            pass
        else:
            raise Exception("Unknown quarantine policy")

        # If a testing policy is active, add the contacts to the screening test intervention
        if self.test_policy != "none":
            if self.test_policy in {"class", "class+school_one_off"}:
                test_inds = class_contacts
            elif self.test_policy == "close":
                test_inds = close_contacts
            elif self.test_policy == "school":
                school_members = self.get_school_members(sim, trace_inds)
                test_inds = school_members
            else:
                raise Exception(f"Unknown test policy '{self.test_policy}'")

            self.date_stop_daily_testing[test_inds] = sim.t + 7

        # UPDATE TEST-TO-STAY INDS
        # Include in daily testing if the date_stop_daily_testing is greater than the current date
        # If they've never been a contact, the value is NaN so they won't get tested
        self.daily_test_intervention.inds = cv.true(self.date_stop_daily_testing > sim.t)

        if 'school_one_off' in self.test_policy:
            for school in sim.people.contacts['primary_school'].get_school_ids(trace_inds):
                if not self.primary_screened[school]:
                    self.primary_screened[school] = True
                    self.screen_school(sim, sim.people.contacts['primary_school'], school)

            for school in sim.people.contacts['high_school'].get_school_ids(trace_inds):
                if not self.high_screened[school]:
                    self.high_screened[school] = True
                    self.screen_school(sim, sim.people.contacts['high_school'], school)


    def screen_school(self, sim, layer, school_id):
        inds = np.fromiter(layer.student_schools[school_id].union((layer.teacher_schools[school_id])), cv.default_int)
        inds = cvu.ifalsei(sim.people.diagnosed | sim.people.quarantined | sim.people.dead, inds)  # Only test people that haven't been diagnosed already
        self.delayed_antigen_intervention._test(sim, inds)


    def select_cases(self, sim):
        """
        Identify people diagnosed in school at this timestep
        """
        inds = cvu.true(sim.people.date_diagnosed == sim.t)  # Diagnosed this time step, time to trace
        if len(inds):
            inds = inds[cvv.cykhash_isin(inds.astype(np.int32), self.school_lookup)]  # Only include people that belong to a school network
        return inds


    def get_close_contacts(self, sim: cv.Sim, trace_inds: np.ndarray) -> np.ndarray:
        """
        Identify all close contacts
        """
        return np.fromiter(sim.people.contacts["primary_school"].get_close_contacts(trace_inds).union(sim.people.contacts["high_school"].get_close_contacts(trace_inds)), dtype=cv.default_int)

    def get_school_members(self, sim, trace_inds):
        """
        Identify all school members
        """
        primary = sim.people.contacts["primary_school"].get_school_members(trace_inds)
        high = sim.people.contacts["high_school"].get_school_members(trace_inds)
        return np.fromiter(primary.union(high), dtype=np.int64)


    def get_class_contacts(self, sim, trace_inds):
        """
        Identify classroom contacts only

        In the base class, the trace time is the same per-layer, but derived classes might
        provide different functionality e.g. sampling the trace time from a distribution. The
        return value of this method is a dict keyed by trace time so that the `Person` object
        can be easily updated in `contact_tracing.notify_contacts`

        Args:
            sim: Simulation object
            trace_inds: Indices of people to trace

        Returns: {trace_time: np.array(inds)} dictionary storing which people to notify
        """

        class_contacts = sim.people.contacts["primary_school"].get_class_contacts(trace_inds).union(sim.people.contacts["high_school"].get_class_contacts(trace_inds))
        return np.fromiter(class_contacts, dtype=np.int64)


    def quarantine_contacts(self, sim, contacts):
        """
        Order contacts to quarantine tomorrow
        """
        is_dead = cvu.true(sim.people.dead)  # Find people who are not alive
        contact_inds = np.setdiff1d(contacts, is_dead)  # Do not notify contacts who are dead
        sim.people.known_contact[contact_inds] = True
        sim.people.date_known_contact[contact_inds] = np.fmin(sim.people.date_known_contact[contact_inds], sim.t + 1)  # If they are already a known contact, don't modify their known contact date
        sim.people.schedule_quarantine(contact_inds, start_date=sim.t + 1, period=7)  # Schedule quarantine for the notified people to start on the date they will be notified
        self.notified_via_school[contact_inds] = True

class SchoolAnalyzer(cv.Analyzer):
    # Record outcomes _in the school of the seeded person_
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.index_case = None #: Index of the first diagnosis
        self.is_teacher = None #: True if the index case was a teacher
        self.new_school_infections = None
        self.new_school_diagnoses = None
        self.new_community_infections = None
        self.new_community_diagnoses = None
        self.person_days_lost = None

        self._class_contacts = None
        self._close_contacts = None
        self._school_members = None
        self._school_students = None
        self._community_members = None

    def complete_initialization(self, sim):
        self.index_case = list(sim.people.infection_log.successors(None))[0]

        if self.index_case in sim.people.contacts["primary_school"]:
            layer = sim.people.contacts["primary_school"]
        elif self.index_case in sim.people.contacts["high_school"]:
            layer = sim.people.contacts["high_school"]
        else:
            raise Exception('Seed infection was not in a school?')

        self.is_teacher = self.index_case in layer.teachers

        self.new_school_infections = np.zeros_like(sim.tvec, dtype=cv.default_float)
        self.new_school_diagnoses = np.zeros_like(sim.tvec, dtype=cv.default_float)
        self.new_community_infections = np.zeros_like(sim.tvec, dtype=cv.default_float)
        self.new_community_diagnoses = np.zeros_like(sim.tvec, dtype=cv.default_float)
        self.person_days_lost = np.zeros_like(sim.tvec, dtype=cv.default_float)

        self._class_contacts = np.fromiter(layer.get_class_contacts(self.index_case), dtype=cv.default_int)
        self._close_contacts = np.fromiter(layer.get_close_contacts(self.index_case), dtype=cv.default_int)
        self._school_members = np.fromiter(layer.get_school_members(self.index_case), dtype=cv.default_int)
        self._school_students = np.intersect1d(self._school_members, layer.students)
        self._community_members = sim.people.contacts['H'].find_contacts(self._school_members)
        self._community_members = np.setdiff1d(self._community_members, self._school_members)

    def apply(self, sim):
        if self.index_case is None:
            self.complete_initialization(sim)

        self.new_school_infections[sim.t] = np.count_nonzero(sim.people.date_exposed[self._school_members] == sim.t)
        self.new_school_diagnoses[sim.t] = np.count_nonzero(sim.people.date_diagnosed[self._school_members] == sim.t)
        self.new_community_infections[sim.t] = np.count_nonzero(sim.people.date_exposed[self._community_members] == sim.t)
        self.new_community_diagnoses[sim.t] = np.count_nonzero(sim.people.date_diagnosed[self._community_members] == sim.t)

        iv = sim.get_intervention(SchoolTracing)
        self.person_days_lost[sim.t] = np.count_nonzero( (sim.people.quarantined[self._school_students] & iv.notified_via_school[self._school_students]) | (sim.people.diagnosed[self._school_students] & ~sim.people.recovered[self._school_students]))


    def finalize(self, sim):
        super().finalize(sim)

        add_result(sim, "new_school_infections", self.new_school_infections * sim.rescale_vec)
        add_result(sim, "new_school_diagnoses", self.new_school_diagnoses * sim.rescale_vec)
        add_result(sim, "cum_school_infections", np.cumsum(sim.results['new_school_infections']))
        add_result(sim, "cum_school_diagnoses", np.cumsum(sim.results['new_school_diagnoses']))

        add_result(sim, "new_community_infections", self.new_community_infections * sim.rescale_vec)
        add_result(sim, "new_community_diagnoses", self.new_community_diagnoses * sim.rescale_vec)
        add_result(sim, "cum_community_infections", np.cumsum(sim.results['new_community_infections']))
        add_result(sim, "cum_community_diagnoses", np.cumsum(sim.results['new_community_diagnoses']))

        add_result(sim, "person_days_lost", self.person_days_lost * sim.rescale_vec)
        add_result(sim, "cum_person_days_lost", np.cumsum(sim.results['person_days_lost']))
