import covasim as cv
import pandas as pd
import covasim_aus_schools as cvv
import covasim.misc as cvm
import sciris as sc
from functools import partial
import logging
import networkx as nx
import covasim.utils as cvu
import numpy as np

mixing_H = pd.read_csv(cvv.datadir / "mixing_H_extended.csv", index_col="Age group")
reference_ages = pd.read_csv(cvv.datadir / "reference_ages.csv", index_col="age", squeeze=True)
target_ages = pd.read_csv(cvv.datadir / "age.csv", index_col="age", squeeze=True)
households = pd.read_csv(cvv.datadir / "households.csv", index_col="size", squeeze=True)

# Parameters that are setting-specific (these differ between states, or regions within states)
employment_rate = 0.69  # source: greater sydney workforce participation rate August 2021 download 6291002 https://www.abs.gov.au/statistics/labour/employment-and-unemployment/labour-force-australia-detailed/latest-release#data-download
school_class_size = 24  # (differs between states and regional/metro) source:

retail_pr = 0.106  # retail as a proportion of all work download RQ1 Aug 2021 greater sydney https://www.abs.gov.au/statistics/labour/employment-and-unemployment/labour-force-australia-detailed/latest-release#labour-market-regions-sa4-
ent_pr = 0.022     # arts and recreation services from RQ1
hosp_pr = 0.034    # accommodation and food services from RQ1
non_retail_pr = 1 - (retail_pr + ent_pr + hosp_pr)

# Define common layer attributes
def get_layers():
    layers = {}
    layers["H"] = {"trace_prob": 1, "trace_time": 0, "quar_factor": 1, "iso_factor": 0.2}
    layers["child_care"] = {"trace_prob": 0.95, "trace_time": 1, "quar_factor": 0.01, "iso_factor": 0.01}
    layers["primary_school"] = {"trace_prob": 0.95, "trace_time": 1, "quar_factor": 0.01, "iso_factor": 0.01}
    layers["high_school"] = {"trace_prob": 0.95, "trace_time": 1, "quar_factor": 0.01, "iso_factor": 0.01}
    layers["non_retail_work"] = {"trace_prob": 0.95, "trace_time": 1, "quar_factor": 0.1, "iso_factor": 0.1}
    layers["retail_work"] = {"trace_prob": 0.95, "trace_time": 1, "quar_factor": 0.1, "iso_factor": 0.1}
    layers["entertainment"] = {"trace_prob": 0.5, "trace_time": 1, "quar_factor": 0, "iso_factor": 0}
    layers["cafe_restaurant"] = {"trace_prob": 0.5, "trace_time": 1, "quar_factor": 0, "iso_factor": 0}
    layers["pub_bar"] = {"trace_prob": 0.5, "trace_time": 1, "quar_factor": 0, "iso_factor": 0}
    layers["transport"] = {"trace_prob": 0.1, "trace_time": 1, "quar_factor": 0.01, "iso_factor": 0.01}
    layers["public_parks"] = {"trace_prob": 0.1, "trace_time": 1, "quar_factor": 0, "iso_factor": 0}
    layers["social"] = {"trace_prob": 0.75, "trace_time": 1, "quar_factor": 0.5, "iso_factor": 0.2}
    layers["C"] = {"trace_prob": 0.1, "trace_time": 1, "quar_factor": 0.2, "iso_factor": 0.2}
    layers["aged_care"] = {"trace_prob": 0.95, "trace_time": 1, "quar_factor": 0.2, "iso_factor": 0.2}
    layers["church"] = {"trace_prob": 0.5, "trace_time": 1, "quar_factor": 0.01, "iso_factor": 0.01}
    layers["cSport"] = {"trace_prob": 0.5, "trace_time": 1, "quar_factor": 0, "iso_factor": 0}
    return layers

def get_people(people_seed: int, pop_size: int, cross_classroom="base") -> cv.People:
    """
    Create cv.People instance

    Args:
        people_seed:
        pop_size:

    Returns:

    """

    if cross_classroom == "base":
        primary_random_student = 2
        high_random_student = 5
    elif cross_classroom == "double":
        primary_random_student = 4
        high_random_student = 10
    elif cross_classroom == "max":
        primary_random_student = 11
        high_random_student = 22
    else:
        raise Exception(f"Unknown cross classroom '{cross_classroom}'")

    cv.set_seed(people_seed)
    people = cvv.People.new(pop_size, mixing_H, reference_ages, households, target_ages=target_ages)

    workforce = people.get_age_eligible(age_lb=18, age_ub=65,
                                        proportion=employment_rate)  # proportion corresponds to employment rate

    # SCHOOLS
    students = people.get_age_eligible(age_lb=0, age_ub=4, proportion=0.545)
    layer = cvv.ChildCareLayer(
        baseline_beta=0.273972603,
        people=people,
        students=students,
        potential_teachers=workforce,
        mean_classroom_size=20,
        label="child_care",
    )
    workforce = workforce[~np.isin(workforce, layer.teachers)]  # Remove newly assigned staff from the workforce
    people.contacts[layer.label] = layer

    students = people.get_age_eligible(age_lb=5, age_ub=11, proportion=1)
    layer = cvv.PrimarySchoolLayer(
        baseline_beta=1,
        people=people,
        students=students,
        potential_teachers=workforce,
        mean_school_size=298,
        std_school_size=238,
        mean_class_size=22,
        min_class_size=10,
        mean_random_student_contacts=primary_random_student,
        mean_random_staff_contacts=5,
        beta_classroom=0.246575342,
        beta_student_student=0.028493151,
        beta_teacher_teacher=0.246575342,
        label="primary_school",
    )  # estimated from https://www.abs.gov.au/statistics/people/education/schools/latest-release  # estimated from https://www.abs.gov.au/statistics/people/education/schools/latest-release
    workforce = workforce[~np.isin(workforce, layer.teachers)]  # Remove newly assigned staff from the workforce
    people.contacts[layer.label] = layer

    students = people.get_age_eligible(age_lb=12, age_ub=17, proportion=1)
    layer = cvv.HighSchoolLayer(
        baseline_beta=1,
        people=people,
        students=students,
        mean_school_size=622,
        std_school_size=379,
        potential_teachers=workforce,
        student_teacher_ratio=12,
        mean_student_student_classroom_contacts=22 * 2,
        mean_student_teacher_classroom_contacts=6,
        mean_student_student_random_contacts=high_random_student,
        mean_teacher_teacher_random_contacts=5,
        beta_classroom=0.246575342 / 2,
        beta_student_student=0.124109589,
        beta_teacher_teacher=0.246575342,
        label="high_school",
    )
    workforce = workforce[~np.isin(workforce, layer.teachers)]  # Remove newly assigned staff from the workforce
    people.contacts[layer.label] = layer

    # Divide remaining people into the different workplace layers
    proportion_of_workforce = pd.Series(
        {
            "non_retail_work": non_retail_pr,
            "retail_work": retail_pr,
            "entertainment": ent_pr,
            "cafe_restaurant": hosp_pr / 2,
            "pub_bar": hosp_pr / 2,
        }
    )
    number_of_people = (proportion_of_workforce * len(workforce)).round().astype(int)
    number_of_people.iloc[-1] = len(workforce) - number_of_people.iloc[:-1].sum()
    worker_inds = {}
    for i in range(len(proportion_of_workforce)):
        k = number_of_people.index[i]
        offset = number_of_people[:i].sum()
        worker_inds[k] = workforce[offset: (offset + number_of_people[i])]  # Partition the workforce

    # WORK LAYERS
    layer = cvv.ClusterLayer(label="non_retail_work", inds=worker_inds["non_retail_work"], mean_cluster_size=5,
                             baseline_beta=0.28)
    people.contacts[layer.label] = layer

    public = people.get_age_eligible(age_lb=12, proportion=0.7)
    layer = cvv.PublicFacingLayer(
        baseline_beta=1,
        label="retail_work",
        staff_inds=worker_inds["retail_work"],
        mean_staff_cluster_size=5,
        staff_beta=0.28,
        public_inds=public,
        public_beta=0.042739726,
        public_staff_beta=0.042739726,
        mean_public_contacts=8,
        mean_public_staff_contacts=2,
    )
    people.contacts[layer.label] = layer

    public = people.get_age_eligible(age_lb=15, proportion=0.3)
    layer = cvv.PublicFacingLayer(
        baseline_beta=1,
        label="entertainment",
        staff_inds=worker_inds["entertainment"],
        mean_staff_cluster_size=5,
        staff_beta=0.28,
        public_inds=public,
        public_beta=0.008219178,
        public_staff_beta=0.008219178,
        mean_public_contacts=25,
        mean_public_staff_contacts=2,
    )
    people.contacts[layer.label] = layer

    public = people.get_age_eligible(age_lb=12, proportion=0.6)
    layer = cvv.PublicFacingLayer(
        baseline_beta=1,
        label="cafe_restaurant",
        staff_inds=worker_inds["cafe_restaurant"],
        mean_staff_cluster_size=5,
        staff_beta=0.28,
        public_inds=public,
        public_beta=0.042739726,
        public_staff_beta=0.042739726,
        mean_public_contacts=8,
        mean_public_staff_contacts=2,
    )
    people.contacts[layer.label] = layer

    public = people.get_age_eligible(age_lb=18, proportion=0.4)
    layer = cvv.PublicFacingLayer(
        baseline_beta=1,
        label="pub_bar",
        staff_inds=worker_inds["pub_bar"],
        mean_staff_cluster_size=5,
        staff_beta=0.28,
        public_inds=public,
        public_beta=0.056986301,
        public_staff_beta=0.056986301,
        mean_public_contacts=8,
        mean_public_staff_contacts=2,
    )
    people.contacts[layer.label] = layer

    ## OTHER LAYERS

    inds = people.get_age_eligible(proportion=1)
    layer = cvv.RandomLayer(label="C", mean_contacts=1, inds=inds, dynamic=True, baseline_beta=0.1)
    people.contacts[layer.label] = layer

    inds = people.get_age_eligible(proportion=0.11)
    layer = cvv.ClusterLayer(label="church", mean_cluster_size=20, inds=inds, baseline_beta=0.042739726)
    people.contacts[layer.label] = layer

    inds = people.get_age_eligible(age_lb=4, age_ub=30, proportion=0.34)
    layer = cvv.ClusterLayer(label="cSport", mean_cluster_size=30, inds=inds, baseline_beta=0.071232877)
    people.contacts[layer.label] = layer

    age_distribution = pd.read_csv(cvv.datadir / "abs_transport.csv", index_col="Age group", squeeze=False)
    inds = people.get_age_eligible(age_distribution=age_distribution, proportion=0.114)
    layer = cvv.RandomLayer(label="transport", inds=inds, mean_contacts=25, dynamic=True, baseline_beta=0.164383562)
    people.contacts[layer.label] = layer

    inds = people.get_age_eligible(proportion=0.6)
    layer = cvv.RandomLayer(label="public_parks", mean_contacts=10, inds=inds, dynamic=True, baseline_beta=0.028493151)
    people.contacts[layer.label] = layer

    inds = people.get_age_eligible(proportion=1, age_lb=15)
    layer = cvv.RandomLayer(label="social", mean_contacts=6, dispersion=2, inds=inds, baseline_beta=0.124109589)
    people.contacts[layer.label] = layer

    inds = people.get_age_eligible(proportion=0.07, age_lb=65)
    layer = cvv.ClusterLayer(label="aged_care", mean_cluster_size=12, inds=inds, baseline_beta=0.578)
    people.contacts[layer.label] = layer

    ## FINALIZE
    layers = get_layers()
    assert set(people.contacts.keys()) == set(layers.keys()), "Layer listing does not match the layer objects that were instantiated"
    return people


if __name__ == "__main__":
    people = get_people(0, int(1e5))
