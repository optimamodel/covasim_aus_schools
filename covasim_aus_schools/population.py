# FUNCTIONS TO CREATE POPULATIONS

from collections import defaultdict

import covasim as cv
import covasim.defaults as cvd
import covasim.utils as cvu
import covasim_aus_schools as cvv
import numba as nb
import numpy as np
import pandas as pd
import networkx as nx
from covasim_aus_schools import logger
import sciris as sc


## MAIN API


class InfectionGraph(nx.DiGraph):
    # Graph structure compatible with cv.People expecting a list implementation for the infection log
    def append(self, infection_log_entry: dict):
        self.add_edge(infection_log_entry["source"], infection_log_entry["target"], date=infection_log_entry["date"], layer=infection_log_entry["layer"])

    @property
    def transmissions(self):
        out = []
        for a, b in self.edges:
            d = self.get_edge_data(a,b)
            out.append({'from':a,'to':b,'date':d['date'], 'layer': d['layer']})
        return sorted(out, key=lambda x: x['date'])


class People(cv.People):
    """
    Victoria people class

    This class overloads update_contacts to regenerate
    the specific network types e.g. clustered defined in this
    module. By overloading update_contacts, a separate
    intervention to update the networks is no longer required.

    Any dynamic layers contained within People should

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infection_log = InfectionGraph()  # Replace infection log with graph for SecondRingTracing
        self.fully_vaccinated = np.full(len(self), False, dtype=bool)

    def initialize(self):
        assert not self.initialized, "Double initialization"
        super().initialize()

        # Make sure we only have VictoriaLayer instances, so that all layers are
        # guaranteed to use custom behaviour defined in this project
        for layer in self.contacts.values():
            assert isinstance(layer, VictoriaLayer)

    def make_naive(self, inds):
        """
        Override make_naive to preserve vaccinated status. This helps to ensure
        vaccination status is not lost by rescaling, until Covasim is updated
        """
        vaccinated = self.vaccinated[inds].copy()
        fully_vaccinated = self.fully_vaccinated[inds].copy()
        date_vaccinated = self.date_vaccinated[inds].copy()
        super().make_naive(inds)
        self.vaccinated[inds] = vaccinated
        self.fully_vaccinated[inds] = fully_vaccinated
        self.date_vaccinated[inds] = date_vaccinated

    def set_prognoses(self):

        # Don't allow double initialization
        if getattr(self, "initialized", False):
            raise Exception("Cannot re-initialize prognoses yet")

        output = super().set_prognoses()

        self.baseline_symp_prob = self.symp_prob.copy()
        self.baseline_severe_prob = self.severe_prob.copy()
        self.baseline_crit_prob = self.crit_prob.copy()
        self.baseline_death_prob = self.death_prob.copy()
        self.baseline_rel_sus = self.rel_sus.copy()
        self.baseline_rel_trans = self.rel_trans.copy()

        return output

    def update_contacts(self):
        # Unconditionally update layers (no need for dynam_layer)
        for name, layer in self.contacts.items():
            layer.update()  # The update methods change `Layer.beta` therefore we need to do any layer updates *before* resetting beta
            layer.reset_beta()  # Every timestep, reset beta in case a vaccine mandate changed it (and the layer was not otherwise updated)
        return self.contacts

    def check_diagnosed(self):
        # Record diagnoses on the day the result is recieved
        # By default, diagnoses are recorded on the day tested in Covasim
        super().check_diagnosed()
        return np.sum(self.date_diagnosed == self.t)

    @classmethod
    def new(cls, n_people: int, mixing: pd.DataFrame, reference_ages: pd.Series, households: pd.Series, target_ages: pd.Series = None) -> cv.People:
        """
        Construct a cv.People object from population data files

        This will construct a cv.People based on demographics, and a household layer based on the mixing matrix.
        These are done together because it's simpler to populate households based on the mixing matrix until a
        until a sufficient number of agents have been created, rather than creating the agents first and then
        trying to assign them to households to match the mixing matrix.

        Args:
            n_people: Number of simulation agents to create
            mixing: The mixing matrix dataframe (obtained by loading the mixing matrix CSV file)
            reference: Distribution of reference ages. This should be a pd.Dataframe/pd.Series where the index is the age, and the value is the (relative or absolute) number of people of that age
            households: Distribution of households of different sizes. Should be a pd.Dataframe/pd.Series where the index is the number of people in the household, and the value is the (relative or absolute) number of households of that size.
            target_ages: Optionally specify distribution of target ages - used to rescale mixing matrix
        Returns: - A `cv.People` instance with a household contact layer

        """

        if target_ages is not None:
            mixing = rescale_mixing_matrix(mixing, reference_ages, households, target_ages)

        # First work out how many households of each size we need
        total_people = sum(households.index * households.values)  # total_people = household_size * n_households
        household_percent = households / total_people
        n_households = (n_people * household_percent).round().astype(int)
        n_households[1] += n_people - sum(n_households * n_households.index)  # adjust single-person households to fill the gap

        # Then, select a reference person age for the first person in each household
        household_heads = np.random.choice(reference_ages.index, size=sum(n_households), p=reference_ages.values / sum(reference_ages))

        h_clusters, ages = _make_households(n_households, n_people, household_heads, mixing)

        contacts = cv.Contacts()
        contacts["H"] = StaticClusterLayer(h_clusters, label="Household", baseline_beta=1)
        people = cls(pars={"pop_size": n_people}, age=ages)
        people.contacts = contacts

        return people

    def get_age_eligible(self, *, proportion, age_lb=None, age_ub=None, age_distribution=None) -> np.array:
        """
        Return person indices for people satisfying filter

        Ages can be specified EITHER as an inclusive lower/upper bound, or with ``pd.Series`` specifying
        a distribution of ages. The dataframe looks like

            0-4: 100
            5-9: 150

        If a distribution is specified, the lower and upper bounds should not be specified.

        Args:
            age_lb: Lower bound on age (inclusive)
            age_ub: Upper bound on age (inclusive); or
            age_distribution: pd.Series specifying age distribution of layer participants
            proportion: Proportion of eligible people

        Returns: An array of person indices for agents satisfying the age filter

        """

        assert proportion >= 0 and proportion <= 1

        if age_distribution is not None:
            assert age_lb is None
            assert age_ub is None

            age_dist = age_distribution.copy()  # Don't modify the original
            age_dist.index = age_dist.index.map(cvv.parse_age_range).set_names(["age_low", "age_high"])
            age_dist = age_dist / age_dist.sum()  # normalize
            age_dist.columns = ["proportion"]
            age_dist["denominator"] = 0

            p_include = np.zeros_like(self.age, dtype=cv.default_float)
            for low, high in age_dist.index:
                match_age = (self.age >= low) & (self.age <= high)
                age_dist.loc[(low, high), "denominator"] = np.count_nonzero(match_age)
                p_include[match_age] = age_dist.at[(low, high), "proportion"]

            n_required = proportion * len(self)  # Target number of self
            p_scale = n_required / (age_dist["proportion"] * age_dist["denominator"]).sum()
            p_include *= p_scale
            inds = cv.binomial_filter(p_include, self.indices())

        else:
            age_min = 0 if age_lb is None else age_lb
            age_max = np.inf if age_ub is None else age_ub

            age_eligible = cvu.true((self.age >= age_min) & (self.age <= age_max))
            n_people = int(proportion * len(age_eligible))
            inds = np.random.choice(age_eligible, n_people, replace=False).astype(cv.default_int)

        return inds


def rescale_mixing_matrix(mixing_matrix, reference_ages, households, target_ages):
    # CALCULATE SCALED MATRIX

    # Calculate based on the household size distribution, what proportion of the total
    # number of people in the people will be household heads, and therefore drawn from the
    # reference distribution
    proportion_via_reference = sum(households.values) / sum(households.index.values * households.values)

    # What is the distribution of ages based on the mixing matrix?
    age_lb = [int(x.split(" to ")[0]) for x in mixing_matrix.index]

    # Bin the reference ages. Every reference person selected gets passed through the household
    # routine, therefore we include all ages at this point
    reference_ages_binned = reference_ages.copy()
    reference_ages_binned.index = np.digitize(reference_ages.index, age_lb) - 1  # First, find the index of the bin that the reference person belongs to
    reference_ages_binned = reference_ages_binned.groupby(level=0).sum()
    reference_ages_binned = reference_ages_binned / reference_ages_binned.sum()  # The fraction of reference ages in each age bin of the mixing matrix

    matrix_component = mixing_matrix.multiply(reference_ages_binned.values, axis=0)  # The axis reflects essentially no 0-15 year olds being reference people
    matrix_dist = matrix_component.sum(axis=0)  # The axis reflects some 0-15 year olds being generated via other reference people
    matrix_dist = matrix_dist / matrix_dist.sum()
    matrix_dist.index = reference_ages_binned.index

    # The final distribution is the average of the reference_distribution*reference_proportion and the matrix_dist*matrix proportion
    # We need to choose weighting factors for the matrix that minimize the discrepancy with the target distribution
    # Thus oversampling ages at each row. This is still approximate but should be better than nothing
    # So easiest would be to interpolate onto the matrix ages, which is where the changes need to be implemented

    # When sampling, we only generate people based on the age ranges in the mixing matrix e.g. a 100 year old
    # reference person is assigned to the last bin e.g. 75-80, and then the oldest person that can be selected
    # to add to the household would be 80. By scaling the mixing matrix based on this, we will end up with too
    # many people being added to the oldest age bin, and not enough people above it i.e. 95 year olds in the
    # target distribution get turned into 75-80 year olds. This is still a better approximation of the population
    # pyramid given that the older ages have worse prognoses.
    target_ages_binned = target_ages.copy()
    target_ages_binned.index = np.digitize(target_ages_binned.index, age_lb) - 1  # First, find the index of the bin that the reference person belongs to
    target_ages_binned = target_ages_binned.groupby(level=0).sum()
    target_ages_binned = target_ages_binned / target_ages_binned.sum()  # The fraction of reference ages in each age bin of the mixing matrix

    # We want these two distributions to be the same
    # plt.plot(reference_ages_binned*proportion_via_reference + matrix_dist*(1-proportion_via_reference))
    # plt.plot(target_ages_binned)

    # So compute a scaling factor based on them being equal
    factor = (target_ages_binned - reference_ages_binned * proportion_via_reference) / (matrix_dist * (1 - proportion_via_reference))

    # Apply the scaling factor to each row of the mixing matrix
    scaled_mixing_matrix = mixing_matrix.multiply(factor.values, axis=1)  # axis=1 so that each row gets multiplied by the factor

    return scaled_mixing_matrix


def add_special_contacts(people: cv.People, layers: pd.DataFrame):
    # For NSW, add schools and all types of workplaces together so that staff
    # can be allocated exclusively across all of these layers

    # Work out who is in the workforce
    age_eligible = cvu.true((people.age >= layers.at["workforce", "age_lb"]) & (people.age <= layers.at["workforce", "age_ub"]))
    n_people = int(layers.at["workforce", "proportion"] * len(age_eligible))
    workforce = np.random.choice(age_eligible, n_people, replace=False)  # Note that this also shuffles the indices

    for layer_name, layer in layers.to_dict(orient="index").items():
        if layer["cluster_type"] in {"school", "prim_school", "high_school", "child_care"}:
            students = people.get_age_eligible(age_lb=layers.at[layer_name, "age_lb"], age_ub=layers.at[layer_name, "age_ub"], proportion=layers.at[layer_name, "proportion"])
            if layer["cluster_type"] == "school":
                people.contacts[layer_name] = SchoolLayer(people=people, students=students, mean_classroom_size=layers.at[layer_name, "contacts"], potential_teachers=workforce, label=layer_name)
            elif layer["cluster_type"] == "primary_school":
                people.contacts[layer_name] = PrimarySchoolLayer(people=people, students=students, mean_classroom_size=layers.at[layer_name, "contacts"], potential_teachers=workforce, label=layer_name)
            elif layer["cluster_type"] == "high_school":
                people.contacts[layer_name] = HighSchoolLayer(people=people, students=students, mean_classroom_size=layers.at[layer_name, "contacts"], potential_teachers=workforce, label=layer_name)
            elif layer["cluster_type"] == "child_care":
                people.contacts[layer_name] = ChildCareLayer(people=people, students=students, mean_classroom_size=layers.at[layer_name, "contacts"], potential_teachers=workforce, label=layer_name)
            else:
                raise Exception("Unknown layer type")
            workforce = workforce[~np.isin(workforce, people.contacts[layer_name].teachers)]  # Remove newly assigned staff from the workforce

    # Downscale by the proportion outside the LGA
    n_outside_lga = int(round(len(workforce) * layers.at["outside_lga", "proportion"]))
    people.outside_lga, workforce = workforce[:n_outside_lga], workforce[n_outside_lga:]

    # Downscale outside LGA by authorized
    n_authorized = int(round(len(people.outside_lga) * layers.at["authorized_outside_lga", "proportion"]))
    people.authorized_outside_lga = people.outside_lga[:n_authorized]

    # Divide remaining people into the different workplace layers
    proportions = layers["proportion_of_workforce"].dropna()
    number_of_people = (proportions * len(workforce)).round().astype(int)
    number_of_people.iloc[-1] = len(workforce) - number_of_people.iloc[:-1].sum()

    for layer_name, layer in layers.loc[number_of_people.index].to_dict(orient="index").items():
        worker_inds, workforce = workforce[: number_of_people[layer_name]], workforce[number_of_people[layer_name]:]  # Partition the workforce

        if layer["cluster_type"] == "public":
            # Now we need to select indices of people from the general public
            # A restaurant worker could dine in another restaurant, therefore we do NOT remove the worker_inds from the public_inds

            age_min = 0 if pd.isna(layer["age_lb"]) else layer["age_lb"]
            age_max = np.inf if pd.isna(layer["age_ub"]) else layer["age_ub"]
            age_eligible = cvu.true((people.age >= age_min) & (people.age <= age_max))
            n_people = int(layer["proportion"] * len(age_eligible))
            public_inds = np.random.choice(age_eligible, n_people, replace=False)

            people.contacts[layer_name] = PublicFacingLayer(
                staff_inds=worker_inds,
                public_inds=public_inds,
                mean_staff_cluster_size=layer["contacts"],
                mean_public_contacts=layer["public_contacts"],
                mean_public_staff_contacts=layer["staff_public_contacts"],
                staff_beta=layer["staff_beta"],
                public_beta=layer["public_beta"],
                label=layer_name,
            )

        elif layer["cluster_type"] == "work":
            # Create a clustered layer based on the mean cluster size
            people.contacts[layer_name] = ClusterLayer(worker_inds, mean_cluster_size=layer["contacts"], dynamic=(not pd.isna(layer["dynamic"])), label=layer_name)
        else:
            raise Exception('Unexpected layer type for a layer that has "proportion_of_workforce" defined?')


def add_other_contacts(people: cv.People, layers: pd.DataFrame, exclude=None, age_distributions=None):
    """
    Add layers according to a layer file

    Args:
        people: A cv.People instance to add new layers to
        layer_members: Dict containing {layer_name:[indexes]} specifying who is able to have interactions within each layer
        layerfile: Dataframe from `layers.csv` where the index is the layer name
        exclude: Indices of people to exclude, by layer e.g. {'W':teacher_inds}
        age_distribution: A grouped age distribution specification e.g. {'transport':{'0-4':300,'5-9':200}} which takes precedence over the age_ub and age_lb
    """

    if exclude is None:
        exclude = {}

    if age_distributions is None:
        age_distributions = {}

    for layer_name, layer in layers.iterrows():

        if pd.isna(layer["cluster_type"]) or layer["cluster_type"] in {"home", "school", "primary_school", "high_school", "work", "public", "child_care"}:
            # Ignore these cluster types, as they should be instantiated with
            # - home: make_people()
            # - school: add_school_contacts()
            # - work/public: add_work_contacts() or add_nsw_work_contacts() (otherwise unsupported)
            continue

        if layer_name in age_distributions:
            age_dist = age_distributions[layer_name].copy()

            age_dist.index = age_dist.index.map(cvv.parse_age_range).set_names(["age_low", "age_high"])
            age_dist = age_dist / age_dist.sum()  # normalize
            age_dist.columns = ["proportion"]
            age_dist["denominator"] = 0
            p_include = np.zeros_like(people.age, dtype=cv.default_float)
            for low, high in age_dist.index:
                match_age = (people.age >= low) & (people.age <= high)
                age_dist.loc[(low, high), "denominator"] = np.count_nonzero(match_age)
                p_include[match_age] = age_dist.at[(low, high), "proportion"]

            n_required = layer["proportion"] * len(people)  # Target number of people
            p_scale = n_required / (age_dist["proportion"] * age_dist["denominator"]).sum()
            p_include *= p_scale
            inds = cv.binomial_filter(p_include, people.indices())

        else:
            age_min = 0 if pd.isna(layer["age_lb"]) else layer["age_lb"]
            age_max = np.inf if pd.isna(layer["age_ub"]) else layer["age_ub"]
            age_eligible = cvu.true((people.age >= age_min) & (people.age <= age_max))
            n_people = int(layer["proportion"] * len(age_eligible))
            inds = np.random.choice(age_eligible, n_people, replace=False)

        if layer_name in exclude:
            inds = inds[~np.isin(inds, exclude[layer_name])]

        if layer["cluster_type"] == "cluster":
            # Create a clustered layer based on the mean cluster size
            people.contacts[layer_name] = ClusterLayer(inds, layer["contacts"], dynamic=(not pd.isna(layer["dynamic"])), label=layer_name)
        elif layer["cluster_type"] == "random":
            people.contacts[layer_name] = RandomLayer(inds, layer["contacts"], layer["dispersion"], dynamic=(not pd.isna(layer["dynamic"])), label=layer_name)
        else:
            raise Exception(f'Unknown clustering type {layer["cluster_type"]}')


## EXTERNALLY-FACING HELPERS


class VictoriaLayer(cv.Layer):

    def __init__(self, *args, baseline_beta, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline_beta = baseline_beta  # This gets used to populate pars['beta_layer'] during simulation construction

    def update(self):
        # Disable default layer update method so that custom structures cannot be accidentally lost
        pass

    def reset_beta(self):
        # This gets called every timestep to reset the beta dictionary
        self["beta"] = self.beta.copy()

    def validate(self):
        # Disable validation since the implementation of self.beta means
        # that Covasim's validation will fail
        return

    def __repr__(self):
        return f'<{self.__class__.__name__} "{self.label}" ({len(self.members)} members, {len(self)} contacts)>'


class StaticClusterLayer(VictoriaLayer):
    """
    Cluster layer with fixed, precomputed clusters

    This layer type is intended for use with clusters that are pre-computed externally -
    mainly households
    """

    def __init__(self, clusters, *args, **kwargs):
        """

        Args:
            clusters: List of lists, ``[[person_ids],...]`` defining each cluster
            *args: Passed to ``VictoriaLayer``
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        self.clusters = clusters  # Store clusters for later use
        self["p1"], self["p2"] = ClusterLayer.cluster_arrays(clusters)
        self.beta = np.ones(len(self["p1"]), dtype=cvd.default_float)
        self.validate()


class ClusterLayer(VictoriaLayer):
    def __init__(self, inds, mean_cluster_size, dynamic=False, *args, **kwargs):
        """
        Convert a list of clusters to a Covasim layer

        Assumes fully connected clusters. This class also implements additional hooks
        for manipulating the contacts in the layer

        - `prop_open` - selectively close entire clusters
        - `attendance` - clip edges by removing people from the clusters (temporarily). This allows vaccines to be targeted
                         at only the remaining people (by checking the `members` attribute after setting attendance). Otherwise,
                         it should produce similar effects to simply decreasing beta.

        The attendance captures some people not participating in their assigned clusters.
        For example, people working from home. Whereas the proportion open captures entire sites
        being closed.

        """

        super().__init__(*args, **kwargs)
        logger.debug(f'Creating ClusterLayer "{self.label}" with {len(inds)} members')

        self.inds = inds
        self.mean_cluster_size = mean_cluster_size
        self.dynamic = dynamic  #: Update the contacts every timestep

        self._clusters = None  #: A list of lists with the clusters
        self._prop_open = 1.0  #: Proportion of clusters that are active. At instantiation, start with everything open
        self._attendance = 1.0  #: Proportion of people that participate in clusters. At instantiation, start with complete clusters
        self._attendance_priority = np.random.permutation(inds)  #: Array of indices storing the order in which people should be excluded, if the attendance is less than 1. People at the start of the array get excluded first
        self._is_open = None  #: Boolean array storing whether or not a cluster is active and should be included in contact arrays

        self.update(force=True)

    @property
    def prop_open(self):
        return self._prop_open

    @prop_open.setter
    def prop_open(self, proportion):
        if proportion != self._prop_open:
            self._prop_open = proportion
            self._update_contacts()

    @property
    def attendance(self):
        return self._attendance

    @attendance.setter
    def attendance(self, proportion):
        if proportion != self._attendance:
            self._attendance = proportion
            self._update_contacts()

    def _update_contacts(self):
        """
        Regenerate contact arrays

        This method regenerates the p1 and p2 arrays when the prop_open or attendance changes
        It does NOT change the clusters themselves

        Returns:

        """

        self._is_open[:] = True
        if self.prop_open < 1:
            n_to_open = int(round(self._prop_open * len(self._clusters)))
            self._is_open[n_to_open:] = False
        else:
            n_to_open = len(self._clusters)

        self["p1"], self["p2"] = self.cluster_arrays(cluster for cluster, is_open in zip(self._clusters, self._is_open) if is_open)

        # Now remove contacts for any people that are not participating
        if self._attendance < 1:
            # nb. this is where the people appearing in the array first get excluded first is implemented, because
            # `inds_to_exclude` are drawn starting from the beginning of the attendance_priority array
            inds_to_exclude = self._attendance_priority[: int((1 - self._attendance) * len(self._attendance_priority))]
            contacts_to_exclude = np.isin(self["p1"], inds_to_exclude) | np.isin(self["p2"], inds_to_exclude)
            self["p1"] = self["p1"][~contacts_to_exclude]
            self["p2"] = self["p2"][~contacts_to_exclude]

        self.beta = np.ones(len(self["p1"]), dtype=cvd.default_float)

        logger.debug(f"{self.label}: Opening {n_to_open} of {len(self._clusters)} venues with {100 * self._attendance:.0f}% attendance")

        self.validate()

    def update(self, force: bool = False) -> None:
        """
        Regenerate contacts

        This only needs to be called if the clusters are changed. Set `prop_open` directly
        to immediately change which contacts are active

        Args:
            force: If True, ignore the `self.dynamic` flag. This is required for initialization.

        """

        if not self.dynamic and not force:
            return

        self._clusters = self.create_poisson_clusters(self.inds, self.mean_cluster_size, 0)
        self._is_open = np.ones(len(self._clusters)).astype(bool)  #: Regenerate cluster opening flag
        self._update_contacts()

    @staticmethod
    @nb.njit((cvd.nbint[:], cvd.nbfloat, cvd.nbint))
    def create_poisson_clusters(people_to_cluster: list, mean_cluster_size: float, min_cluster_size: int) -> list:
        """
        Assign people to clusters with Poisson size distribution

        Returns a list of clusters

        Args:
            people_to_cluster: Indexes of people to cluster e.g. [1,5,10,12,13]
            mean_cluster_size: Mean cluster size (poisson distribution)
            min_cluster_size: Any clusters smaller than this size will be skipped or redistributed. Set to 0 for no minimum size

        Returns: List of lists of clusters e.g. [[1,5],[10,12,13]]
        """

        people_to_cluster = np.random.permutation(people_to_cluster)
        clusters = []
        n_people = len(people_to_cluster)
        n_remaining = n_people

        while n_remaining > min_cluster_size:
            this_cluster = cvu.poisson(mean_cluster_size)  # Sample the cluster size

            if this_cluster > n_remaining:
                this_cluster = n_remaining
            elif this_cluster < min_cluster_size:
                continue

            if this_cluster > 0:
                clusters.append(list(people_to_cluster[(n_people - n_remaining) + np.arange(this_cluster)]))
                n_remaining -= this_cluster

        if n_remaining > 0:
            # If there are people that still need to be allocated
            # Work out how many need to be added to the existing clusters
            if len(clusters) == 0:
                # If there are fewer people than the minimum cluster size, then no clusters will have been created. Assign them all to one cluster
                clusters = [list(people_to_cluster)]
            else:
                to_assign = cv.choose_r(len(clusters), n_remaining)
                for cluster, person in zip(to_assign, people_to_cluster[-n_remaining:]):
                    clusters[cluster].append(person)

        return clusters

    @staticmethod
    @nb.njit((cvd.nbint[:], cvd.nbfloat, cvd.nbfloat, cvd.nbint))
    def create_normal_clusters(people_to_cluster: list, mean_cluster_size: float, std_cluster_size: float, min_cluster_size: int) -> list:
        """
        Assign people to clusters with normal size distribution

        Returns a list of clusters. It is a list type so that appending further items to the cluster
        (e.g., appending teachers to a class of students) is efficient

        Args:
            people_to_cluster: Indexes of people to cluster e.g. [1,5,10,12,13]
            mean_cluster_size: Mean cluster size (poisson distribution)
            min_cluster_size: Any clusters smaller than this size will be skipped or redistributed. Set to 0 for no minimum size

        Returns: List of lists of clusters e.g. [[1,5],[10,12,13]]
        """

        people_to_cluster = np.random.permutation(people_to_cluster)
        clusters = []
        n_people = len(people_to_cluster)
        n_remaining = n_people

        while n_remaining > min_cluster_size:

            this_cluster = int(np.round(np.random.normal(loc=mean_cluster_size, scale=std_cluster_size)))

            if this_cluster > n_remaining:
                this_cluster = n_remaining
            elif this_cluster < min_cluster_size:
                continue

            if this_cluster > 0:
                clusters.append(list(people_to_cluster[(n_people - n_remaining) + np.arange(this_cluster)]))
                n_remaining -= this_cluster

        if n_remaining > 0:
            # If there are people that still need to be allocated
            # Work out how many need to be added to the existing clusters
            to_assign = cv.choose_r(len(clusters), n_remaining)
            for cluster, person in zip(to_assign, people_to_cluster[-n_remaining:]):
                clusters[cluster].append(person)

        return clusters

    @staticmethod
    def cluster_arrays(clusters):
        # Convert a list of lists of clusters into an edge representation
        # where each cluster is fully connected

        p1 = []
        p2 = []

        for cluster in clusters:
            for i, a in enumerate(cluster):
                for j, b in enumerate(cluster):
                    if j < i:
                        p1.append(a)
                        p2.append(b)

        return np.array(p1, dtype=cvd.default_int), np.array(p2, dtype=cvd.default_int)


class RandomLayer(VictoriaLayer):
    """
    Dynamic random contacts

    No network structure, just randomly sampled contacts based on the mean
    number of contacts per person. Can be overdispersed. If the dynamic
    flag is set (e.g. during construction) then the contacts will be resampled
    at each timestep using the parameters passed in at initialization (i.e. which
    people have random contacts in this layer, and the mean contacts+dispersion)
    """

    def __init__(self, inds, mean_contacts, dispersion=None, dynamic=False, *args, **kwargs):
        """

        Args:
            inds:
            mean_contacts:
            dispersion: Level
            dynamic: If True, the layer will change each timestep
        """
        super().__init__(*args, **kwargs)
        self.inds = inds
        self.mean_contacts = mean_contacts
        self.dispersion = dispersion
        self.dynamic = dynamic
        self.update(force=True)

    @staticmethod
    @nb.njit
    def get_contacts(inds, number_of_contacts):
        """
        Efficiently generate contacts

        Note that because of the shuffling operation, each person is assigned 2N contacts
        (i.e. if a person has 5 contacts, they appear 5 times in the 'source' array and 5
        times in the 'target' array). Therefore, the `number_of_contacts` argument to this
        function should be HALF of the total contacts a person is expected to have, if both
        the source and target array outputs are used (e.g. for social contacts)

        adjusted_number_of_contacts = np.round(number_of_contacts / 2).astype(cvd.default_int)

        Whereas for asymmetric contacts (e.g. staff-public interactions) it might not be necessary

        Args:
            inds: List/array of person indices
            number_of_contacts: List/array the same length as `inds` with the number of unidirectional
            contacts to assign to each person. Therefore, a person will have on average TWICE this number
            of random contacts.

        Returns: Two arrays, for source and target


        """

        total_number_of_half_edges = np.sum(number_of_contacts)

        count = 0
        source = np.zeros((total_number_of_half_edges,), dtype=cvd.default_int)
        for i, person_id in enumerate(inds):
            n_contacts = number_of_contacts[i]
            source[count: count + n_contacts] = person_id
            count += n_contacts
        target = np.random.permutation(source)

        return source, target

    def update(self, force: bool = False) -> None:
        """
        Regenerate contacts

        Args:
            force: If True, ignore the `self.dynamic` flag. This is required for initialization.

        """

        if not self.dynamic and not force:
            return

        n_people = len(self.inds)

        # sample the number of edges from a given distribution
        if pd.isna(self.dispersion):
            number_of_contacts = cvu.n_poisson(rate=self.mean_contacts, n=n_people)
        else:
            number_of_contacts = cvu.n_neg_binomial(rate=self.mean_contacts, dispersion=self.dispersion, n=n_people)

        number_of_contacts = np.round(number_of_contacts / 2).astype(cvd.default_int)  # One-way contacts
        self["p1"], self["p2"] = self.get_contacts(self.inds, number_of_contacts)
        self.beta = np.ones(len(self["p1"]), dtype=cvd.default_float)
        self.validate()


class PublicFacingLayer(VictoriaLayer):
    # A public-facing layer has a cluster of staff, and random public contacts

    # POLICY DIMENSIONS
    # beta_layer: reduce transmission along all edges e.g., mask wearing
    # venue_capacity: 4sqm, reduce number of public contacts quadratically. Approximately equivalent to reducing the public beta value
    # rel_staff_beta: Work from home, reduces effective number of staff contacts (randomly, via beta)
    # prop_open: Close some workplaces, removes some staff clusters and also a proportion of public

    # Other parameters
    # rel_public_beta: Reduce probability of contact/transmission for public. Typically prefer using venue_capacity and prop_open instead

    def __init__(self, staff_inds, public_inds, mean_staff_cluster_size, mean_public_contacts, mean_public_staff_contacts, staff_beta, public_beta, *args, **kwargs):
        super().__init__(*args, **kwargs)

        logger.debug(f'Creating PublicFacingLayer "{self.label} with {len(staff_inds)} staff members')
        # Of the people that participate in a layer, how many actually participate each day? Depends on beta.
        # The baseline layer beta represents relative transmission risk assuming the interaction takes place.
        self._staff_layer = ClusterLayer(staff_inds, mean_staff_cluster_size, label=f"{self.label} (staff)", baseline_beta=None)

        self.public_inds = public_inds
        self.mean_public_contacts = mean_public_contacts  # Mean number of public contacts with each other - comparable to the original layer number of contacts
        self.mean_public_staff_contacts = mean_public_staff_contacts  # Number of staff contacts per public

        # Update the layer after changing any of these quantities (and also prop_open)
        self.venue_capacity = 1.0  # Specify fraction of possible public attending - this acts quadratically on beta
        self.rel_staff_beta = 1.0  # Could use this to change proportion working from home
        self.rel_public_beta = 1.0  # Normally wouldn't change this, modulate via venue_capacity.
        self.rel_public_staff_beta = 1.0  # Probably wouldn't change this, but could relate to mask mandates for staff and not customers (i.e. staff-staff and staff-public interactions are lower risk, but not public-public)

        self._baseline_staff_beta = staff_beta  # Baseline staff beta. Don't change this, it's needed to be able to lift policies
        self._baseline_public_beta = public_beta  # Baseline public-staff beta. Don't change this, it's needed to be able to lift policies
        self._baseline_public_staff_beta = public_beta  # Baseline public-staff beta. Don't change this, it's needed to be able to lift policies

        assert self.baseline_beta == 1, 'Baseline beta should be set to 1 as the baseline values in the absence of policies are specified per contact'

        self.update()

    ## FORWARDED ATTRIBUTES FOR THE STAFF CLUSTER COMPONENT

    @property
    def prop_open(self):
        return self._staff_layer.prop_open

    @prop_open.setter
    def prop_open(self, proportion):
        # If the proportion has changed, then `self.update()` will need to be called later on
        # e.g. in Restrictions.apply_package
        self._staff_layer.prop_open = proportion  # Update the staff layer to work out which staff are active

    @property
    def attendance(self):
        return self._staff_layer.attendance

    @attendance.setter
    def attendance(self, proportion):
        # If the attendance has changed, then `self.update()` will need to be called later on
        # e.g. in Restrictions.apply_package
        self._staff_layer.attendance = proportion  # Update the staff layer to work out which staff are active

    @property
    def _attendance_priority(self):
        return self._staff_layer._attendance_priority

    @_attendance_priority.setter
    def _attendance_priority(self, value):
        self._staff_layer._attendance_priority = value

    ## OTHER METHODS

    @property
    def staff_inds(self):
        # note that the `members` Layer attribute reflects the people that actually
        # have contacts e.g. if the staff attendance is reduced, then people working from
        # home won't appear in this array
        return self._staff_layer.members

    @property
    def staff_src(self):
        return self._staff_layer["p1"]

    @property
    def staff_tgt(self):
        return self._staff_layer["p2"]

    def update(self) -> None:
        # Sample new random contacts and update p1/p2 matrices

        # 1. Select which publics participate - this is linear in the proportion of locations that are open
        #    i.e. if only half of the restaurants are open, only half of the people who normally go to restaurants will go
        #         and if only half as many people can attend the venue, only 25% of total contacts would occur
        public_inds = cvu.binomial_filter(self.prop_open * self.venue_capacity, self.public_inds)

        # 2. Assign their number of contacts. This is also linear in the venue capacity e.g. if the venue is only at 50% capacity, an individual will
        #    only interact with half as many people. This should result in a quadratic reduction in contacts i.e. half as many people go, and of the people
        #    that go, they have only half the contacts

        # 2a. Assign the random public-public interactions
        # Members of the public interact with each other, therefore the p2 (target) array is a shuffled version of
        # the src array, and people have approximately twice as many contacts as a result. Therefore for asymmetric
        # contacts where we actually want to have N rather than 2N contacts, we divide the number of contacts by 2
        number_of_contacts = cvu.n_poisson(rate=self.mean_public_contacts * self.venue_capacity, n=len(public_inds))
        number_of_contacts = np.round(number_of_contacts / 2).astype(cvd.default_int)  # One-way contacts
        public_src, public_tgt = RandomLayer.get_contacts(public_inds, number_of_contacts)

        # 2b. Assign the public-staff interactions
        # We need to first compute the number of contacts each person has with a staff member (the source inds)
        # The target inds are *not* from the general public, since they're chosen from the staff. Therefore, people
        # only appear N times in the src array, and the number of contacts does *not* need to be adjusted
        number_of_contacts = cvu.n_poisson(rate=self.mean_public_staff_contacts, n=len(public_inds))
        public_staff_src, _ = RandomLayer.get_contacts(public_inds, number_of_contacts)
        public_staff_tgt = self.staff_inds[cv.choose_r(len(self.staff_inds), len(public_staff_src))]

        self["p1"] = np.concatenate((self.staff_src, public_src, public_staff_src))
        self["p2"] = np.concatenate((self.staff_tgt, public_tgt, public_staff_tgt))
        staff_beta = self.rel_staff_beta * self._baseline_staff_beta * np.ones_like(self.staff_src, dtype=cvd.default_float)
        public_beta = self.rel_public_beta * self._baseline_public_beta * np.ones_like(public_src, dtype=cvd.default_float)
        public_staff_beta = self.rel_public_staff_beta * self._baseline_public_staff_beta * np.ones_like(public_staff_src, dtype=cvd.default_float)
        self.beta = np.concatenate((staff_beta, public_beta, public_staff_beta))
        self.validate()


def random_contacts_within_clusters(clusters, mean_number_of_contacts):
    """
    Random contacts drawn from within clusters

    Args:
        clusters:
        mean_number_of_contacts: Actual mean number of contacts (will be adjusted down for bidirectionaal contacts)

    Returns:

    """

    srcs = []
    tgts = []

    for cluster in clusters:
        number_of_contacts = cvu.n_poisson(rate=mean_number_of_contacts, n=len(cluster))
        number_of_contacts = np.round(number_of_contacts / 2).astype(cvd.default_int)  # One-way contacts
        cluster_array = np.fromiter(cluster, dtype=cvd.default_int)
        src, tgt = RandomLayer.get_contacts(cluster_array, number_of_contacts)
        srcs.append(src)
        tgts.append(tgt)

    return np.concatenate(srcs), np.concatenate(tgts)




class SchoolLayer(VictoriaLayer):

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     # self._age = age.copy()  #: array of all ages - typically copied from people.age

        # assert hasattr(self, 'students')  # return indices of students
        # assert hasattr(self, 'teachers')  # return indices of teachers
        # assert hasattr(self, 'student_schools') #: list of sets of students e.g. [{1,2},{5,12}}
        # assert hasattr(self, 'teacher_schools') #: list of sets of teachers e.g. [{1,2},{5,12}}
        # assert hasattr(self, '_original_beta') #: Individual beta values re-applied when opening/closing year groups

    # @property
    # def students(self) -> np.ndarray: # return indices of students
    #     return self._students
    #
    # @property
    # def teachers(self) -> np.ndarray: # return indices of teachers
    #     return self._teachers
    # @property
    # def student_schools(self) -> list:
    #     return self.student_schools #: list of sets of students e.g. [{1,2},{5,12}}
    #
    # @property
    # def teacher_schools(self) -> list:
    #     return self.teacher_schools #: list of sets of teachers e.g. [{1,2},{5,12}}

    # @property
    # def student_ages(self) -> np.ndarray:
    #     return np.unique(self.age[self.students])

    def get_school_members(self, person_ids) -> set:
        person_ids = set(sc.promotetoarray(person_ids))

        # Work out which schools
        school_inds = set()
        for i, students in enumerate(self.student_schools):
            if students.intersection(person_ids):  # If any of the indices are in this pool of students
                school_inds.add(i)
        for i, teachers in enumerate(self.teacher_schools):
            if teachers.intersection(person_ids):  # If any of the indices are in this pool of students
                school_inds.add(i)

        # Work out who is in those schools
        members = set()
        for i in school_inds:
            members.update(self.student_schools[i])
            members.update(self.teacher_schools[i])

        return members

    def get_class_contacts(self, person_ids): # return classroom contacts
        raise NotImplementedError

    def get_non_class_contacts(self, person_ids):
        raise NotImplementedError

    def get_close_contacts(self, person_ids):
        if not isinstance(person_ids, np.ndarray):
            person_ids = np.array([person_ids], dtype=np.int64)
        return cvu.find_contacts(self['p1'], self['p2'], person_ids)

    def get_school_ids(self, person_ids):
        """
        Return school IDs for requested people

        Args:
            person_ids:

        Returns: Array of school IDs

        """
        if not isinstance(person_ids, np.ndarray):
            person_ids = np.array([person_ids], dtype=np.int64)
        person_ids = set(person_ids)

        match_schools = []
        for i, (students, teachers) in enumerate(zip(self.student_schools, self.teacher_schools)):
            if person_ids.intersection(students) or person_ids.intersection(teachers):
                match_schools.append(i)
        return match_schools

    def set_open_ages(self, ages=None):
        """
        Set which school students can to go to school

        Args:
            ages: Specify ages that have school contacts
                - `None` - all students go to school
                - `[]` - no students go to school (should be equivalent to setting beta to 0)
                - `[5,6,7,...]` - list of ages (not school years) to send to school with 100% daily attendance

                Default value of `None` means ALL students go to school. Otherwise, `[]` means no students to go school

        Returns:

        """

        logger.debug(f"Updating classrooms")

        self.beta = self._original_beta.copy()  # Restore the original beta values

        if ages is None:
            return # this resets beta, all students go to school
        else:

            # Normalize age representation to a dictionary
            if not isinstance(ages, dict):
                # convert [5,6,7] to {5:1, 6:1, 7:1}
                ages = {age: 1 for age in ages}  # A list of ages is shorthand for 100% attendance

            # Map the requested school attendance onto the
            rel_betas = {}
            for age in self.student_ages:
                rel_betas[age] = ages.get(age, 0)

            # Now sanitize the ages
            self.beta[:] = self._original_beta.copy()

            # Send teachers home by default
            self.beta[np.isin(self['p1'], self.teachers) & np.isin(self['p2'], self.teachers)] = 0

            for age, rel_beta in rel_betas.items():
                if rel_beta:
                    logger.debug(f"Turning on classrooms for {age} year olds (rel_beta={rel_beta})")
                else:
                    logger.debug(f"Turning off classrooms for {age} year olds")

                # For student-student contacts, we need to multiply by a factor of rel_beta for each
                # of P1 and P2. This is because if each person has a 50% chance of attending, then there
                # is only a 25% chance that the interaction takes place.
                age_inds = (self._p1_age == age)
                self.beta[age_inds] *= rel_beta
                age_inds = (self._p2_age == age)
                self.beta[age_inds] *= rel_beta

            # Resolve teacher-teacher contacts
            # Teachers are in school if at least one student interacts with them
            # Note that they are guaranteed to be in school so if P1 is present and P2 is
            # not present, then the interaction doesn't take place at all
            nonzero_inds = set(self['p1'][self.beta > 0]).union(set(self['p2'][self.beta > 0])) # All IDs of people at school
            active_teachers = np.fromiter(nonzero_inds.intersection(self.teachers), dtype=cv.default_int) # IDs of all teachers at school
            teacher_inds = np.isin(self['p1'], active_teachers) & np.isin(self['p2'], active_teachers) # IDs of pairwise contacts where both teachers are at school
            self.beta[teacher_inds] = self._original_beta[teacher_inds]

        # WARNING - Note that `layer.reset_beta()` must be called in order for the change in beta
        # value to take effect. This automatically happens via RestrictionSchedule but must otherwise
        # be manually triggered


class PrimarySchoolLayer(SchoolLayer):
    def __init__(self, *, people: cv.People, students: np.array, mean_school_size: float, std_school_size: float, mean_class_size: float, min_class_size: int, potential_teachers: np.array, mean_random_student_contacts, mean_random_staff_contacts, beta_classroom: float, beta_student_student: float, beta_teacher_teacher: float, **kwargs):
        """

        Note that there is a hardcoded minimum school size of 100 people/school
        noting that for very small schools (e.g. regional schools) the contact networks
        may be different again because there is age mixing within classrooms

        Args:
            students: Array of person IDs of students to assign contacts in this layer
            mean_school_size: Mean school size (number of students per school)
            std_school_size:
            mean_class_size: Mean number of students per classroom
            min_class_size:
            potential_teachers:
            mean_random_student_contacts:
            mean_random_staff_contacts:
            **kwargs: e.g. `label`
        """

        super().__init__(**kwargs)

        self._assign_classrooms(people, students, mean_school_size, std_school_size, mean_class_size, min_class_size)
        self._assign_teachers(potential_teachers)

        # Contact arrays for classrooms
        self._classroom_p1, self._classroom_p2 = ClusterLayer.cluster_arrays(self._classrooms)
        classroom_beta = beta_classroom * np.ones(len(self._classroom_p1), dtype=cvd.default_float)

        # Contact arrays for non-classroom student interactions
        self._student_p1, self._student_p2 = random_contacts_within_clusters(self.student_schools, mean_random_student_contacts)
        student_beta = beta_student_student * np.ones_like(self._student_p1, dtype=cvd.default_float)

        # Contact arrays for teacher-to-teacher interactions
        self._teacher_p1, self._teacher_p2 = random_contacts_within_clusters(self.teacher_schools, mean_random_staff_contacts)
        teacher_beta = beta_teacher_teacher * np.ones_like(self._teacher_p1, dtype=cvd.default_float)

        self["p1"] = np.concatenate([self._classroom_p1, self._student_p1, self._teacher_p1])
        self["p2"] = np.concatenate([self._classroom_p2, self._student_p2, self._teacher_p2])
        self.beta = np.concatenate([classroom_beta, student_beta, teacher_beta])
        self._original_beta = self.beta.copy()

        self._p1_age = people.age[self["p1"]]
        self._p2_age = people.age[self["p2"]]

        self.validate()

        assert self.baseline_beta == 1, 'Baseline beta should be set to 1 as the baseline values in the absence of policies are specified per contact'

    def get_class_contacts(self, person_id):
        # Class contacts associated with a person
        if not isinstance(person_id, np.ndarray):
            person_id = np.array([person_id], dtype=np.int64)
        return cvu.find_contacts(self._classroom_p1, self._classroom_p2, person_id)

    def get_non_class_contacts(self, person_ids):
        # Non-classroom contacts associated with a person
        if not isinstance(person_ids, np.ndarray):
            person_ids = np.array([person_ids], dtype=np.int64)
        student_contacts = cvu.find_contacts(self._student_p1, self._student_p2, person_ids)
        teacher_contacts = cvu.find_contacts(self._teacher_p1, self._teacher_p2, person_ids)
        return student_contacts.union(teacher_contacts)

    def _assign_classrooms(self, people, students, mean_school_size, std_school_size, mean_class_size, min_class_size):
        """

        Args:
            people:
            students: Person ID of the students
            mean_school_size:
            std_school_size:

        Returns:

        """

        # First, assign students to a school
        schools = ClusterLayer.create_normal_clusters(students, mean_school_size, std_school_size, min_cluster_size=100)  # nb. minimum school size is hardcoded here

        self.student_schools = [set(x) for x in schools]  #: List of sets, containing students at each school

        # Next, assign classrooms within schools
        self._classrooms = []  # List of lists, containing members of each classroom
        self._classroom_schools = []  # List of school IDs, one for each classroom

        for i, school in enumerate(schools):
            school = np.array(school, dtype=cvd.default_int)
            ages = np.unique(people.age[school])
            for age in ages:
                children_thisage = school[people.age[school] == age]
                new_classrooms = ClusterLayer.create_poisson_clusters(children_thisage, float(mean_class_size), min_class_size)
                self._classrooms.extend(new_classrooms)
                self._classroom_schools.extend([i] * len(new_classrooms))

        self.students = students.copy()
        self.student_ages = np.unique(people.age[self.students])

    def _assign_teachers(self, potential_teachers):
        """
        Produce teachers

        - Add one teacher to each classroom
        - Populate `self.teachers`
        - Populate `self.teacher_schools` to group teachers by school

        Args:
            potential_teachers:

        Returns:

        """
        # For primary schools, assume 1 teacher per classroom

        # Return updated classrooms and teachers
        # Note that the classrooms will be updated in place, if they are a mutable type e.g. list-of-list
        self.teachers = np.random.choice(potential_teachers, len(self._classrooms), replace=False)
        self.teacher_schools = [set() for _ in range(len(self.student_schools))]  # List of sets, containing teachers at each school

        for teacher, classroom, school in zip(self.teachers, self._classrooms, self._classroom_schools):
            classroom.append(teacher)
            self.teacher_schools[school].add(teacher)


class HighSchoolLayer(SchoolLayer):
    def __init__(self, *, people: cv.People, students: np.array, mean_school_size: float, std_school_size: float, potential_teachers: np.array, student_teacher_ratio: float, mean_student_student_classroom_contacts, mean_student_teacher_classroom_contacts, mean_student_student_random_contacts, mean_teacher_teacher_random_contacts, beta_classroom: float, beta_student_student: float, beta_teacher_teacher: float, **kwargs):  # e.g. 14 to have 1 teacher for every 14 students. Note this is at the school level, not classroom level  # Mean number of student-student contacts within classrooms, per day, per student  # Mean number of teachers interacted with by per day, per student  # Mean number of random (outside classroom) student-student contacts, per student  # Mean number of teacher-teacher contacts, per teacher, per day

        super().__init__(**kwargs)

        self._assign_grades(people, students, mean_school_size, std_school_size)
        self._assign_teachers(potential_teachers, student_teacher_ratio)

        # Contact arrays for classroom student-student. In high schools, this is random contacts within a grade
        self._classroom_ss_p1, self._classroom_ss_p2 = random_contacts_within_clusters(self._grades, mean_student_student_classroom_contacts)
        classroom_ss_beta = beta_classroom * np.ones(len(self._classroom_ss_p1), dtype=cvd.default_float)

        # Contact arrays for classroom student-teacher interactions - within a school
        self._classroom_st_p1, self._classroom_st_p2 = self._get_student_teacher_arrays(mean_student_teacher_classroom_contacts)
        classroom_st_beta = beta_classroom * np.ones(len(self._classroom_st_p1), dtype=cvd.default_float)

        # Contact arrays for non-classroom student-student interactions
        self._nonclassroom_ss_p1, self._nonclassroom_ss_p2 = random_contacts_within_clusters(self.student_schools, mean_student_student_random_contacts)
        non_classroom_ss_beta = beta_student_student * np.ones_like(self._nonclassroom_ss_p1, dtype=cvd.default_float)

        # Contact arrays for teacher-teacher interactions
        self._teacher_teacher_p1, self._teacher_teacher_p2 = random_contacts_within_clusters(self.teacher_schools, mean_teacher_teacher_random_contacts)
        teacher_beta = beta_teacher_teacher * np.ones(len(self._teacher_teacher_p1), dtype=cvd.default_float)

        self["p1"] = np.concatenate([self._classroom_ss_p1, self._classroom_st_p1, self._nonclassroom_ss_p1, self._teacher_teacher_p1])
        self["p2"] = np.concatenate([self._classroom_ss_p2, self._classroom_st_p2, self._nonclassroom_ss_p2, self._teacher_teacher_p2])
        self.beta = np.concatenate([classroom_ss_beta, classroom_st_beta, non_classroom_ss_beta, teacher_beta])
        self._original_beta = self.beta.copy()

        self._p1_age = people.age[self["p1"]]
        self._p2_age = people.age[self["p2"]]

        self.validate()

        assert self.baseline_beta == 1, 'Baseline beta should be set to 1 as the baseline values in the absence of policies are specified per contact'

    def get_class_contacts(self, person_id) -> set:
        # Class contacts associated with a person
        if not isinstance(person_id, np.ndarray):
            person_id = np.array([person_id], dtype=np.int64)
        return cvu.find_contacts(np.concatenate([self._classroom_ss_p1, self._classroom_st_p1]), np.concatenate([self._classroom_ss_p2, self._classroom_st_p2]), person_id)

    def get_non_class_contacts(self, person_id) -> set:
        # School contacts associated with a person
        # This does NOT include class contacts
        # To find all contacts, use `Layer.find_contacts(person_id)`
        if not isinstance(person_id, np.ndarray):
            person_id = np.array([person_id], dtype=np.int64)
        student_contacts = cvu.find_contacts(self._nonclassroom_ss_p1, self._nonclassroom_ss_p2, person_id)
        teacher_contacts = cvu.find_contacts(self._teacher_teacher_p1, self._teacher_teacher_p2, person_id)
        return student_contacts.union(teacher_contacts)

    def _get_student_teacher_arrays(self, mean_student_teacher_classroom_contacts: float):
        """

        Args:
            self:
            mean_student_teacher_contacts: This is the average number of teachers that a student interacts with per day

        Returns:

        """
        srcs = []
        tgts = []
        for students, teachers in zip(self.student_schools, self.teacher_schools):
            number_of_contacts = cvu.n_poisson(rate=mean_student_teacher_classroom_contacts, n=len(students))
            src, _ = RandomLayer.get_contacts(np.fromiter(students, dtype=cvd.default_int), number_of_contacts)
            teachers = np.fromiter(teachers, dtype=cvd.default_int)
            tgt = teachers[cv.choose_r(len(teachers), len(src))]
            srcs.append(src)
            tgts.append(tgt)

        return np.concatenate(srcs), np.concatenate(tgts)

    def _assign_grades(self, people, students, mean_school_size, std_school_size):
        """

        Args:
            people:
            students: Person ID of the students
            mean_school_size:
            std_school_size:

        Returns:

        """
        # First, assign students to a school
        schools = ClusterLayer.create_normal_clusters(students, mean_school_size, std_school_size, min_cluster_size=100)

        self.student_schools = [set(x) for x in schools]  #: List of sets, containing students at each school

        # For high schools, there are no classrooms - only grades
        # Similar to having a single classroom for all students of the particular age
        self._grades = []  # List of lists, containing members of each grade (within each school)
        self._grade_schools = []  # List of school IDs, one for each grade/cohort

        for i, school in enumerate(schools):
            school = np.array(school)
            ages = np.unique(people.age[school])
            for age in ages:
                children_thisage = school[people.age[school] == age]
                self._grades.append(list(children_thisage))
                self._grade_schools.append(i)

        self.students = students.copy()
        self.student_ages = np.unique(people.age[self.students])

    def _assign_teachers(self, potential_teachers, student_teacher_ratio):
        """
        Produce teachers

        For high schools, this is done by assigning a group of teachers to a school

        - Populate `self.teachers`
        - Populate `self.teacher_schools` to group teachers by school

        Args:
            potential_teachers:

        Returns:

        """

        self.teacher_schools = [set() for _ in range(len(self.student_schools))]  # List of sets, containing teachers at each school

        potential_teachers = np.random.permutation(potential_teachers)  # shuffle teachers - also copying them

        ptr = 0
        for i, school in enumerate(self.student_schools):
            n_teachers = int(np.maximum(1, np.round(len(school) / student_teacher_ratio)))  # number of teachers at this school
            self.teacher_schools[i].update(potential_teachers[ptr: ptr + n_teachers])
            ptr += n_teachers

        self.teachers = np.fromiter(set.union(*self.teacher_schools), dtype=cvd.default_int)

class ChildCareLayer(VictoriaLayer):
    def __init__(self, people: cv.People, students: np.array, mean_classroom_size: float, potential_teachers: np.array, *args, **kwargs):
        """

        Args:
            people: cv.People instance
            students: Array of indices of students to be assigned to classrooms
            mean_classroom_size: Classroom size distribution mean (Poisson)
            potential_teachers:
            ages:
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)

        classrooms = self._assign_classrooms(people, students, mean_classroom_size)
        self.students = np.concatenate(classrooms)
        classrooms, self.teachers = self._assign_teachers(people, classrooms, potential_teachers)

        self["p1"], self["p2"] = ClusterLayer.cluster_arrays(classrooms)
        self.beta = np.ones(len(self["p1"]), dtype=cvd.default_float)
        self._original_beta = self.beta.copy()

        self._p1_age = people.age[self["p1"]]
        self._p2_age = people.age[self["p2"]]

        self.validate()

        logger.debug(f'Creating ChildCareLayer "{self.label} with {len(self.students)} students and {len(self.teachers)} teachers')
        self._open_ages = None

    @staticmethod
    def _assign_classrooms(people, students, mean_classroom_size):
        classrooms = []
        ages = np.unique(people.age[students])
        for age in ages:
            children_thisage = students[people.age[students] == age]
            classrooms.extend(ClusterLayer.create_poisson_clusters(children_thisage, mean_classroom_size, 0))
        return classrooms

    @staticmethod
    def _assign_teachers(people, classrooms, potential_teachers):
        # For child care, the student-teacher ratio depends on the age
        # Ratios based on https://www.acecqa.gov.au/nqf/educator-to-child-ratios
        educator_to_child_ratio = {0: 4, 1: 4, 2: 4, 3: 10, 4: 10}  # An error will be raised if an unexpected age is entered in layers.csv, more ages can be added here but would be good to know
        potential_teachers = np.random.permutation(potential_teachers.copy())  # It should already be randomized, but just in case its not
        teachers = []
        idx = 0  # Teacher index
        for i in range(len(classrooms)):
            age = people.age[classrooms[i][0]]
            n_teachers = int(np.ceil(len(classrooms[i]) / educator_to_child_ratio[age]))
            classrooms[i].extend(potential_teachers[idx: idx + n_teachers])
            teachers.extend(potential_teachers[idx: idx + n_teachers].tolist())
            idx += n_teachers

        return classrooms, teachers

    def set_open_ages(self, ages=None):
        """
        Set which school students can to go to school

        Args:
            ages: Specify ages that have school contacts
                - `None` - all students go to school
                - `[]` - no students go to school (should be equivalent to setting beta to 0)
                - `[5,6,7,...]` - list of ages (not school years) to send to school with 100% daily attendance

                Default value of `None` means ALL students go to school. Otherwise, `[]` means no students to go school

        Returns:

        """

        logger.debug(f"Updating classrooms")
        self._open_ages = ages

        if ages is None:
            # Shortcut to allow all students to attend by default
            self.beta[:] = 1
        elif not ages:
            # handle [] or {}
            self.beta[:] = 0
        else:

            # Normalize age representation to a dictionary
            if not isinstance(ages, dict):
                # convert [5,6,7] to {5:1, 6:1, 7:1}
                ages = {age: 1 for age in ages}

            self.beta[:] = 0
            for age, beta in ages.items():
                logger.debug(f"Turning on classrooms for {age} year olds (rel_beta={beta})")
                self.beta[self._p1_age == age] = beta
                self.beta[self._p2_age == age] = beta


## INTERNAL HELPERS FOR HOUSEHOLD GENERATION

## Fast choice implementation
# From https://gist.github.com/jph00/30cfed589a8008325eae8f36e2c5b087
# by Jeremy Howard https://twitter.com/jeremyphoward/status/955136770806444032
@nb.njit
def sample(n, q, J, r1, r2):
    res = np.zeros(n, dtype=np.int32)
    lj = len(J)
    for i in range(n):
        kk = int(np.floor(r1[i] * lj))
        if r2[i] < q[kk]:
            res[i] = kk
        else:
            res[i] = J[kk]
    return res


class AliasSample:
    def __init__(self, probs):
        self.K = K = len(probs)
        self.q = q = np.zeros(K)
        self.J = J = np.zeros(K, dtype=np.int)

        smaller, larger = [], []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small, large = smaller.pop(), larger.pop()
            J[small] = large
            q[large] = q[large] - (1.0 - q[small])
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

    def draw_one(self):
        K, q, J = self.K, self.q, self.J
        kk = int(np.floor(np.random.rand() * len(J)))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]

    def draw_n(self, n):
        r1, r2 = np.random.rand(n), np.random.rand(n)
        return sample(n, self.q, self.J, r1, r2)


def _sample_household_cluster(sampler, bin_lower, bin_upper, reference_age, n):
    """
    Return list of ages in a household/location based on mixing matrix and reference person age
    """

    ages = [reference_age]  # The reference person is in the household/location

    if n > 1:
        idx = np.digitize(reference_age, bin_lower) - 1  # First, find the index of the bin that the reference person belongs to
        sampled_bins = sampler[idx].draw_n(n - 1)

        for bin in sampled_bins:
            ages.append(int(round(np.random.uniform(bin_lower[bin] - 0.5, bin_upper[bin] + 0.5))))

    return np.array(ages)


def _make_households(n_households, pop_size, household_heads, mixing_matrix):
    """

    The mixing matrix is a direct read of the CSV file, with index corresponding to 'Age group' i.e.

    >>> mixing_matrix = pd.read_csv('mixing_H.csv',index_col='Age group')
    >>> mixing_matrix
                   0 to 4    5 to 9
        Age group
        0 to 4     0.659868  0.503965
        5 to 9     0.314777  0.895460


    :param n_households:
    :param pop_size:
    :param household_heads:
    :return:
        h_clusters: a list of lists in which each sublist contains
                    the IDs of the people who live in a specific household
        ages: flattened array of ages, corresponding to the UID positions
    """
    mixing_matrix = mixing_matrix.div(mixing_matrix.sum(axis=1), axis=0)
    samplers = [AliasSample(mixing_matrix.iloc[i, :].values) for i in range(mixing_matrix.shape[0])]  # Precompute samplers for each reference age bin

    age_lb = [int(x.split(" to ")[0]) for x in mixing_matrix.index]
    age_ub = [int(x.split(" to ")[1]) for x in mixing_matrix.index]

    h_clusters = []
    uids = np.arange(0, pop_size)
    ages = np.zeros(pop_size, dtype=int)
    h_added = 0
    p_added = 0

    for h_size, h_num in n_households.iteritems():
        for household in range(h_num):
            head = household_heads[h_added]
            # get ages of people in household
            household_ages = _sample_household_cluster(samplers, age_lb, age_ub, head, h_size)
            # add ages to ages array
            ub = p_added + h_size
            ages[p_added:ub] = household_ages
            # get associated UID that defines a household cluster
            h_ids = uids[p_added:ub]
            h_clusters.append(h_ids)
            # increment sliding windows
            h_added += 1
            p_added += h_size
    return h_clusters, ages
