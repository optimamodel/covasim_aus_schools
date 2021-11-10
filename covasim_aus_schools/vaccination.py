import covasim as cv
import pandas as pd
import covasim_aus_schools as cvv
import covasim.misc as cvm
import sciris as sc
from functools import partial
import networkx as nx
import covasim.utils as cvu
from collections import defaultdict
import numba as nb

import numpy as np
from covasim_aus_schools import logger
from .analyzers import add_result
import matplotlib.pyplot as plt

# GENERATE THE RUNS


class VaccineAnalyzer(cv.Analyzer):
    # nb. for simulations that have rising immunity, this is just tracking whether the person had at
    # least 1 dose

    def __init__(self):
        super().__init__()
        self.data = defaultdict(list)
        self.df = None  # Store final output as a dataframe

    def apply(self, sim: cv.Sim):
        vac = cvu.true(sim.people.vaccinated)
        nonvac = cvu.false(sim.people.vaccinated)

        self.data["n_vac_severe"].append(np.sum(sim.people.severe[vac]))
        self.data["n_nonvac_severe"].append(np.sum(sim.people.severe[nonvac]))
        self.data["n_vac_critical"].append(np.sum(sim.people.critical[vac]))
        self.data["n_nonvac_critical"].append(np.sum(sim.people.critical[nonvac]))

        self.data["new_vac_infections"].append(np.sum(sim.people.date_exposed[vac] == sim.t))
        self.data["new_vac_severe"].append(np.sum(sim.people.date_severe[vac] == sim.t))
        self.data["new_vac_critical"].append(np.sum(sim.people.date_critical[vac] == sim.t))
        self.data["new_vac_deaths"].append(np.sum(sim.people.date_dead[vac] == sim.t))

        self.data["new_nonvac_infections"].append(np.sum(sim.people.date_exposed[nonvac] == sim.t))
        self.data["new_nonvac_severe"].append(np.sum(sim.people.date_severe[nonvac] == sim.t))
        self.data["new_nonvac_critical"].append(np.sum(sim.people.date_critical[nonvac] == sim.t))
        self.data["new_nonvac_deaths"].append(np.sum(sim.people.date_dead[nonvac] == sim.t))

        # self.data["new_deaths"].append(np.sum(sim.people.date_dead == sim.t)) # Validation

    def finalize(self, sim: cv.Sim):
        df = pd.DataFrame.from_dict(self.data)
        assert len(df) == sim.t

        df = df.multiply(sim.rescale_vec, axis=0)

        cs = df[[x for x in df.columns if x.startswith("new_")]].cumsum()
        cs.columns = [x.replace("new_", "cum_") for x in cs.columns]

        df = pd.concat([df, cs], axis=1)

        df.index = sim.datevec
        df.index.name = "date"

        # assert max(sim.results['cum_deaths']-df['cum_deaths']) < 1e-3 # Check that the rescaling is correctly done and the time points align
        self.df = df


# Simple single-dose instant protection vaccine


class VaccinationProgram(cv.Intervention):
    """
    Vaccine program. For the moment, this intervention assumes it's the only vaccine
    """

    def __init__(self, sequence, poi, pos, poh, poc, pod, vac_per_week):
        """

        Args:
            sequence:
            poi:
            pos:
            poh:
            poc:
            pod:
            people_per_day: Number of people to acquire full protection per day
                - A scalar number of people
                - A function `f(sim) -> float`
        """
        super().__init__()
        self.sequence = sequence  #: Array of person inds, in the order in which to vaccinate
        self.poi = poi
        self.pos = pos
        self.poh = poh
        self.poc = poc
        self.pod = pod
        self.people_per_day = int((vac_per_week / 7) / 2)  # People per day
        self._agents_per_day = None
        self.vaccine_coverage = None

    def initialize(self, sim: cv.Sim):
        super().initialize(sim)
        # The vaccination scaling is done only once, because the number of vaccines distributed amonst the
        # population being modelled needs to dynamically increase with the rescale factor. That is, it doesn't
        # depend on the rescale_vec, but rather it depends on the total population scale
        self._agents_per_day = int(np.round(self.people_per_day / sim["pop_scale"]))
        self.vaccine_coverage = np.zeros(sim.npts)

    def vaccinate(self, sim, n_agents):
        # Vaccinate n agents
        assert sim.initialized, "Must initialize the cv.Sim prior to vaccinating anyone"

        n_agents = min(len(self.sequence), n_agents)
        inds = self.sequence[:n_agents]
        self.sequence = self.sequence[n_agents:]

        # All-or-nothing immunity against infection
        sus_inds = cvu.binomial_filter(self.poi, inds)  # nb. poi=1 means that all vaccinated people are immune
        sim.people.rel_sus[sus_inds] = 0

        # Overall probabilities
        sim.people.symp_prob[inds] *= 1 - self.pos
        sim.people.severe_prob[inds] *= 1 - self.poh
        sim.people.crit_prob[inds] *= 1 - self.poc
        sim.people.death_prob[inds] *= 1 - self.pod

        sim.people.vaccinated[inds] = True

    def apply(self, sim: cv.Sim):
        if len(self.sequence):
            self.sequence = self.sequence[~sim.people.vaccinated[self.sequence]]  # Remove anyone that has already been vaccinated e.g. by another intervention
            self.sequence = self.sequence[~sim.people.dead[self.sequence]]  # Remove anyone that is dead
            self.vaccinate(sim, self._agents_per_day)

        alive = ~sim.people.dead
        self.vaccine_coverage[sim.t] = sim.people.vaccinated[alive].sum() / alive.sum()


# Simplified vaccine with immunity timecourse - single dose parametrization


class Vaccine:
    def __init__(self, name, dose_interval, immunity_timecourse, protection_timecourse, prevent_infection, prevent_transmission, prevent_symp, prevent_severe, prevent_crit, prevent_death):
        """

        Args:
            characteristics:  Vaccine characteristic dictionary - containing poi, pos, poh, poc, pod
            rising_immunity:
        """

        self.name = name
        self.dose_interval = dose_interval  # Track when the second dose is scheduled to be delivered, for the purpose of counting second doses. Set to None for if single dose only
        self.prevent_infection = prevent_infection  # Prevention of infection
        self.prevent_transmission = prevent_transmission
        self.prevent_symp = prevent_symp  # Prevention of symptoms
        self.prevent_severe = prevent_severe  # Prevention of hospitalisation
        self.prevent_crit = prevent_crit  # Prevention of critical
        self.prevent_death = prevent_death  # Prevention of death

        self._immunity_timecourse = immunity_timecourse  # Timecourse for protection against infection (poi); [(t,v)] list of interpolation control points
        self._protection_timecourse = protection_timecourse  # Timecourse for protection against all other states (pos, poh, poc, pod); [(t,v)] list of interpolation control points

    def _interpolate(self, vals: list, t):
        vals = sorted(vals, key=lambda x: x[0])  # Make sure values are sorted
        assert len({x[0] for x in vals}) == len(vals)  # Make sure time points are unique
        return np.interp(t, [x[0] for x in vals], [x[1] for x in vals], left=vals[0][1], right=vals[-1][1])

    def immunity_timecourse(self, t: np.array) -> np.array:
        return self._interpolate(self._immunity_timecourse, t)

    def protection_timecourse(self, t: np.array) -> np.array:
        return self._interpolate(self._protection_timecourse, t)

    @property
    def full_protection_time(self) -> int:
        # Return time taken to reach full immunity
        return max(x[0] for x in self._immunity_timecourse + self._protection_timecourse)

    # Constructors for common vaccine types
    @classmethod
    def pfizer(cls, dose_interval=21):
        # Standard Pfizer parameters, parametrized by dose interval

        time_to_first_dose_peak = 12  # Immunity after first dose reaches its peak after this time
        if dose_interval < time_to_first_dose_peak:
            # If the second dose is delivered before the first dose peak is reached, then set a control point accordingly
            immunity_timecourse = [(0, 0), (dose_interval, 0.71 * dose_interval / time_to_first_dose_peak), (dose_interval + 12, 1)]
            protection_timecourse = [(0, 0), (dose_interval, 0.94 * dose_interval / time_to_first_dose_peak), (dose_interval + 12, 1)]
        else:
            immunity_timecourse = [(0, 0), (time_to_first_dose_peak - 1e-3, 0.71), (dose_interval, 0.71), (dose_interval + 12, 1)]
            protection_timecourse = [(0, 0), (time_to_first_dose_peak - 1e-3, 0.94), (dose_interval, 0.94), (dose_interval + 12, 1)]

        vaccine_characteristics = {
            "prevent_infection": 0.8,
            "prevent_transmission": 0.65 * 0.48, #0.48 adjustment is because the model has a reduction in onward transmission due to asymptomatic cases
            "prevent_symp": 0.20,
            "prevent_severe": 0.81,
            "prevent_crit": 0.0,
            "prevent_death": 0.0,
        }
        return cls(name="pfizer", dose_interval=dose_interval, immunity_timecourse=immunity_timecourse, protection_timecourse=protection_timecourse, **vaccine_characteristics)

    @classmethod
    def astra_zeneca(cls, dose_interval=84):
        # Standard AZ parameters, parametrized by dose interval
        time_to_first_dose_peak = 28  # Immunity after first dose reaches its peak after this time
        # standard_interval = 84
        peak_protection = 1  # min(1, dose_interval / standard_interval)

        if dose_interval < time_to_first_dose_peak:
            # If the second dose is delivered before the first dose peak is reached, then set a control point accordingly
            immunity_timecourse = [(0, 0), (dose_interval, 0.69 * dose_interval / time_to_first_dose_peak), (dose_interval + 12, peak_protection)]
            protection_timecourse = [(0, 0), (dose_interval, 0.87 * dose_interval / time_to_first_dose_peak), (dose_interval + 12, peak_protection)]
        else:
            immunity_timecourse = [(0, 0), (time_to_first_dose_peak - 1e-3, 0.69), (dose_interval, 0.69), (dose_interval + 12, peak_protection)]
            protection_timecourse = [(0, 0), (time_to_first_dose_peak - 1e-3, 0.87), (dose_interval, 0.87), (dose_interval + 12, peak_protection)]

        vaccine_characteristics = {
            "prevent_infection": 0.67,
            "prevent_transmission": 0.36 * 0.83, #0.48 adjustment is because the model has a reduction in onward transmission due to asymptomatic cases
            "prevent_symp": 0.12,
            "prevent_severe": 0.76,
            "prevent_crit": 0.0,
            "prevent_death": 0.0,
        }
        return cls(name="astra_zeneca", dose_interval=dose_interval, immunity_timecourse=immunity_timecourse, protection_timecourse=protection_timecourse, **vaccine_characteristics)


class TimedVaccinationProgram(cv.Intervention):
    # This intervention models people receiving a vaccine with immunity that builds over time

    leaky = True  # Flag for leaky vs non-leaky vaccines (applies to all vaccination programs)

    def __init__(self, vaccine, sequence=None, num_doses=0, *args, **kwargs):
        """

        Args:
            vaccine: A ``Vaccine`` instance (defined above)
            sequence:
            num_doses: - A scalar, a callable `fcn(sim)` or an array the same size as sim.tvec

        """
        super().__init__(*args, **kwargs)
        self.sequence = sequence  # Specify vaccine sequence, None means random order for everyone. Otherwise, an array or a callable
        self.num_doses = num_doses  # Specify number of doses as scalar, dict (by date or day), or callable function
        self.vaccine = vaccine  # e.g. `rising_immunity_pfizer_3w` - should be sorted
        self.n_people_vaccinated = None
        self.n_agents_vaccinated = None

    def initialize(self, sim: cv.Sim):
        super().initialize(sim)

        self._vaccinated = np.full(sim.n, False, dtype=bool)  # True if someone was vaccinated using THIS vaccine
        self._date_immune = np.full(sim.n, fill_value=np.nan, dtype=cv.default_float)  # Track date people became immune due to this intervention
        self._pending_immunity = np.full(sim.n, False, dtype=bool)  # Boolean flag for whether people are immune or not
        self.n_people_vaccinated = np.zeros(sim.npts)
        self.n_agents_vaccinated = np.zeros(sim.npts)

        # Convert any dates to simulation days
        if isinstance(self.num_doses, dict):
            self.num_doses = {sim.day(k): v for k, v in self.num_doses.items()}

        # Convert the vaccine sequence into an array
        if callable(self.sequence):
            self.sequence = self.sequence(sim.people)
        elif self.sequence is None:
            self.sequence = np.random.permutation(sim.n)
        else:
            self.sequence = sc.promotetoarray(self.sequence)

        self._immunity_timecourse = self.vaccine.immunity_timecourse(np.arange(0, self.vaccine.full_protection_time + 1))  # Cache the immunity function
        self._protection_timecourse = self.vaccine.protection_timecourse(np.arange(0, self.vaccine.full_protection_time + 1))  # Cache the immunity function

    # At the start, we want to vaccinate a bunch of people, and start them out with a level of prior immunity
    def update_immunity(self, sim, t):

        # For the remaining vaccine characteristics, scale the outcome by proportion of protection
        vaccinated = cv.true(self._vaccinated)  # Indices of people that were vaccinated using this intervention
        date_vaccinated = sim.people.date_vaccinated[vaccinated]  # Vaccination date for people vaccinated using this intervention
        duration_since_vaccinated = sim.t - date_vaccinated
        duration_since_vaccinated = np.minimum(duration_since_vaccinated, len(self._protection_timecourse) - 1).astype(cv.default_int)  # Max out protection
        assert not np.any(duration_since_vaccinated < 0)  # Cannot have negative durations, can disable this check for performance if required

        # Update fully vaccinated status for anyone that has recieved their second dose (if applicable)
        if self.vaccine.dose_interval is not None:
            gain_fully_vaccinated = cv.true((~sim.people.fully_vaccinated[vaccinated]) & (duration_since_vaccinated >= self.vaccine.dose_interval))
            sim.people.fully_vaccinated[vaccinated[gain_fully_vaccinated]] = True

        # Apply immunity
        if not self.leaky:
            # Update immunity for anyone that is due to gain immunity by today
            pending_immunity = cv.true(self._pending_immunity)  # Indices of people that have pending immunity
            immune_today = cvu.itrue(t >= self._date_immune[pending_immunity], pending_immunity)  # Indices of people that should gain immunity today
            sim.people.rel_sus[immune_today] = 0
            self._pending_immunity[immune_today] = False
        else:
            immunity = self._immunity_timecourse[duration_since_vaccinated]
            sim.people.rel_sus[vaccinated] = sim.people.baseline_rel_sus[vaccinated] * (1 - self.vaccine.prevent_infection * immunity)

        # Update protection for today
        protection = self._protection_timecourse[duration_since_vaccinated]
        sim.people.rel_trans[vaccinated] = sim.people.baseline_rel_trans[vaccinated] * (1 - self.vaccine.prevent_transmission * protection)
        sim.people.symp_prob[vaccinated] = sim.people.baseline_symp_prob[vaccinated] * (1 - self.vaccine.prevent_symp * protection)
        sim.people.severe_prob[vaccinated] = sim.people.baseline_severe_prob[vaccinated] * (1 - self.vaccine.prevent_severe * protection)
        sim.people.crit_prob[vaccinated] = sim.people.baseline_crit_prob[vaccinated] * (1 - self.vaccine.prevent_crit * protection)
        sim.people.death_prob[vaccinated] = sim.people.baseline_death_prob[vaccinated] * (1 - self.vaccine.prevent_death * protection)

    @staticmethod
    def _get_date_immune(inds, n_immune):
        """
        Args:
            inds: array of person indices
            n_immune: array of how many people should be immune for each day after vaccination (the first day is 0)
                      The length of this array is arbitrary, the length of this list defines the maximum value present
                      in the output array
        Returns: A list the same length as `inds`
        """

        n_gain_immunity = np.diff(n_immune)  # Number of people that gain immunity each day
        day_immune = np.zeros(inds.shape, dtype=cv.default_int)

        count = 0
        for i in range(0, len(n_gain_immunity)):
            n_contacts = n_gain_immunity[i]
            day_immune[count : count + n_contacts] = i + 1  # The first entry (0th) in n_gain_immunity corresponds to gaining immunity 1 day afterwards
            count += n_contacts

            if count == len(day_immune):
                break

        return np.random.permutation(day_immune)  # Shuffle the order in which people gain immunity

    def vaccinate(self, sim, inds, t=None, update_immunity=True):
        """
        Use this function to vaccinate a group of people

        Args:
            sim:
            inds:
            t: Override vaccination date relative to simulation date (e.g. for historical vaccination). Can be a day index or a date

        Returns:

        """

        if t is None:
            t = sim.t
        elif not sc.isnumber(t):
            t = sim.day(t)

        # Validate indices of people to vaccinate - *essential* that we don't vaccinate anyone using this
        # intervention that has already been vaccinated, otherwise this intervention will interact unintentionally
        # with any other interventions that vaccinated that same person
        assert not np.any(sim.people.vaccinated[inds] | sim.people.dead[inds]), "Cannot re-vaccinate people with this intervention"  # nb. can disable this check for performance later on if needed

        # logger.debug(f'{self.label}: Vaccinating {len(inds)} agents at {t=}')
        sim.people.vaccinated[inds] = True
        if self.vaccine.dose_interval is None:
            # If it's a single dose vaccine, they are fully vaccinated after the first dose
            sim.people.fully_vaccinated[inds] = True
        self._vaccinated[inds] = True  # Record that they have been vaccinated by this intervention
        sim.people.date_vaccinated[inds] = t

        if not self.leaky:
            immune_inds = cv.binomial_filter(self.vaccine.prevent_infection, inds)  # Indices of people that will eventually gain immunity if 100% relative protection is reached
            n_immune = (self._immunity_timecourse * len(immune_inds)).astype(int)
            self._date_immune[immune_inds] = self._get_date_immune(immune_inds, n_immune)
            self._pending_immunity[immune_inds] = True  # Flag that these people should have their immunity updated

        if t >= 0:
            # Record the new vaccinations etc. if they are taking place during the simulation timeframe
            # We assume that the vaccination rollout continues outside the pool of agents being modelled here.
            # Following from the example in the `apply` method, if we vaccinate 10 agents, this corresponds to
            # vaccinating 5 people within the area being modelled i.e. 10*5. However, the area being modelled
            # accounts for a (5/10) fraction of the total population. Therefore the number of people vaccinated
            # is in fact 10*5/(5/10). As before, the factors cancel out, and we actually just multiply
            # by the pop_scale here
            new_vaccinated = int(len(inds) * sim["pop_scale"])
            sim.people.flows["new_vaccinations"] += new_vaccinated
            sim.people.flows["new_vaccinated"] += new_vaccinated
            self.n_people_vaccinated[t] += new_vaccinated
            self.n_agents_vaccinated[t] += int(len(inds))

        if update_immunity:
            self.update_immunity(sim, t=t)

    def apply(self, sim, t=None, num_people=None):
        """
        Use this function to vaccinate a number of people, selected based on the stored sequence
        """

        if t is None:
            t = sim.t
        elif not sc.isnumber(t):
            t = sim.day(t)

        # Work out how many *people* to vaccinate today - matches reported numbers of doses for a jurisdiction
        if num_people is None:
            if sc.isnumber(self.num_doses):
                num_people = self.num_doses
            elif callable(self.num_doses):
                num_people = self.num_doses(sim)
            elif t in self.num_doses:
                num_people = self.num_doses[t]
            else:
                num_people = 0

        # Suppose we have a pop_scale of 10, and a current scale factor of 5, with 100 agents.
        # That means that we have a total population of 1000 people, and currently 1 person
        # represents 5 people. If we want to vaccinate 100 people in the population, how many
        # agents does this correspond to? Since we are currently modelling 500 people with 100
        # agents, assuming the 100 vaccines are uniformly distributed, we need to allocate
        # n = 100*(5/10) = 50 vaccines to the pool of people being modelled. Further, we then
        # need to divide this number of people by the current scale factor to get the number of
        # agents corresponding to 50 people. That is, 50/5=10 agents. Since the overall calculation is
        # (100*5/10)/5 this is just equivalent to dividing by the overall pop scale.
        num_agents = int(np.round(num_people / sim["pop_scale"]))

        if num_agents and len(self.sequence):
            # People are ineligible for vaccination if they are already vaccinated or if they are dead
            # However, people who have died and are then returned to the simulation by rescaling still need
            # to be eligible for vaccination. Therefore we don't actually remove them from the schedule since they
            # could come back to life at any time in the simulation
            eligible = self.sequence[~sim.people.vaccinated[self.sequence] & ~sim.people.dead[self.sequence]]
            inds = eligible[:num_agents]  # nb. this indexing
        else:
            inds = np.array([], dtype=cv.default_int)

        # Vaccinate them
        if len(inds):
            # Check inds at this point, in case num_agents > 0 but nobody was eligible and thus no vaccinations were performed
            self.vaccinate(sim, inds, t=t)
        else:
            self.update_immunity(sim, t=t)

        return inds


class TimedVaccineCoverageAnalyzer(cv.Analyzer):
    def initialize(self, sim: cv.Sim):
        super().initialize(sim)
        self.first_dose_coverage = np.zeros_like(sim.tvec, dtype=cv.default_float)
        self.second_dose_coverage = np.zeros_like(sim.tvec, dtype=cv.default_float)

    def finalize(self, sim: cv.Sim):
        super().finalize(sim)
        add_result(sim, "vac_first_dose_coverage", self.first_dose_coverage)
        add_result(sim, "vac_second_dose_coverage", self.second_dose_coverage)

    def apply(self, sim: cv.Sim):
        self.first_dose_coverage[sim.t] = sim.people.vaccinated.sum() / len(sim.people)
        self.second_dose_coverage[sim.t] = sim.people.fully_vaccinated.sum() / len(sim.people)


class VaccineCoverageByAge(cv.Analyzer):
    def initialize(self, sim: cv.Sim):
        super().initialize(sim)

        logger.warning(f"Warning: {self.__class__.__name__} is a performance intensive analyzer")

        coverage = np.full((sim.npts, sim.n), fill_value=False, dtype=bool)  # At each time, store the vaccination status of each person
        self.first_dose = sc.odict()
        self.second_dose = sc.odict()
        for iv in sim.get_interventions(TimedVaccinationProgram):
            if iv.vaccine.name not in self.first_dose:
                self.first_dose[iv.vaccine.name] = coverage.copy()
                self.second_dose[iv.vaccine.name] = coverage.copy()
        self._start_day = sim["start_day"]
        self._ages = sim.people.age.copy()

    def apply(self, sim: cv.Sim):
        for iv in sim.get_interventions(TimedVaccinationProgram):
            self.first_dose[iv.vaccine.name][sim.t, :] = self.first_dose[iv.vaccine.name][sim.t, :] | (sim.people.vaccinated & iv._vaccinated)
            self.second_dose[iv.vaccine.name][sim.t, :] = self.second_dose[iv.vaccine.name][sim.t, :] | (sim.people.fully_vaccinated & iv._vaccinated)

    def plot_timeseries(self, ax=None):

        age_bins = [(0, 11), (12, 15), (16, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 79), (80, 84), (85, np.inf)]

        vaccines = ["pfizer", "astra_zeneca"]  # hardcode the vaccines

        light_blue = "#deebf7"
        mid_blue = "#9ecae1"
        dark_blue = "#3182bd"
        light_red = "#fee0d2"
        mid_red = "#fc9272"
        dark_red = "#de2d26"
        gray = "#999999"

        az_colors = [mid_blue, light_blue]
        pfizer_colors = [mid_red, light_red]

        tvec = np.arange(self.first_dose[0].shape[0])

        # Proportion of population in each age group
        stacked_series = []  # Alternate [second dose, first dose, unvaccinated] for each age group
        colors = []
        for age_lower, age_upper in age_bins:
            ages = (self._ages >= age_lower) & (self._ages <= age_upper)
            total = 0
            for vaccine in vaccines:
                second_dose = (self.second_dose[vaccine] & ages).sum(axis=1)  # .sum()
                first_dose = (self.first_dose[vaccine] & ages).sum(axis=1) - second_dose

                total += first_dose
                total += second_dose

                stacked_series.append(second_dose)
                stacked_series.append(first_dose)

                if vaccine == "pfizer":
                    colors += pfizer_colors
                else:
                    colors += az_colors

            unvaccinated = ages.sum() - total
            stacked_series.append(unvaccinated)
            colors.append(gray)

        if ax is None:
            fig, ax = plt.subplots()

        ax.stackplot(tvec, stacked_series, colors=colors)

        for age_lower, age_upper in age_bins:
            ages = self._ages <= age_upper
            ax.axhline(ages.sum(), color="k")

        yticks = []
        yticklabels = []

        for i in range(len(age_bins)):
            ages = age_bins[i]

            if ages[1] == np.inf:
                yl = f"{ages[0]}+"
            else:
                yl = f"{ages[0]}-{ages[1]}"

            age_proportions = ((self._ages <= ages[0]).sum(), (self._ages <= ages[1]).sum())

            yticks.append((age_proportions[1] - age_proportions[0]) / 2 + age_proportions[0])
            yticklabels.append(yl)

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontweight="bold")
        ax.set_ylim(0, len(self._ages))

        ax.set_xlim(0, len(tvec) - 1)

        ax.set_xlabel("Days")

    def plot_bar_graph(self, t=None, ax=None, age_bins=None):
        if t is None:
            t = self.first_dose[0].shape[0] - 1
        elif not sc.isnumber(t):
            t = sc.day(t, start_day=self._start_day)

        vaccines = ["pfizer", "astra_zeneca"]  # hardcode the vaccines

        light_blue = "#deebf7"
        mid_blue = "#9ecae1"
        dark_blue = "#3182bd"
        light_red = "#fee0d2"
        mid_red = "#fc9272"
        dark_red = "#de2d26"
        gray = "#cccccc"

        if age_bins is None:
            ages = np.arange(0, max(self._ages) + 1)
            xvals = ages[:-1]
        else:
            ages = [cvv.parse_age_range(age_bins[0])[0]] + [cvv.parse_age_range(x)[1] for x in age_bins]
            xvals = age_bins

        az, _ = np.histogram(self._ages[self.first_dose["astra_zeneca"][t, :]], ages)
        pfizer, _ = np.histogram(self._ages[self.first_dose["pfizer"][t, :]], ages)
        total, _ = np.histogram(self._ages, ages)

        total_coverage = (az.sum() + pfizer.sum()) / total.sum()
        if age_bins is None:
            adult_coverage = (pfizer[16:].sum() + az[16:].sum()) / total[16:].sum()

        total_proportion = total / total.sum()

        az = np.divide(az * total_proportion, total, out=np.zeros_like(az, dtype=cv.default_float), where=total > 0)
        pfizer = np.divide(pfizer * total_proportion, total, out=np.zeros_like(pfizer, dtype=cv.default_float), where=total > 0)

        if ax is None:
            fig, ax = plt.subplots()

        ax.bar(xvals, pfizer, label="Pfizer", color=dark_red)
        ax.bar(xvals, az, bottom=pfizer, label="Astra-Zeneca", color=dark_blue)
        ax.bar(xvals, total_proportion - az - pfizer, bottom=az + pfizer, label="Unvaccinated", color=gray)

        ax.set_xlabel("Age")
        ax.set_ylabel("Proportion of total population")
        ax.legend()

        if age_bins is None:
            ax.set_title(f"{sc.date(t, start_date=self._start_day,as_date=False)}\nTotal vaccine coverage: {total_coverage*100:.0f}% ({adult_coverage*100:.0f}% of 16+)")
        else:
            ax.set_title(f"{sc.date(t, start_date=self._start_day,as_date=False)}\nTotal vaccine coverage: {total_coverage*100:.0f}%")


### Vaccine mandates

# A vaccine mandate is implemented by setting the per-person beta in each layer to 0 (or otherwise
# scaling it down by the proportion required)

from cykhash import Int32Set_from_buffer, isin_int32


def cykhash_isin(a, lookup):
    # lookup e.g. cykhash.Int32Set_from_buffer(inds_to_match.astype(np.int32))
    result = np.empty(a.size, dtype=np.bool)
    isin_int32(a, lookup, result)  # running time O(b.size)
    return result


class VaccineMandate(cv.Intervention):
    def __init__(self, layers, compliance, start_date, end_date=np.inf, **kwargs):
        # WARNING - A layer should only appear in ONE vaccine mandate intervention - otherwise there'll be a cumulative reduction in unvaccinated transmission
        super().__init__(**kwargs)
        self.layers = layers  #: Layers that the vaccine mandate applies to
        self.compliance = compliance  #: Proportion of unvaccinated that are compliant. 1 means that nobody unvaccinated attends the layer anyway
        self.start_date = start_date
        self.end_date = end_date

    def initialize(self, sim):
        super().initialize(sim)
        self.start_date = sim.day(self.start_date)
        self.end_date = sim.day(self.end_date) if not self.end_date == np.inf else np.inf

    def _find_ineligible(self, sim) -> np.array:
        """
        Determine who is eligible for participation under the mandate

        Args:
            sim:

        Returns: Array of indices for people that are eligible

        """
        # Return array of people that are eligible for contacts
        raise NotImplementedError

    def apply(self, sim):
        if not (sim.t >= self.start_date and sim.t < self.end_date):  # End date is not inclusive, so that way a single dose mandate can end on the same day a double dose mandate begins
            return

        ineligible = self._find_ineligible(sim)  # Find people that should be excluded by the mandate
        ineligible_lookup = Int32Set_from_buffer(ineligible.astype(np.int32))

        for layer_name in self.layers:
            layer = sim.people.contacts[layer_name]
            p1_ineligible = cykhash_isin(layer["p1"], ineligible_lookup)
            p2_ineligible = cykhash_isin(layer["p2"], ineligible_lookup)
            layer["beta"][p1_ineligible] *= 1 - self.compliance
            layer["beta"][p2_ineligible] *= 1 - self.compliance

            n_excluded = np.count_nonzero(p1_ineligible | p2_ineligible)

            if len(layer) > 0:
                logger.debug(f"Day {sim.t}: Exclusions in {layer_name} applied to {n_excluded} ({n_excluded/len(layer)*100:.0f}% of contacts)")
            else:
                logger.debug(f"Day {sim.t}: Exclusions in {layer_name} applied to {n_excluded} (no contacts)")


class SingleDoseVaccineMandate(VaccineMandate):
    def _find_ineligible(self, sim):
        return cv.false(sim.people.vaccinated)


class DoubleDoseVaccineMandate(VaccineMandate):
    def _find_ineligible(self, sim):
        return cv.false(sim.people.fully_vaccinated)


class RetailVaccineMandate(VaccineMandate):
    def __init__(self, *args, staff_compliance, public_compliance, double_dose, **kwargs):
        super().__init__(*args, compliance=0, **kwargs)
        self.staff_compliance = staff_compliance
        self.public_compliance = public_compliance
        self.double_dose = double_dose

    def initialize(self, sim):
        super().initialize(sim)
        for layer in self.layers:
            assert isinstance(sim.people.contacts[layer], cvv.PublicFacingLayer), "RetailVaccineMandate must operate on PublicFacingLayer instances"

    def apply(self, sim):
        if not (sim.t >= self.start_date and sim.t < self.end_date):  # End date is not inclusive, so that way a single dose mandate can end on the same day a double dose mandate begins
            return

        for layer_name in self.layers:
            layer = sim.people.contacts[layer_name]

            # Vaccinated staff
            staff = layer._staff_layer.inds.astype(np.int32)
            public = layer.public_inds.astype(np.int32)

            if self.double_dose:
                vaccinated_staff = staff[sim.people.fully_vaccinated[staff]]
                vaccinated_public = public[sim.people.fully_vaccinated[public]]
            else:
                vaccinated_staff = staff[sim.people.vaccinated[staff]]
                vaccinated_public = public[sim.people.vaccinated[public]]

            staff_lookup = Int32Set_from_buffer(staff)
            public_lookup = Int32Set_from_buffer(public)
            vaccinated_staff_lookup = Int32Set_from_buffer(vaccinated_staff)
            vaccinated_public_lookup = Int32Set_from_buffer(vaccinated_public)

            ineligible_staff_p1 = cykhash_isin(layer["p1"], staff_lookup) & ~cykhash_isin(layer["p1"], vaccinated_staff_lookup)
            ineligible_staff_p2 = cykhash_isin(layer["p2"], staff_lookup) & ~cykhash_isin(layer["p2"], vaccinated_staff_lookup)
            ineligible_public_p1 = cykhash_isin(layer["p1"], public_lookup) & ~cykhash_isin(layer["p1"], vaccinated_public_lookup)
            ineligible_public_p2 = cykhash_isin(layer["p2"], public_lookup) & ~cykhash_isin(layer["p2"], vaccinated_public_lookup)

            layer["beta"][ineligible_staff_p1] *= 1 - self.staff_compliance
            layer["beta"][ineligible_staff_p2] *= 1 - self.staff_compliance
            layer["beta"][ineligible_public_p1] *= 1 - self.public_compliance
            layer["beta"][ineligible_public_p2] *= 1 - self.public_compliance

            n_excluded = np.count_nonzero(ineligible_staff_p1 | ineligible_staff_p2 | ineligible_public_p1 | ineligible_public_p2)
            if len(layer) > 0:
                logger.debug(f"Day {sim.t}: Exclusions in {layer_name} applied to {n_excluded} ({n_excluded/len(layer)*100:.0f}% of contacts)")
            else:
                logger.debug(f"Day {sim.t}: Exclusions in {layer_name} applied to {n_excluded} (no contacts)")

            if len(layer) > 0:
                logger.debug(f"Day {sim.t}: Excluded {len(staff)-len(vaccinated_staff)} staff of {len(staff)}")
