import covasim as cv
import pandas as pd
import covasim_aus_schools as cvv
import covasim.utils as cvu
from covasim_aus_schools import logger
import matplotlib.pyplot as plt
import numpy as np


def add_result(sim, name, vals):
    if name in sim.results:
        raise Exception(f'Attempted to add result "{name}" which already exists')
    sim.results[name] = cv.Result(name=name, npts=sim.npts)
    sim.results[name].values = vals.copy()


class CommunityExposureDays(cv.Analyzer):
    def initialize(self, sim):
        super().initialize(sim)
        self.new_exposure_days = np.zeros_like(sim.tvec)

    def finalize(self, sim):
        super().finalize(sim)
        add_result(sim, "new_exposure_days", self.new_exposure_days * sim.rescale_vec)
        add_result(sim, "cum_exposure_days", np.cumsum(sim.results["new_exposure_days"].values))

    def apply(self, sim):

        # For people that were diagnosed today
        inds = cvu.true(sim.people.date_diagnosed == sim.t)
        # If they have been diagnosed, then the sim.people.quarantined flag has already been reset
        # Therefore we instead need to check the date_quarantined, which does not get cleared
        quarantined = np.isfinite(sim.people.date_quarantined[inds])

        # Count how many days between them being diagnosed and them entering quarantine
        # If they didn't enter quarantine, then we need to go to the current date
        # But virtually nobody will not be quarantined due to the requirement to isolate while waiting for a test
        new_exposures_quarantined = np.minimum(sim.t, sim.people.date_quarantined[inds[quarantined]]) - sim.people.date_infectious[inds[quarantined]]
        new_exposures_nonquarantined = sim.t - sim.people.date_infectious[inds[~quarantined]]

        self.new_exposure_days[sim.t] = new_exposures_quarantined[new_exposures_quarantined > 0].sum() + new_exposures_nonquarantined[new_exposures_nonquarantined > 0].sum()


class IsolationStatus(cv.Analyzer):
    def initialize(self, sim):
        super().initialize(sim)
        self.fully_isolated = np.zeros_like(sim.tvec)
        self.partially_isolated = np.zeros_like(sim.tvec)
        self.not_isolated = np.zeros_like(sim.tvec)

    def finalize(self, sim):
        super().finalize(sim)
        add_result(sim, "fully_isolated", self.fully_isolated * sim.rescale_vec)
        add_result(sim, "partially_isolated", self.partially_isolated * sim.rescale_vec)
        add_result(sim, "not_isolated", self.not_isolated * sim.rescale_vec)

    def apply(self, sim):

        inds = cvu.true(sim.people.date_diagnosed == sim.t)

        # If they have been diagnosed, then the sim.people.quarantined flag has already been reset
        # Therefore we instead need to check the date_quarantined, which does not get cleared
        date_infectious = sim.people.date_infectious[inds]
        date_quarantined = sim.people.date_quarantined[inds]
        date_diagnosed = sim.people.date_diagnosed[inds]  # Should all be today...

        quarantined = np.isfinite(date_quarantined)

        self.not_isolated[sim.t] = (~quarantined).sum() + np.sum((date_quarantined[quarantined] >= date_diagnosed[quarantined]))
        self.fully_isolated[sim.t] = np.sum(date_quarantined[quarantined] <= date_infectious[quarantined])  # <= because if they were quarantined on the day they became infectious, that happens before transmission occurs
        self.partially_isolated[sim.t] = np.sum((date_quarantined[quarantined] > date_infectious[quarantined]) & (date_quarantined[quarantined] < date_diagnosed[quarantined]))

        assert self.not_isolated[sim.t] + self.fully_isolated[sim.t] + self.partially_isolated[sim.t] == len(inds)


class SymptomToPositiveTime(cv.Analyzer):
    # nb it's possible for the same agent to get diagnosed multiple
    # times, if they are reset. Therefore, we can't store this by agent but instead
    # need to just store it as a record

    def initialize(self, sim):
        super().initialize(sim)
        self._records = []

    def finalize(self, sim):
        super().finalize(sim)
        df = pd.DataFrame(self._records, columns=["day", "symptom_to_positive"])
        df = df.fillna(-1)
        df["record"] = True
        df = df.set_index(["day", "symptom_to_positive"])
        df = df.groupby(level=["day", "symptom_to_positive"]).count()
        df = df.unstack("day")
        self.df = df

    def apply(self, sim):

        # For people that were diagnosed today
        inds = cvu.true(sim.people.date_diagnosed == sim.t)
        for date_symptomatic, date_tested in zip(sim.people.date_symptomatic[inds], sim.people.date_tested[inds]):
            if not np.isfinite(date_symptomatic) or date_symptomatic > date_tested:
                self._records.append((sim.t, np.nan))
            else:
                self._records.append((sim.t, date_tested - date_symptomatic))


class NetTransmission(cv.Analyzer):
    """
    Simple net transmission metric

    This represents the potential transmission due to mixing (but not accounting for contact tracing/isolation etc.)
    Essentially just showing the number of immediate downstream infections for an undiagnosed person per day
    Note that the actual number of downstream infections would be cumulative over the infectious period.

    This analyzer is mainly intended to monitor the impact of restrictions to confirm they are impacting dynamics
    and to qualitatively assess their relative impacts.
    """

    def initialize(self, sim):
        super().initialize(sim)
        self.net_transmission = np.zeros_like(sim.tvec, dtype=cv.default_float)
        self.transmission = {layer: np.zeros_like(sim.tvec) for layer in sim.people.contacts}
        self.transmission_no_beta = {layer: np.zeros_like(sim.tvec) for layer in sim.people.contacts}

    def apply(self, sim):
        for layer in sim.people.contacts:
            self.transmission[layer][sim.t] = sim["beta_layer"][layer] * sim.people.contacts[layer]["beta"].sum()
            self.transmission_no_beta[layer][sim.t] = sim.people.contacts[layer]["beta"].sum()

        self.net_transmission[sim.t] = sum(x[sim.t] for x in self.transmission.values()) / sim.n

    def finalize(self, sim):
        super().finalize(sim)
        add_result(sim, "net_transmission", self.net_transmission)


class MysteryCases(cv.Analyzer):
    def initialize(self, sim):
        super().initialize(sim)
        self.mystery_cases = np.zeros_like(sim.tvec)

    def finalize(self, sim):
        super().finalize(sim)
        add_result(sim, "new_mystery_cases", self.mystery_cases * sim.rescale_vec)

    def apply(self, sim):
        self.mystery_cases[sim.t] = np.sum((sim.people.date_diagnosed == sim.t) & ~sim.people.known_contact)


class LogRestrictions(cv.Analyzer):
    def initialize(self, sim):
        super().initialize(sim)
        self.restrictions = [None] * sim.npts

    def apply(self, sim):
        self.restrictions[sim.t] = sim._restrictions


class ProportionQuarantined(cv.Analyzer):
    def initialize(self, sim):
        super().initialize(sim)
        self.prop_agents_quarantined = np.zeros_like(sim.tvec, dtype=cv.default_float)

    def finalize(self, sim):
        super().finalize(sim)
        add_result(sim, "prop_agents_quarantined", self.prop_agents_quarantined)

    def apply(self, sim):
        self.prop_agents_quarantined[sim.t] = sim.people.quarantined.sum() / sim.n


class RollingAverageDiagnoses(cv.Analyzer):
    # Add rolling average diagnoses to the results
    def finalize(self, sim):
        # nb. Results have already been scaled by this point
        super().finalize(sim)

        def get_average(window):
            avg = cvv.rolling_average(sim.results["new_diagnoses"].values, window)
            return abs(np.around(avg, decimals=5))

        add_result(sim, "new_diagnoses_7d_avg", get_average(7))
        add_result(sim, "new_diagnoses_14d_avg", get_average(14))

    def apply(self, sim):
        return


class LogPrognoses(cv.Analyzer):
    # Add rolling average diagnoses to the results
    def initialize(self, sim):
        super().initialize(sim)
        self.rel_sus = []
        self.rel_trans = []
        self.symp_prob = []
        self.severe_prob = []
        self.crit_prob = []
        self.death_prob = []

    def finalize(self, sim):
        # nb. Results have already been scaled by this point
        super().finalize(sim)
        self.rel_sus = np.vstack(self.rel_sus)
        self.rel_trans = np.vstack(self.rel_trans)
        self.symp_prob = np.vstack(self.symp_prob)
        self.severe_prob = np.vstack(self.severe_prob)
        self.crit_prob = np.vstack(self.crit_prob)
        self.death_prob = np.vstack(self.death_prob)

    def apply(self, sim):
        self.rel_sus.append(sim.people.rel_sus.copy())
        self.rel_trans.append(sim.people.rel_trans.copy())
        self.symp_prob.append(sim.people.symp_prob.copy())
        self.severe_prob.append(sim.people.severe_prob.copy())
        self.crit_prob.append(sim.people.crit_prob.copy())
        self.death_prob.append(sim.people.death_prob.copy())


class KnownActiveAnalyzer(cv.Analyzer):
    # Record daily known active cases
    def initialize(self, sim):
        super().initialize(sim)
        self.n_known_active = np.zeros(sim.npts, dtype=cv.default_float)
        self.n_known_severe = np.zeros(sim.npts, dtype=cv.default_float)

    def finalize(self, sim):
        super().finalize(sim)
        add_result(sim, "n_known_active", self.n_known_active * sim.rescale_vec)
        add_result(sim, "n_known_severe", self.n_known_severe * sim.rescale_vec)

    def apply(self, sim):
        self.n_known_active[sim.t] = (sim.people.exposed & sim.people.diagnosed).sum()
        self.n_known_severe[sim.t] = (sim.people.exposed & sim.people.diagnosed & sim.people.severe).sum()


class ExtraHospitalOutputs(cv.Analyzer):
    # Age distribution of severe and critical, each day
    # n_active each time point
    # known_active

    # Add rolling average diagnoses to the results
    age_bins = np.arange(0, 120)

    def initialize(self, sim):
        super().initialize(sim)
        self.age_active = []
        self.age_known_active = []
        self.age_severe = []
        self.age_critical = []

    def finalize(self, sim):
        # nb. Results have already been scaled by this point
        super().finalize(sim)
        self.age_active = np.vstack(self.age_active)  # Age distribution of active cases
        self.age_known_active = np.vstack(self.age_known_active)  # Age distribution of known active cases
        self.age_severe = np.vstack(self.age_severe)  # Age distribution of known severe cases
        self.age_critical = np.vstack(self.age_critical)  # Age distribution of known critical

    def apply(self, sim):
        active = sim.people.exposed
        known_active = sim.people.exposed & sim.people.diagnosed
        self.age_active.append(np.histogram(sim.people.age[active], self.age_bins)[0])
        self.age_known_active.append(np.histogram(sim.people.age[known_active], self.age_bins)[0])
        self.age_severe.append(np.histogram(sim.people.age[known_active & sim.people.severe], self.age_bins)[0])
        self.age_critical.append(np.histogram(sim.people.age[known_active & sim.people.critical], self.age_bins)[0])

    def plot(self):
        plt.figure()
        plt.bar(self.age_bins[:-1], self.age_active[100, :])
        plt.bar(self.age_bins[:-1], self.age_known_active[100, :])
        plt.bar(self.age_bins[:-1], self.age_severe[100, :])
        plt.bar(self.age_bins[:-1], self.age_critical[100, :])

        np.nanmax(self.age_severe.sum(axis=1) / self.age_known_active.sum(axis=1))
        np.nanmax(self.age_critical.sum(axis=1) / self.age_known_active.sum(axis=1))
        np.nanmax(self.age_severe.sum(axis=1) / self.age_active.sum(axis=1))
        np.nanmax(self.age_critical.sum(axis=1) / self.age_active.sum(axis=1))
        np.nanmax(self.age_critical.sum(axis=1) / self.age_severe.sum(axis=1))


class NumHospICU(cv.Analyzer):
    """
    Track number of hospitalisations and ICU admissions derived from severe and critical outcomes.
    """

    def initialize(self, sim):
        super().initialize(sim)
        self.n_hospitalised = np.zeros_like(sim.tvec, dtype=cv.default_float)
        self.n_icu = np.zeros_like(sim.tvec, dtype=cv.default_float)

    def finalize(self, sim):
        super().finalize(sim)
        add_result(sim, "n_hospitalised", self.n_hospitalised * sim.rescale_vec)
        add_result(sim, "n_icu", self.n_icu * sim.rescale_vec)

    def apply(self, sim):
        idx = np.digitize(sim.people.age, sim.pars["prognoses"]["age_cutoffs"]) - 1  # Bin for each person
        self.n_hospitalised[sim.t] = (sim.people.severe * sim.pars["prognoses"]["hosp_given_severe"][idx]).sum()
        self.n_icu[sim.t] = (sim.people.critical * sim.pars["prognoses"]["ICU_given_critical"][idx]).sum()

    def plot(self, sim):
        plt.figure()
        plt.plot(sim.tvec, sim.results["n_hospitalised"], color="b", alpha=1, label="Hospitalisations")
        plt.plot(sim.tvec, sim.results["n_icu"], color="r", alpha=1, label="ICU usage")
        plt.plot(sim.tvec, sim.results["n_severe"], color="g", alpha=1, label="Severe")
        plt.plot(sim.tvec, sim.results["n_critical"], color="c", alpha=1, label="Critical")
        plt.legend()


class AdultVaccinated(cv.Analyzer):
    def initialize(self, sim):
        super().initialize(sim)
        self.prop_adult_vaccinated = np.zeros_like(sim.tvec, dtype=cv.default_float)

    def finalize(self, sim):
        super().finalize(sim)
        add_result(sim, "prop_adult_vaccinated", self.prop_adult_vaccinated)

    def apply(self, sim):
        age_vals = sim.people.age >= 16
        self.prop_adult_vaccinated[sim.t] = sim.people.fully_vaccinated[age_vals].sum() / (sim.people.age >= 16).sum()


# class VacSevereCritical(cv.Analyzer):
#     def initialize(self, sim):
#         super().initialize(sim)
#         self.new_hospital_vac = np.zeros_like(sim.tvec, dtype=cv.default_float)
#         self.new_hospital_nonvac = np.zeros_like(sim.tvec, dtype=cv.default_float)
#         self.new_icu_vac = np.zeros_like(sim.tvec, dtype=cv.default_float)
#         self.new_icu_nonvac = np.zeros_like(sim.tvec, dtype=cv.default_float)
#
#     def finalize(self, sim):
#         super().finalize(sim)
#         add_result(sim, "n_hospitalised", self.n_hospitalised * sim.rescale_vec)
#         add_result(sim, "n_icu", self.n_icu * sim.rescale_vec)
#
#     def apply(self, sim):
#
#         self.n_hospitalised[sim.t] = (sim.people.severe * sim.pars["prognoses"]["hosp_given_severe"][idx]).sum()
#         self.n_icu[sim.t] = (sim.people.critical * sim.pars["prognoses"]["ICU_given_critical"][idx]).sum()
#
#         # new_severe
#         new_severe =        cv.true(sim.people.date_severe == sim.t)
#         new_severe_vac = np.count_nonzero(sim.people.vaccinated[new_severe])
#         new_severe_nonvac = len(new_severe)-len(new_severe_vac)
#
#         new_critical =        cv.true(sim.people.date_critical == sim.t)
#         new_critical_vac = np.count_nonzero(sim.people.vaccinated[new_critical])
#         new_critical_nonvac = len(new_critical)-len(new_critical_vac)
#
#
#         np.count_nonzero(~sim.people.vaccinated[new_severe])
#
#         self.new_hospital_vac = np.zeros_like(sim.tvec, dtype=cv.default_float)
#
#
#         print(new_severe)
#         # self.severe, self.date_severe
#         # inds = self.check_inds(self.severe, self.date_severe, filter_inds=self.is_exp)
#         #


class DiagnosedSevere(cv.Analyzer):
    """
    Record whether a person was severely ill at the time of their diagnosis
    """

    def initialize(self, sim):
        super().initialize(sim)
        self.new_diagnoses_severe = np.zeros_like(sim.tvec, dtype=cv.default_float)
        self.prop_diagnoses_severe = np.zeros_like(sim.tvec, dtype=cv.default_float)

    def finalize(self, sim):
        super().finalize(sim)
        add_result(sim, "new_diagnoses_severe", self.new_diagnoses_severe * sim.rescale_vec)
        add_result(sim, "prop_diagnoses_severe", self.prop_diagnoses_severe)

    def apply(self, sim):
        inds = cvu.true(sim.people.date_diagnosed == sim.t)
        self.new_diagnoses_severe[sim.t] = sim.people.severe[inds].sum()
        if len(inds):
            self.prop_diagnoses_severe[sim.t] = self.new_diagnoses_severe[sim.t] / len(inds)


class ProportionSevere(cv.Analyzer):
    """
    Cumulative severe/diagnosed
    """

    def apply(self, sim):
        return

    def finalize(self, sim):
        super().finalize(sim)
        a = sim.results["cum_severe"].values
        b = sim.results["cum_diagnoses"].values
        add_result(sim, "proportion_severe", np.divide(a, b, out=np.zeros_like(a), where=b > 0))


class DiagnosedVaccinated(cv.Analyzer):
    """
    Record whether a person was vaccinated at the time of their diagnosis
    """

    age_bins = np.array((0, 15, 59, 120))

    def initialize(self, sim):
        super().initialize(sim)
        logger.warning(f"Warning: {self.__class__.__name__} is a performance intensive analyzer")

        self.diagnosed = []
        self.diagnosed_vaccinated = []
        self.diagnosed_fully_vaccinated = []

    def finalize(self, sim):
        super().finalize(sim)
        self.diagnosed = np.vstack(self.diagnosed)
        self.diagnosed_vaccinated = np.vstack(self.diagnosed_vaccinated)
        self.diagnosed_fully_vaccinated = np.vstack(self.diagnosed_fully_vaccinated)

    def apply(self, sim):
        diagnosed_today = cvu.true(sim.people.date_diagnosed == sim.t)
        self.diagnosed.append(np.histogram(sim.people.age[diagnosed_today], self.age_bins)[0])

        inds = diagnosed_today[sim.people.vaccinated[diagnosed_today]]
        self.diagnosed_vaccinated.append(np.histogram(sim.people.age[inds], self.age_bins)[0])

        inds = diagnosed_today[sim.people.fully_vaccinated[diagnosed_today]]
        self.diagnosed_fully_vaccinated.append(np.histogram(sim.people.age[inds], self.age_bins)[0])


class SchoolTPAnalyzer(cv.Analyzer):
    # Check transmission in schools and childcare

    ages = np.arange(0, 18)

    def initialize(self, sim):
        super().initialize(sim)
        logger.warning(f"Warning: {self.__class__.__name__} is a performance intensive analyzer")
        self.tp = np.zeros((sim.npts, len(self.ages)))

        self._age_cache = {}
        for layer_name, layer in sim.people.contacts.items():
            if isinstance(layer, cvv.SchoolLayer) or isinstance(layer, cvv.PrimarySchoolLayer) or isinstance(layer, cvv.HighSchoolLayer):
                self._age_cache[layer_name] = {}
                for i, age in enumerate(self.ages):
                    self._age_cache[layer_name][age] = (sim.people.age[layer["p1"]] == age) | (sim.people.age[layer["p2"]] == age)

    def finalize(self, sim):
        super().finalize(sim)
        self.tp = np.vstack(self.tp)
        self._age_cache = None

    def apply(self, sim):
        for layer_name, layer in sim.people.contacts.items():
            if isinstance(layer, cvv.SchoolLayer) or isinstance(layer, cvv.PrimarySchoolLayer) or isinstance(layer, cvv.HighSchoolLayer):
                for i, age in enumerate(self.ages):
                    inds = self._age_cache[layer_name][age]
                    self.tp[sim.t, i] += sim["beta_layer"][layer_name] * layer["beta"][inds].sum()
