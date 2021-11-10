import datetime as dt
from collections import defaultdict

import covasim as cv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pylab as pl
from covasim import utils as cvu
from scipy.interpolate import interp1d
import sciris as sc
import itertools

from covasim_aus_schools.analyzers import add_result


class PolicySchedule(cv.Intervention):
    def __init__(self, baseline: dict, policies: dict):
        """
        Create policy schedule

        The policies passed in represent all of the possible policies that a user
        can subsequently schedule using the methods of this class

        Example usage:

            baseline = {'H':1, 'S':0.75}
            policies = {}
            policies['Close schools'] = {'S':0}
            schedule = PolicySchedule(baseline, policies)
            schedule.add('Close schools', 10) # Close schools on day 10
            schedule.end('Close schools', 20) # Reopen schools on day 20
            schedule.remove('Close schools')  # Don't use the policy at all

        Args:
            baseline: Baseline (relative) beta layer values e.g. {'H':1, 'S':0.75}
            policies: Dict of policies containing the policy name and relative betas for each policy e.g. {policy_name: {'H':1, 'S':0.75}}

        """
        super().__init__()
        self._baseline = baseline  #: Store baseline relative betas for each layer
        self.policies = sc.dcp(policies)  #: Store available policy interventions (name and effect)
        for policy in self.policies:
            self.policies[policy] = {k: v for k, v in self.policies[policy].items() if not pd.isna(v)}
            assert set(self.policies[policy].keys()).issubset(self._baseline.keys()), f'Policy "{policy}" has effects on layers not included in the baseline'
        self.policy_schedule = []  #: Store the scheduling of policies [(start_day, end_day, policy_name)]
        self.days = {}  #: Internal cache for when the beta_layer values need to be recalculated during simulation. Updated using `_update_days`

    def start(self, policy_name: str, start_day: int) -> None:
        """
        Change policy start date

        If the policy is not already present, then it will be added with no end date

        Args:
            policy_name: Name of the policy to change start date for
            start_day: Day number to start policy

        Returns: None

        """
        n_entries = len([x for x in self.policy_schedule if x[2] == policy_name])
        if n_entries < 1:
            self.add(policy_name, start_day)
            return
        elif n_entries > 1:
            raise Exception("start_policy() cannot be used to start a policy that appears more than once - need to manually add an end day to the desired instance")

        for entry in self.policy_schedule:
            if entry[2] == policy_name:
                entry[0] = start_day

        self._update_days()

    def end(self, policy_name: str, end_day: int) -> None:
        """
        Change policy end date

        This only works if the policy only appears once in the schedule. If a policy gets used multiple times,
        either add the end days upfront, or insert them directly into the policy schedule. The policy should
        already appear in the schedule

        Args:
            policy_name: Name of the policy to end
            end_day: Day number to end policy (policy will have no effect on this day)

        Returns: None

        """

        n_entries = len([x for x in self.policy_schedule if x[2] == policy_name])
        if n_entries < 1:
            raise Exception("Cannot end a policy that is not already scheduled")
        elif n_entries > 1:
            raise Exception("end_policy() cannot be used to end a policy that appears more than once - need to manually add an end day to the desired instance")

        for entry in self.policy_schedule:
            if entry[2] == policy_name:
                if end_day <= entry[0]:
                    raise Exception(f"Policy '{policy_name}' starts on day {entry[0]} so the end day must be at least {entry[0]+1} (requested {end_day})")
                entry[1] = end_day

        self._update_days()

    def add(self, policy_name: str, start_day: int, end_day: int = np.inf) -> None:
        """
        Add a policy to the schedule

        Args:
            policy_name: Name of policy to add
            start_day: Day number to start policy
            end_day: Day number to end policy (policy will have no effect on this day)

        Returns: None

        """
        assert policy_name in self.policies, "Unrecognized policy"
        self.policy_schedule.append([start_day, end_day, policy_name])
        self._update_days()

    def remove(self, policy_name: str) -> None:
        """
        Remove a policy from the schedule

        All instances of the named policy will be removed from the schedule

        Args:
            policy_name: Name of policy to remove

        Returns: None

        """

        self.policy_schedule = [x for x in self.policy_schedule if x[2] != policy_name]
        self._update_days()

    def _update_days(self) -> None:
        # This helper function updates the list of days on which policies start or stop
        # The apply() function only gets run on those days
        self.days = {x[0] for x in self.policy_schedule}.union({x[1] for x in self.policy_schedule if np.isfinite(x[1])})

    def _compute_beta_layer(self, t: int) -> dict:
        # Compute beta_layer at a given point in time
        # The computation is done from scratch each time
        beta_layer = self._baseline.copy()
        for start_day, end_day, policy_name in self.policy_schedule:
            rel_betas = self.policies[policy_name]
            if t >= start_day and t < end_day:
                for layer in beta_layer:
                    if layer in rel_betas:
                        beta_layer[layer] *= rel_betas[layer]
        return beta_layer

    def apply(self, sim: cv.BaseSim):
        if sim.t in self.days:
            sim["beta_layer"] = self._compute_beta_layer(sim.t)
            if sim["verbose"]:
                print(f"PolicySchedule: Changing beta_layer values to {sim['beta_layer']}")
                for entry in self.policy_schedule:
                    if sim.t == entry[0]:
                        print(f"PolicySchedule: Turning on {entry[2]}")
                    elif sim.t == entry[1]:
                        print(f"PolicySchedule: Turning off {entry[2]}")

    def plot_gantt(self, max_time=None, start_date=None, interval=None, pretty_labels=None):
        """
        Plot policy schedule as Gantt chart

        Returns: A matplotlib figure with a Gantt chart

        """
        fig, ax = plt.subplots()
        if max_time:
            max_time += 5
        else:
            max_time = np.nanmax(np.array([x[1] for x in self.policy_schedule if np.isfinite(x[1])]))

        # end_dates = [x[1] for x in self.policy_schedule if np.isfinite(x[1])]
        if interval:
            xmin, xmax = ax.get_xlim()
            ax.set_xticks(pl.arange(xmin, xmax + 1, interval))

        if start_date:

            @ticker.FuncFormatter
            def date_formatter(x, pos):
                return (start_date + dt.timedelta(days=x)).strftime("%b-%d")

            ax.xaxis.set_major_formatter(date_formatter)
            if not interval:
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.set_xlabel("Dates")
            ax.set_xlim((0, max_time + 5))  # Extend a few days so the ends of policies can be seen

        else:
            ax.set_xlim(0, max_time + 5)  # Extend a few days so the ends of policies can be seen
            ax.set_xlabel("Days")
        schedule = sc.dcp(self.policy_schedule)
        if pretty_labels:
            policy_index = {pretty_labels[x]: i for i, x in enumerate(self.policies.keys())}
            for p, pol in enumerate(schedule):
                pol[2] = pretty_labels[pol[2]]
            colors = sc.gridcolors(len(pretty_labels))
        else:
            policy_index = {x: i for i, x in enumerate(self.policies.keys())}
            colors = sc.gridcolors(len(self.policies))
        ax.set_yticks(np.arange(len(policy_index.keys())))
        ax.set_yticklabels(list(policy_index.keys()))
        ax.set_ylim(0 - 0.5, len(policy_index.keys()) - 0.5)

        for start_day, end_day, policy_name in schedule:
            if not np.isfinite(end_day):
                end_day = 1e6  # Arbitrarily large end day
            ax.broken_barh([(start_day, end_day - start_day)], (policy_index[policy_name] - 0.5, 1), color=colors[policy_index[policy_name]])

        return fig


class SeedInfection(cv.Intervention):
    """
    Seed a fixed number of infections

    This class facilities seeding a fixed number of infections on a per-day
    basis.

    Infections will only be seeded on specified days

    """

    def __init__(self, infections: dict):
        """

        Args:
            infections: Dictionary with {day_index:n_infections}

        """
        super().__init__()
        self.infections = infections  #: Dictionary mapping {day: n_infections}. Day can be an int, or a string date like '20200701'

    def initialize(self, sim):
        super().initialize(sim)
        self.infections = {sim.day(k): v for k, v in self.infections.items()}  # Convert any day strings to ints
        self.n_seeded = np.zeros(sim.npts)

    def apply(self, sim):
        if sim.t in self.infections:
            susceptible_inds = cvu.true(sim.people.susceptible)

            if len(susceptible_inds) < self.infections[sim.t]:
                raise Exception("Insufficient people available to infect")

            targets = cvu.choose(len(susceptible_inds), self.infections[sim.t])
            target_inds = susceptible_inds[targets]
            sim.people.infect(inds=target_inds, layer="seed_infection")
            self.n_seeded[sim.t] = len(target_inds)

    def finalize(self, sim=None):
        super().finalize(sim)
        add_result(sim, "n_agents_seeded", self.n_seeded)
        add_result(sim, "n_people_seeded", self.n_seeded * sim.rescale_vec)


class ScaledImports(cv.Intervention):
    """
    Seed fixed number of people per day

    Normally, n_imports corresponds to a number of agents seeded per day. Therefore, the
    number of infections seeded in the total population remains constant over time regardless
    of rescaling, while the rate of incursions in the modelled agents increases over time, as
    dynamic rescaling expands the number of people represented by the pool of agents.

    In contrast, this intervention seeds a fixed absolute number of people within the modelled agents.
    Therefore, it starts out as 1:1 but then the rate of imports decreases as the simulation scale increases.
    This corresponds to the modelled agents representing a non-random sample of the larger population initially.
    For example, seeding 1 infection/day but choosing that to occur only within the initial 50,000 people.

    So for example, we have by default

    sim['n_imports']: # of people/day/pop_size

    whereas ScaledImports has

    ScaledImports.n_imports : # of people/day

    """

    def __init__(self, n_imports: float, susceptibility_weighting=False):
        """
        Args:
            n_imports: Mean number of people infected per day

        """

        super().__init__()
        self.n_imports = n_imports  # Number of imported people per day
        self.n_imported_agents = None  # Number of agents actually seeded this timestep
        self.susceptibility_weighting = susceptibility_weighting

    def initialize(self, sim):
        super().initialize(sim)
        self.n_imported_agents = np.zeros_like(sim.tvec)

    def apply(self, sim):
        n_agents = cvu.poisson(self.n_imports / sim.rescale_vec[sim.t])  # Imported cases

        if n_agents > 0:
            # Choose from people who are susceptible
            weights = np.zeros(len(sim.people))
            weights[sim.people.susceptible] = 1
            if self.susceptibility_weighting:
                weights *= sim.people.rel_sus

            importation_inds = cvu.choose_w(weights, n_agents, unique=True)

            # It's possible that for some reason the import fails because the agent is uninfectable
            # Hence the number of people returned by `People.infect` could potentially differ to len(importation_inds)
            self.n_imported_agents[sim.t] = sim.people.infect(inds=importation_inds, layer="importation")


class DynamicTrigger(cv.Intervention):
    """
    Execute callback during simulation execution
    """

    def __init__(self, condition, action, once_only=False):
        """
        Args:
            condition: A function `condition(sim)` function that returns True or False
            action: A function `action(sim)` that runs if the condition was true
            once_only: If True, the action will only execute once
        """
        super().__init__()
        self.condition = condition  #: Function that
        self.action = action
        self.once_only = once_only
        self._ran = False

    def apply(self, sim):
        """
        Check condition and execute callback
        """
        if not (self.once_only and self._ran) and self.condition(sim):
            self.action(sim)
            self._ran = True


class test_prob_quarantine(cv.Intervention):
    """
    Probability-based testing with quarantine

    This variation of `test_prob` adds quarantine while testing, with a configurable compliance
    (which affects whether people enter quarantine or not)
    """

    def __init__(self, symp_prob, asymp_prob, symp_quar_prob, asymp_quar_prob, sensitivity, quarantine_compliance, *args, test_delay_mean=None, vac_symp_prob=np.nan, exclude=None, test_delay=None, **kwargs):
        """
        Args:
            symp_prob:
            asymp_prob:
            symp_quar_prob:
            asymp_quar_prob:
            sensitivity:
            test_delay_mean: Mean test delay - note that the minimum test delay is 1, so test_delay_mean=0 will result in all tests taking exactly 1 day to be returned
            test_delay: If specified, don't sample from the test delay
            quarantine_compliance:
            *args:
            vac_symp_prob: If provided, specify an ABSOLUTE testing rate for symptomatic vaccinated people. If nan, they will test at the same rate as the unvaccinated people
            exclude: If provided, specify a list/array of indices of people that should not be tested via this intervention
            **kwargs:
        """

        super().__init__(*args, **kwargs)

        assert (test_delay_mean is None) != (test_delay is None), 'Either the mean test delay or the absolute test delay must be specified'

        self.symp_prob = symp_prob
        self.asymp_prob = asymp_prob
        self.symp_quar_prob = symp_quar_prob
        self.asymp_quar_prob = asymp_quar_prob
        self.sensitivity = sensitivity
        self.test_delay_mean = test_delay_mean
        self.test_delay = test_delay
        self.quarantine_compliance = quarantine_compliance  #: Compliance level for individuals in general population isolating after testing. People already in quarantine are assumed to be compliant
        self.vac_symp_prob = vac_symp_prob

        self.n_tests = None
        self.n_positive = None  # Record how many tests were performed that will come back positive
        self.exclude = exclude  # Exclude certain people - mainly to cater for simulations where the index case/incursion should not be diagnosed

        self._scheduled_tests = defaultdict(list)
        self.recorded_tests = defaultdict(set)  # Record all tests every day or diagnostic purposes (Covasim itself only keeps the latest test)

    def initialize(self, sim):
        super().initialize(sim)
        self.n_tests = np.zeros(sim.npts)
        self.n_positive = np.zeros(sim.npts)

    def schedule_test(self, sim, inds, t: int):
        """
        Schedule a test in the future

        If the test is requested today, then test immediately. This is because testing should be run prior to quarantine so that
        quarantine can take place on the day of diagnosis even if the test_delay is 0.

        :param inds: Iterable with person indices to test
        :param t: Simulation day on which to test them
        :return:
        """

        if t == sim.t:
            # If a person is scheduled to test on the same day (e.g., if they are a household contact and get tested on
            # the same day they are notified)
            inds = cvu.ifalsei(sim.people.diagnosed | sim.people.dead, inds)  # Only test people that haven't been diagnosed and are alive
            self._test(sim, inds)
        else:
            self._scheduled_tests[t] += inds.tolist()

    def _test(self, sim, test_inds):
        # After testing (via self.apply or self.schedule_test) perform some post-testing tasks
        # test_inds are the indices of the people that were requested to be tested (i.e. that were
        # passed into sim.people.test, so a test was performed on them
        #
        # CAUTION - this method gets called via both apply() and schedule_test(), therefore it can be
        # called multiple times per timestep, quantities must be incremented rather than overwritten

        sim.people.test(test_inds, test_sensitivity=self.sensitivity, loss_prob=0, test_delay=np.inf)  # Actually test people

        if self.test_delay is not None:
            delays = self.test_delay*np.ones_like(test_inds)
        else:
            delays = np.maximum(1, cvu.n_poisson(self.test_delay_mean, len(test_inds)))

        # Update the date diagnosed
        positive_today = cvu.true(sim.people.date_pos_test[test_inds] == sim.t)

        sim.people.date_diagnosed[test_inds[positive_today]] = sim.t + delays[positive_today]

        # Quarantine while waiting
        if self.quarantine_compliance and len(test_inds):
            # If people are meant to quarantine while waiting for their test, then quarantine some/all of the people waiting for tests
            if self.quarantine_compliance == 1:
                # If fully compliant, keep all indices straight away
                quar_inds = test_inds
                quar_delay = delays
            else:
                # Otherwise, filter by quarantine compliance
                to_quarantine = cvu.n_binomial(self.quarantine_compliance, len(test_inds))  # Boolean array of test_inds to quarantine
                quar_inds = test_inds[to_quarantine]
                quar_delay = delays[to_quarantine]  # Array of associated delays

            # Then iterate over delays, and schedule quarantine for each delay
            for delay in set(quar_delay):
                match_delay = quar_delay == delay  # Indices of quar with people with the specified delay
                sim.people.schedule_quarantine(quar_inds[match_delay], period=delay)

        # Logging
        self.n_positive[sim.t] = len(positive_today)  # Record how many people were tested by this program today, that ended up testing positive

        # For the purpose of counting tests, people in quarantine only count as one person (notwithstanding rescaling)
        # Otherwise, scale up by the pop_scale. Since we model general tests being distributed throughout the population, we assume
        # symptomatic and asymptomatic tests both get applied to people outside of the model, and need to account for the population scale
        # accordingly. However, we also assume that the entire epidemic is contained within the pool of agents, therefore
        tests_in_quarantine = sim.people.quarantined[test_inds].sum()
        tests_not_in_quarantine = len(test_inds) - tests_in_quarantine

        # Store tests performed by this intervention
        n_tests = tests_in_quarantine + tests_not_in_quarantine * sim["pop_scale"] / sim.rescale_vec[sim.t]
        self.n_tests[sim.t] += n_tests  # Store tests performed by this intervention
        sim.results["new_tests"][sim.t] += n_tests  # Update total test count

        # Extra diagnostic output
        # self.recorded_tests[sim.t].update(test_inds)

    def apply(self, sim):
        # First, insert any fixed test probabilities
        vals = np.full(len(sim.people), self.asymp_prob)  # Baseline test probability is asymp_prob
        vals[sim.people.symptomatic] = self.symp_prob  # Symptomatic people test at a higher rate
        vals[sim.people.symptomatic & sim.people.vaccinated] = self.vac_symp_prob  # Symptomatic and vaccinated people test at a different (usually lower) rate
        vals[sim.people.symptomatic & sim.people.quarantined] = self.symp_quar_prob  # Symptomatic and quarantined people test at a different rate - note that this takes priority over vaccinated
        vals[~sim.people.symptomatic & sim.people.quarantined] = self.asymp_quar_prob
        vals[sim.people.severe] = 1  # Severe people guaranteed to test
        if self.exclude is not None:
            vals[self.exclude] = 0  # If someone is excluded, then they shouldn't test via `apply()` (but can still test via a scheduled test)
        vals[self._scheduled_tests[sim.t]] = 1  # People scheduled to test (e.g. via contact tracing) are guaranteed to test
        vals[sim.people.diagnosed] = 0  # People already diagnosed don't test again
        vals[sim.people.dead] = 0  # Dead people don't get tested

        test_inds = cvu.true(cvu.binomial_arr(vals))  # Finally, calculate who actually tests

        self._test(sim, test_inds)


class test_prob_screening(test_prob_quarantine):
    """Fixed probability screening test for pre-defined people"""

    def __init__(self, inds, test_prob, sensitivity, test_delay_mean=None, test_delay=None, *args, **kwargs):
        cv.Intervention.__init__(self, *args, **kwargs)
        # Don't want to call the test_prob_quarantine constructor because we aren't using other variables like symp_prob
        self.test_prob = test_prob
        self.inds = inds  # Can provide a callable to set the people being tested dynamically
        self.sensitivity = sensitivity
        assert (test_delay_mean is None) != (test_delay is None), 'Either the mean test delay or the absolute test delay must be specified'
        self.test_delay_mean = test_delay_mean
        self.test_delay = test_delay
        self.quarantine_compliance = 0  # Screening tests don't have any quarantine associated with them

    def apply(self, sim):
        if self.test_prob:

            if callable(self.inds):
                inds = self.inds(sim)
            else:
                inds = self.inds

            if len(self.inds) == 0:
                return

            inds = cvu.ifalsei(sim.people.diagnosed | sim.people.quarantined | sim.people.dead, inds)  # Only test people that haven't been diagnosed already
            inds = cvu.binomial_filter(self.test_prob, inds)  # Finally, calculate who actually tests
            self._test(sim, inds)


class SecondRingTracing(cv.contact_tracing):
    """
    Class that implements Victorian second-ring quarantine tracing

    This intervention quarantines contacts-of-contacts in households and social layers only.
    App-based tracing is not included. Individuals are assumed to fully comply with
    quarantine orders, if they have been reached. The probability of tracing secondary contacts
    in the social layer is assumed to be the same as for primary contacts.


    """

    def __init__(self, testing_intervention, second_ring_layers, capacity_levels, capacity_fraction, unlimited_capacity_layers, *args, **kwargs):
        """

        :param capacity:
        :param second_ring_layers: Layers to use for second ring tracing e.g. ['H','aged_care'] - set to an empty list if not using second ring
        :param testing_intervention: The testing intervention to use for scheduled tests in quarantine. Must support test scheduling
        :param args:
        :param kwargs:
        """

        super().__init__(*args, presumptive=False, **kwargs)  # Ensure presumptive quarantine is not used, otherwise
        self.second_ring_layers = second_ring_layers  #: List of layers to perform second ring tracing on e.g. ['H','social']
        self.testing_intervention = testing_intervention  # Must support test scheduling
        self.capacity_levels = capacity_levels  # list of values for step function in contact tracing capacity
        self.capacity_fraction = capacity_fraction  # list of contact tracing efficacy for different contact tracing capacity
        self.unlimited_capacity_layers = unlimited_capacity_layers  # list of layers that have no capacity constraint (set to [] if not in use)

        # Ensure that an assumption used in the tracing algorithm is met
        for lkey, trace_time in self.trace_time.items():
            if trace_time == 0 and not lkey == "H":
                raise Exception("Second ring tracing currently assumes H is the only layer with 0 trace time")

    def _find_contacts(self, contacts, sim: cv.Sim, to_trace: dict, layers: list) -> None:
        """

        :param contacts: Contacts dictionary to update in-place
        :param sim:
        :param to_trace: Dictionary of people to trace, {delay: {inds}}
        :param layers: List of layers to include
        :return:
        """

        if not layers:
            return

        for trace_time_offset, trace_ind_set in to_trace.items():

            trace_inds = np.fromiter(trace_ind_set, dtype=np.int64)

            # Trace the infection log
            # Infection log tracing does not take into account quarantine status, only trace probability
            # This is because we assume that if someone did have contacts during quarantine prior to being
            # diagnosed, they would likely report those contacts.
            infections = itertools.chain(sim.people.infection_log.in_edges(trace_inds, data="layer"), sim.people.infection_log.out_edges(trace_inds, data="layer"))
            for source, target, layer in infections:
                if layer in layers and np.random.random_sample() < self.trace_probs[layer]:
                    contacts[trace_time_offset + self.trace_time[layer]].add(source)
                    contacts[trace_time_offset + self.trace_time[layer]].add(target)

            # Now, separate out people in quarantine and not in quarantine
            in_quarantine = sim.people.quarantined[trace_inds]

            # Extract the indices of the people who'll be contacted
            for lkey, trace_prob in self.trace_probs.items():

                if lkey not in layers:
                    continue

                if trace_prob == 0:
                    continue

                contacts_to_quarantine = set()  # People in this layer to quarantine

                # Contacts of people where the person being traced is NOT already in quarantine
                # These people are notified at the full trace_prob
                contacts_not_in_quarantine = sim.people.contacts[lkey].find_contacts(trace_inds[~in_quarantine])
                if len(contacts_not_in_quarantine):
                    contacts_to_quarantine.update(cvu.binomial_filter(trace_prob, contacts_not_in_quarantine))  # Filter the indices according to the probability of being able to trace this layer

                # Contact of people where the person being traced was in quarantine
                # Since people were supposed to be in quarantine, most of the interactions in the layer would not have
                # taken place. In fact, only a 'quar_factor' portion of the interactions would have occurred. Thus, we scale
                # the probability of notification accordingly e.g. if a person had only 10% of their usual work contacts, only
                # 10% as many of their work contacts would be notified if they get diagnosed positive while in quarantine. There is
                # an edge case where a person gets diagnosed right after being quarantined, but we ignore this as an approximation. The
                # main thing is that in the case where they actually transmitted the virus, we still pick that up via the infection log
                contacts_in_quarantine = sim.people.contacts[lkey].find_contacts(trace_inds[in_quarantine])
                if len(contacts_in_quarantine):
                    contacts_to_quarantine.update(cvu.binomial_filter(trace_prob * sim["quar_factor"][lkey], contacts_in_quarantine))  # Filter the indices according to the probability of being able to trace this layer

                contacts[trace_time_offset + self.trace_time[lkey]].update(contacts_to_quarantine)

    def identify_contacts(self, sim: cv.Sim, trace_inds: np.ndarray) -> dict:
        """
        Return contacts to notify by trace time

        Note that the base class uses the `capacity` attribute but this class completely
        replaces the base method and interprets capacity in its own way (because households
        are excluded from the capacity constraint)

        Args:
            sim:
            trace_inds: Indices of people to trace (received positive diagnosis on this timestep)

        Returns: {trace_time: inds}

        """

        # TRACING WORKFLOW
        #
        # 1. For all people being traced, quarantine all layers with infinite tracing capacity (e.g. households, schools) assuming they are notified outside of DoH
        # Some of these layers may still have imperfect trace times and capacity to identify people but they are assumed to have unlimited capacity
        # 2. Then, select a subset of people being traced to account for tracing capacity
        # 3. For those people, quarantine infection log contacts based on the layer trace_prob
        # 4. For those NOT in quarantine, notify layer contacts based on trace_prob
        # 5. For those ALREADY in quarantine, notify layer contacts based on trace_prob*quar_factor

        contacts = defaultdict(set)
        trace_ind_set = set(trace_inds)

        if not trace_ind_set:
            return contacts

        # 1. NOTIFY ALL CONTACTS WITH UNLIMITED CAPACITY (e.g. HOUSE, SCHOOL)
        self._find_contacts(contacts, sim, to_trace={0: trace_inds}, layers=self.unlimited_capacity_layers)

        # 2. ACCOUNT FOR TRACING CAPACITY
        trace_inds = cv.binomial_filter(self.get_trace_capacity(sim, len(trace_inds)), trace_inds)

        # For the first ring of tracing, we have IDENTIFIED all of the people to trace on day 0
        # Therefore, we pass in all of the trace_inds with 0 trace time
        limited_capacity_layers = [layer for layer in self.trace_probs.keys() if layer not in self.unlimited_capacity_layers]
        self._find_contacts(contacts, sim, to_trace={0: trace_inds}, layers=limited_capacity_layers)

        # For the second ring, we pass in all of the people that are being contacted as
        # people to trace. Now, the day they get contacted serves as the first delay
        self._find_contacts(contacts, sim, to_trace=sc.dcp(contacts), layers=self.second_ring_layers)

        array_contacts = {}
        for trace_time, inds in contacts.items():
            inds = inds.difference(trace_ind_set)
            array_contacts[int(trace_time)] = np.fromiter(inds, dtype=np.int64)  # NB the order doesn't matter here because this gets used in a vector operation

        return array_contacts

    def notify_contacts(self, sim, contacts):

        super().notify_contacts(sim, contacts)

        if self.testing_intervention:
            for trace_time, contact_inds in contacts.items():
                self.testing_intervention.schedule_test(sim, contact_inds, sim.t + trace_time)  # Schedule tests when entering quarantine
                self.testing_intervention.schedule_test(sim, contact_inds, sim.t + self.quar_period - 3)  # Schedule a test 3 days before they're meant to leave quarantine (typically day 11)

        n_notified = 0
        for contact_inds in contacts.values():
            n_notified += len(contact_inds)
        self.new_notifications[sim.t] = n_notified

    def initialize(self, sim):
        super().initialize(sim)
        self.new_notifications = np.zeros(sim.npts)

    def finalize(self, sim):
        super().finalize(sim)
        add_result(sim, "new_notifications", self.new_notifications * sim.rescale_vec)
        add_result(sim, "cum_notifications", np.cumsum(sim.results["new_notifications"].values))

    def get_trace_capacity(self, sim, n_agents: float) -> float:
        """
        Calculate fraction of cases to trace

        For example, if there are 500 new diagnoses today, and the capacity
        specifies that at 400 cases, 50% of contacts can be traced, and at 600 cases
        then 40% of cases can be traced, this method returns 0.45.

        Args:
            sim: A cv.Sim instance
            n_agents: Current demand for tracing (number of newly diagnosed agents)

        Returns: Fraction of the demand that can be met


        """
        n_people = n_agents * sim.rescale_vec[sim.t]
        trace_eff = interp1d(self.capacity_levels, self.capacity_fraction, kind="linear")  # Cap is a function that gives the efficacy for a given number of cases
        return trace_eff(n_people)[()]  # Proportion of newly diagnosed people that can be traced in model population


def rolling_average_diagnoses(sim, window: int = 7) -> float:
    """
    Retrieve rolling average diagnoses for the current time

    The calculation assumes zero cases for times prior to the simulation start

    Args:
        window Rolling average window length

    Returns: Rolling average for desired window length

    """

    t_start = max(0, sim.t - window)
    new_diagnoses = sim.results["new_diagnoses"][t_start : sim.t] * sim.rescale_vec[t_start : sim.t]
    return np.sum(new_diagnoses) / window


def rolling_average_trigger(sim, window, value):
    """
    Rolling average trigger

    Fires if the rolling average of new_diagnoses exceeds a set amount.
    Returns `False` if the simulation has not yet run for the window duration.
    The rationale for this is that in the outbreak scenarios, there would
    have been zero cases for the previous week, therefore the rolling average
    can be computed assuming zero cases prior to the

    Note that diagnoses are recorded at the end of each day, thus the trigger fires
    the day after the diagnoses took place.

    Args:
        window Rolling average window length
        value: Rolling average trigger value

    Returns: True if the rolling average exceeds value at the current timestep

    """

    return rolling_average_diagnoses(sim, window) >= (value - 1e-6)  # Subtract 1e-6 to allow tolerance in case of numerical precision issues


class EventSchedule(cv.Intervention):
    """
    Run functions on different days

    iv = EventSchedule()
    iv[1] = lambda sim: print(sim.t)
    iv['2020-04-02'] = lambda sim: print('foo')

    """

    def __init__(self):
        super().__init__()
        self.schedule = dict()

    def __getitem__(self, day):
        return self.schedule[day]

    def __setitem__(self, day, fcn):
        if day in self.schedule:
            raise Exception("Use a list instead to assign multiple functions - or to really overwrite, delete the function for this day first i.e. `del schedule[day]` before performing `schedule[day]=...`")
        self.schedule[day] = fcn

    def __delitem__(self, key):
        del self.schedule[key]

    def initialize(self, sim):
        super().initialize(sim)
        for k, v in list(self.schedule.items()):
            day = sim.day(k)
            if day != k:
                if day in self.schedule:
                    raise Exception(f"Schedule date {k} maps to day {day} which is already present")
                self.schedule[day] = v
                del self.schedule[k]

    def apply(self, sim):
        if sim.t in self.schedule:
            if isinstance(self.schedule[sim.t], list):
                for fcn in self.schedule[sim.t]:
                    fcn(sim)
            else:
                self.schedule[sim.t](sim)
