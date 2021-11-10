# Policies need to be atomic, so that we can mix and match them and go arbitrarily
# from one policy setting to another

# Policy dimensions
# from collections import defaultdict
import pandas as pd
import sciris as sc
import covasim_aus_schools as cvv
from covasim_aus_schools import logger


class Policy:
    """

    Example policies:
    >>> Policy('School', 'Year 12 only', school_years_open=[17])
    >>> Policy('Workplaces (non-retail)', 'Work from home', rel_staff_beta={'non_retail_work': 0.3})
    >>> Policy('Cafes and restaurants', '4sqm', venue_capacity={'non_retail_work': 0.3})


    """

    def __init__(self, setting, level, **kwargs):
        self.setting = setting  #: e.g., 'Masks'. Must match with rows in the package CSV
        self.level = level  #: e.g., 'Indoor only', 'Outdoor in certain settings' etc. Should match with cells in the package CSV
        self.effects = kwargs

    def __repr__(self):
        return f"<Policy({self.setting}, {self.level})>"


class Restrictions:
    # These values are a subset of the supported effects, and are accumulated across all policies outside of the layer
    # This way the values can be set once per layer. Notably,

    multiplicative_attributes = [
        "venue_capacity",  # public layers only, multiplicative effect on cvv.PublicFacingLayer.venue_capacity
        "rel_staff_beta",  # public layers only, multiplicative effect on cvv.PublicFacingLayer.rel_staff_beta
        "rel_public_beta",  # public layers only, multiplicative effect on cvv.PublicFacingLayer.rel_staff_beta
        "rel_public_staff_beta",  # public layers only, multiplicative effect on cvv.PublicFacingLayer.rel_staff_beta
        "prop_open",  # cluster and public layers only, multiplicative effect on Layer.prop_open
        "attendance",  # cluster and public layers only layers, only multiplicative effect on Layer.attendance (or Layer._staff_layer.attendance)
    ]

    special_attributes = [
        "rel_beta",  # all layers, acts on sim['beta_layer']
        "school_years_open",  # School layers only, exclusive effect on cvv.SchoolLayer.set_open_years() (only one policy with this attribute will be used)
    ]

    supported_effects = multiplicative_attributes + special_attributes

    def __init__(self, baseline_beta_layer: dict, policies: list, packages: dict):
        """

        Args:
            policies: List of policies
            package_csv:
        """

        self.baseline_beta_layer = sc.dcp(baseline_beta_layer)

        policies = {(policy.setting, policy.level): policy for policy in policies}
        self.packages = {}
        for package in packages.columns:
            self.packages[package] = []
            for setting, level in packages[package].to_dict().items():
                self.packages[package].append(policies[setting, level])

    def apply_package(self, sim, package_name: str):

        # Apply a particular restriction level to the simulation
        logger.debug(f'Day {sim.t}: Setting restrictions to "{package_name}"')

        sim._restrictions = package_name
        layers = sim.people.contacts

        # First, reset all of the key attributes
        sim["beta_layer"].update(self.baseline_beta_layer)

        attribute_values = {}  # e.g., {prop_open:{'retail':0.5.'non-retail':0.8}}
        for attribute in self.multiplicative_attributes:
            attribute_values[attribute] = {layer_name: 1 for layer_name, layer in layers.items() if hasattr(layer, attribute)}

        # Then go through the policies and accumulate values for beta_layer and multiplicative attributes
        for policy in self.packages[package_name]:
            for attribute, vals in policy.effects.items():
                if attribute == "rel_beta":
                    for layer_name, rel_beta in vals.items():
                        sim["beta_layer"][layer_name] *= rel_beta
                elif attribute in attribute_values:
                    for layer_name, relative_val in vals.items():
                        if layer_name in attribute_values[attribute]:
                            attribute_values[attribute][layer_name] *= relative_val

        # Apply the venue capacity, staff beta, and prop open
        for layer_name, layer in layers.items():
            for attribute in attribute_values:
                if layer_name in attribute_values[attribute]:
                    setattr(layer, attribute, attribute_values[attribute][layer_name])

        # Handle school years
        for policy in self.packages[package_name]:
            if "school_years_open" in policy.effects:
                layers["primary_school"].set_open_ages(policy.effects["school_years_open"])
                layers["high_school"].set_open_ages(policy.effects["school_years_open"])
                break
        else:
            layers["primary_school"].set_open_ages()
            layers["high_school"].set_open_ages()

        # Update all layers
        sim.people.update_contacts()


class RestrictionSchedule(cvv.EventSchedule):
    def __init__(self, restrictions):
        super().__init__()
        self.restrictions = restrictions

    def __setitem__(self, day, package: str):
        assert sc.isstring(package)
        self.schedule[day] = package

    def apply(self, sim):
        if sim.t in self.schedule:
            self.restrictions.apply_package(sim, self.schedule[sim.t])

    def clear(self):
        # Reset schedule
        self.schedule = dict()
