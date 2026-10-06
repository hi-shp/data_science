"""Measured MAIN shape preferences, used only AFTER route-crossing safety."""
import json
from pathlib import Path
import numpy as np


def load_presentation_profile(pixels_per_m):
    """One initialization read, scale measured lengths to current GUI units."""
    source=json.loads(Path(__file__).with_name('gap_presentation_profile.json').read_text())
    return {name:{q:value*pixels_per_m for q,value in quantiles.items()}
            for name,quantiles in source['quantiles'].items()}


def profile_key(gap,distance,distance_profile,width_profile):
    """Bounded lexicographic preferences, no weighted reward or legacy score.

    Typical distance band comes first, then typical compact width band. Exact
    distance/width median differences only resolve ties within those bands.
    Verticality is handled later, inside the selected local passage only.
    """
    def deviation(value,distribution):
        spread=max(distribution['p75']-distribution['p25'],1e-9)
        outside=max(distribution['p25']-value,0.,value-distribution['p75'])/spread
        return outside,abs(value-distribution['median'])/spread
    distance_band,distance_error=deviation(distance,distance_profile)
    width_band,width_error=deviation(float(np.linalg.norm(gap['c2']-gap['c1'])),width_profile)
    return distance_band,width_band,distance_error,width_error
