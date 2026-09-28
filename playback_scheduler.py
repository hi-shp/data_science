"""Wall-clock budget only; never mutates physical state or simulation time."""


def playback_budget(accumulator, elapsed, old_speed, new_speed, base_rate):
    if new_speed != old_speed:
        # Old requested work is not completed simulation time. Discard that
        # catch-up debt, and do not charge the preceding interval to new rate.
        return 0.0
    return accumulator + elapsed*base_rate*new_speed
