"""Convert wall-clock time to fixed-step playback budget."""


def playback_budget(accumulator, elapsed, old_speed, new_speed, base_rate):
    if new_speed != old_speed:
        # Pending work belongs to the old speed, not the newly selected rate.
        return 0.0
    return accumulator + elapsed * base_rate * new_speed
