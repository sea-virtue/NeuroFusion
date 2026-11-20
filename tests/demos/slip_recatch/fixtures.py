"""Synthetic controller fixtures; no hardware access."""
from dataclasses import asdict
from neurofusion.demos.slip_recatch.slip import SlipPrediction, GripperFeedback


def demo_rows(config):
    """Deterministic test fixture, NOT a friction/actuator simulation or BC data."""
    closure = config.grip_closure
    for i in range(401):
        now = 1_000_000_000+i*10_000_000
        onset = 50 <= i < 110
        # Fixture eventually returns to the grip position after timed braking.
        if i < 50: closure = max(config.grip_closure-.002,closure-.00005)
        elif i < 105: closure = config.grip_closure-.002
        else: closure = min(config.grip_closure,closure+.0003)
        prediction = SlipPrediction(i,now,now,'slide' if onset else 'quiet',config.expected_direction,.99,.99)
        yield {'now_ns':now,'feedback':asdict(GripperFeedback(now,closure)),'snn':asdict(prediction)}
