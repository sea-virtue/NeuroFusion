"""Generate logical control ticks; this does not wait or guarantee wall-clock Hz."""
from ..contracts import Kind, Message, Stamp


def control_ticks(start_ns: int, stop_ns: int, hz: int):
    if type(hz) is not int or hz <= 0 or hz > 1_000_000_000:
        raise ValueError('hz must be an integer in [1, 1e9]')
    if start_ns < 0 or stop_ns < start_ns:
        raise ValueError('Invalid tick interval')
    index = 0
    while (time_ns := start_ns + index * 1_000_000_000 // hz) <= stop_ns:
        yield Message(Kind.CONTROL, Stamp(time_ns))
        index += 1
