class EventCSVReader:
    def read(self, path):
        raise NotImplementedError('M02: validate real CSV schema, times and malformed rows')


class EventReplay:
    def play(self, recording, speed=1.0):
        raise NotImplementedError('M02: event-time playback, pause and speed control')
