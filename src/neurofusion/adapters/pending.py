"""Integration stubs; the separate speck-lab remains the verified raw collector."""


class SpeckEventSource:
    def packets(self):
        raise NotImplementedError('Integrate speck-lab acquisition, packetization and host receipt timestamps')


class GelSightSource:
    def samples(self):
        raise NotImplementedError('GelSight capture, frame timestamps and calibration are pending')


class SceneRGBSource:
    def samples(self):
        raise NotImplementedError('Scene RGB capture is pending')


class RobotFeedback:
    def observations(self):
        raise NotImplementedError('Robot feedback and executed-action confirmation are pending')


class RobotCommandSink:
    def publish(self, action):
        raise NotImplementedError('Units, limits, stop behavior, transport and execution feedback must be implemented')
