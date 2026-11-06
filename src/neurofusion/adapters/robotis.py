"""ROBOTIS RH-P12-RN/RN(A), using the official DYNAMIXEL SDK.

Absolute closure is a dimensionless motor-position fraction, NOT jaw width or
force. Read-only by default. Motion is an explicit bounded commissioning session.
"""
from dataclasses import dataclass
import math
import json
from pathlib import Path
import time


def motion_inhibit_reason():
    path = Path(__file__).resolve().parents[3]/'configs/gripper-motion-hold.json'
    return json.loads(path.read_text())['reason'] if path.exists() else None


@dataclass(frozen=True)
class ControlTable:
    name: str
    torque: int
    goal_position: int
    profile_velocity: int
    goal_current: int
    profile_acceleration: int
    present_position: int
    present_velocity: int
    present_current: int
    voltage: int
    temperature: int
    moving: int
    hardware_error: int
    current_limit: int
    velocity_limit: int
    acceleration_limit: int
    min_position: int
    max_position: int
    min_voltage: int
    max_voltage: int
    status_return: int
    current_unit_a: float


TABLES = {
    35073: ControlTable('RH-P12-RN', 562,596,600,604,606,611,615,621,623,625,
                        610,892,30,32,26,40,36,24,22,891,.00402),
    35074: ControlTable('RH-P12-RN(A)',512,564,560,550,556,580,576,574,592,594,
                        570,518,38,44,40,52,48,34,32,516,.001),
}


@dataclass(frozen=True)
class GripperConfig:
    port: str = '/dev/ttyUSB0'
    baud: int = 2_000_000
    device_id: int = 1
    allow_motion: bool = False
    max_step_raw: int = 20
    session_radius_raw: int = 25
    current_cap_a: float = .20
    # Different profile units: RN=.114rpm; RN(A)=.01rpm.
    velocity_rpm: float = .57
    tolerance_raw: int = 3
    timeout_s: float = 4.

    def __post_init__(self):
        if not self.port or type(self.device_id) is not int or not 0 <= self.device_id <= 252:
            raise ValueError('An explicit serial port and unicast device ID are required')
        for key in ('baud','max_step_raw','session_radius_raw','tolerance_raw'):
            if type(getattr(self,key)) is not int or getattr(self,key) <= 0:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('current_cap_a','velocity_rpm','timeout_s'):
            if not math.isfinite(getattr(self,key)) or getattr(self,key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.current_cap_a > .3 or self.velocity_rpm > 1.:
            raise ValueError('Commissioning adapter supports at most .3A and 1rpm')


@dataclass(frozen=True)
class GripperSample:
    host_ns: int
    position_raw: int
    closure: float
    velocity_raw: int
    current_a: float
    voltage_v: float
    temperature_c: int
    moving: bool
    torque_enabled: bool
    hardware_error: int
    goal_position_raw: int
    timestamp_basis: str = 'host_receipt_proxy'

    def observation(self, state, slot=15):
        """q[15] is measured closure; preserve all other robot-state slots.

        This commissioning layout must also be used by future training data.
        Never acknowledge a goal as executed merely because WRITE succeeded.
        """
        from ..models.policy import Observation
        if state.robot.shape[0] != 1 or not 0 <= slot < state.robot.shape[1]:
            raise ValueError('Live gripper feedback requires one sample and a valid q slot')
        value = state.robot.clone()
        value[0,slot] = self.closure
        return Observation('robot',max(time.monotonic_ns(),state.time_ns),value,
                           captured_ns=self.host_ns,capture_time_basis=self.timestamp_basis)


@dataclass(frozen=True)
class GripperReceipt:
    sent_ns: int
    acknowledged_ns: int
    target_raw: int
    target_closure: float
    # A WRITE acknowledgment confirms reception, not mechanical execution.


class RobotisGripper:
    def __init__(self, config=GripperConfig()):
        self.config = config
        self._port = self._packet = self.table = None
        self._armed = False
        self._motion_started = False
        self._saved = {}
        self.writes = []

    def __enter__(self):
        from dynamixel_sdk import PortHandler, PacketHandler
        class WslPort(PortHandler):
            def setPacketTimeout(self, packet_length):
                super().setPacketTimeout(packet_length)
                # Allow USB forwarding jitter; no automatic WRITE retries.
                self.packet_timeout = max(self.packet_timeout,150.)
        self._port = WslPort(self.config.port)
        try:
            if not self._port.openPort() or not self._port.setBaudRate(self.config.baud):
                raise RuntimeError('Cannot open configured U2D2 serial port')
            self._packet = PacketHandler(2.)
            model,result,error = self._packet.ping(self._port,self.config.device_id)
            self._check(result,error)
            if model not in TABLES:
                raise RuntimeError(f'Unsupported model {model}; no register access attempted')
            self.model_number, self.table = model,TABLES[model]
            self.firmware = self._read(6,1)
            t = self.table
            self.limits = {
                'mode':self._read(11,1),'min_raw':self._read(t.min_position,4,True),
                'max_raw':self._read(t.max_position,4,True),
                'current_raw':self._read(t.current_limit,2),
                'velocity_raw':self._read(t.velocity_limit,4),
                'acceleration_raw':self._read(t.acceleration_limit,4),
                'min_voltage_v':self._read(t.min_voltage,2)*.1,
                'max_voltage_v':self._read(t.max_voltage,2)*.1,
                'return_level':self._read(t.status_return,1),
            }
            if self.limits['max_raw'] <= self.limits['min_raw']:
                raise RuntimeError('Invalid hardware position limits')
            return self
        except BaseException:
            self._port.closePort()
            raise

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self._motion_started:
                self.disarm()
        finally:
            self._port.closePort()

    def _check(self, result, error):
        if result != 0:
            raise RuntimeError(self._packet.getTxRxResult(result))
        if error:
            raise RuntimeError(self._packet.getRxPacketError(error))

    def _read(self, address, size, signed=False):
        value,result,error = getattr(self._packet,f'read{size}ByteTxRx')(
            self._port,self.config.device_id,address)
        self._check(result,error)
        return value-(1 << (size*8)) if signed and value & (1 << (size*8-1)) else value

    def _write(self, address, size, value):
        if not self.config.allow_motion:
            raise RuntimeError('Read-only session')
        attempt = {'sent_ns':time.monotonic_ns(),'address':address,'size':size,'value':value}
        self.writes.append(attempt)
        result,error = getattr(self._packet,f'write{size}ByteTxRx')(
            self._port,self.config.device_id,address,value & ((1 << (8*size))-1))
        attempt.update(acknowledged_ns=time.monotonic_ns(),result=result,error=error)
        self._check(result,error)

    def closure(self, raw):
        return (raw-self.limits['min_raw'])/(self.limits['max_raw']-self.limits['min_raw'])

    def raw_position(self, closure):
        if not math.isfinite(closure) or not 0 <= closure <= 1:
            raise ValueError('Physical gripper action is absolute closure in [0,1]')
        return round(self.limits['min_raw']+closure*(self.limits['max_raw']-self.limits['min_raw']))

    def read(self):
        t = self.table
        position = self._read(t.present_position,4,True)
        values = dict(position_raw=position,closure=self.closure(position),
            velocity_raw=self._read(t.present_velocity,4,True),
            current_a=self._read(t.present_current,2,True)*t.current_unit_a,
            voltage_v=self._read(t.voltage,2)*.1,temperature_c=self._read(t.temperature,1),
            moving=bool(self._read(t.moving,1)),torque_enabled=bool(self._read(t.torque,1)),
            hardware_error=self._read(t.hardware_error,1),
            goal_position_raw=self._read(t.goal_position,4,True))
        return GripperSample(host_ns=time.monotonic_ns(),**values)

    def _healthy(self, sample):
        if sample.hardware_error or sample.temperature_c >= 60:
            raise RuntimeError('Gripper hardware error or elevated temperature')
        if not (max(15.,self.limits['min_voltage_v']) <= sample.voltage_v <= min(30.,self.limits['max_voltage_v'])):
            raise RuntimeError('Supply voltage outside commissioning bounds')
        if not 0 <= sample.closure <= 1:
            raise RuntimeError('Measured position outside hardware limits')

    def arm(self):
        reason = motion_inhibit_reason()
        if reason:
            raise RuntimeError(f'Physical motion inhibited: {reason}')
        if not self.config.allow_motion or self._motion_started:
            raise RuntimeError('Explicit, new motion session required')
        if self.limits['mode'] != 5 or self.limits['return_level'] != 2:
            raise RuntimeError('Requires existing current-based position mode and write replies; EEPROM is not changed')
        sample = self.read()
        self._healthy(sample)
        if sample.torque_enabled:
            raise RuntimeError('Another controller may be active; commissioning requires torque initially off')
        t = self.table
        self._origin = sample.position_raw
        self._saved = {address:(size,self._read(address,size)) for address,size in
            ((t.goal_current,2),(t.profile_velocity,4),(t.profile_acceleration,4))}
        current = min(self.limits['current_raw'],int(self.config.current_cap_a/t.current_unit_a))
        velocity_unit = .114 if self.model_number == 35073 else .01
        velocity = min(self.limits['velocity_raw'],max(1,int(self.config.velocity_rpm/velocity_unit)))
        if current <= 0 or velocity <= 0 or self.limits['acceleration_raw'] <= 0:
            raise RuntimeError('Invalid current/velocity/acceleration limits')
        self._motion_started = True  # cleanup even if a WRITE reply is lost
        self._write(t.goal_current,2,current)
        self._write(t.profile_velocity,4,velocity)
        self._write(t.profile_acceleration,4,1)
        self._write(t.goal_position,4,sample.position_raw)  # preload before torque
        self._write(t.torque,1,1)
        after = self.read()
        self._healthy(after)
        if abs(after.position_raw-self._origin) > self.config.tolerance_raw or not after.torque_enabled:
            raise RuntimeError('Unexpected position reset or failed torque enable')
        self._armed = True
        return after

    def command(self, proposal):
        if not self._armed:
            raise RuntimeError('Motion has not been armed')
        if proposal.normalized or len(proposal.robotis) != 1:
            raise ValueError('Normalized network output cannot be sent to hardware')
        if proposal.time_ns > time.monotonic_ns() or time.monotonic_ns()-proposal.time_ns > 1_000_000_000:
            raise ValueError('Command timestamp is stale or in the future')
        target = self.raw_position(float(proposal.robotis[0]))
        sample = self.read()
        self._healthy(sample)
        if not sample.torque_enabled:
            raise RuntimeError('Torque has been disabled')
        if abs(target-sample.position_raw) > self.config.max_step_raw:
            raise ValueError(f'Target {target} exceeds permitted movement from measured {sample.position_raw}')
        if abs(target-self._origin) > self.config.session_radius_raw:
            raise ValueError('Target exceeds commissioning session range')
        sent = time.monotonic_ns()
        self._write(self.table.goal_position,4,target)
        receipt = GripperReceipt(sent,time.monotonic_ns(),target,self.closure(target))
        if self._read(self.table.goal_position,4,True) != target:
            raise RuntimeError('Goal register readback mismatch')
        return receipt

    def wait(self, receipt, on_sample=None):
        deadline = time.monotonic()+self.config.timeout_s
        consecutive = 0
        try:
            while time.monotonic() < deadline:
                sample = self.read()
                if on_sample:
                    on_sample(sample)
                self._healthy(sample)
                if not sample.torque_enabled or abs(sample.position_raw-self._origin) > self.config.session_radius_raw+3:
                    raise RuntimeError('Torque loss or movement outside session range')
                if abs(sample.current_a) > self.config.current_cap_a+.03:
                    raise RuntimeError('Measured current exceeds commissioning cap')
                consecutive = consecutive+1 if abs(sample.position_raw-receipt.target_raw) <= self.config.tolerance_raw else 0
                if consecutive >= 3:
                    return sample
                time.sleep(.05)
            raise TimeoutError('No measured arrival before timeout; torque will be disabled')
        except BaseException:
            self.disarm()
            raise

    def disarm(self):
        self._armed = False
        self._write(self.table.torque,1,0)
        if self._read(self.table.torque,1):
            raise RuntimeError('Could not confirm torque off; check physical power switch')
        for address,(size,value) in self._saved.items():
            self._write(address,size,value)
        self._motion_started = False


def confirmed_execution_observation(receipt, sample, state, tolerance_raw=3):
    """Only measured arrival enters executed feedback, never a transport ACK."""
    from ..models.policy import Observation
    if sample.host_ns < receipt.acknowledged_ns or sample.hardware_error or abs(sample.position_raw-receipt.target_raw) > tolerance_raw:
        raise ValueError('Execution has not been confirmed by fresh measured position')
    value = state.executed.new_tensor([[sample.closure]])
    return Observation('gripper_executed',max(time.monotonic_ns(),state.time_ns),value,
                       captured_ns=sample.host_ns,capture_time_basis='host_receipt_proxy')
