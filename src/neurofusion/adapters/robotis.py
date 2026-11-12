"""ROBOTIS RH-P12-RN/RN(A), using the official DYNAMIXEL SDK.

Mode 5 accepts absolute closure and a current ceiling. Mode 0 accepts signed
motor current, not closure, calibrated contact force, or MIT impedance commands.
Read-only by default. Motion is an explicit bounded commissioning session.
"""
from dataclasses import dataclass, replace
from decimal import Decimal
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


OPERATING_MODES = {'position': 5, 'current': 0}


def _raw_magnitude(value, unit):
    # Device units are decimal; avoid .57/.114 becoming 4.999... then 4 raw.
    return int(Decimal(str(abs(value)))/Decimal(str(unit)))


def decode_rn_state(data):
    """Decode one SDK read of legacy RN registers 596..625, including gaps."""
    if len(data) != 30:
        raise ValueError('Expected exact RN register block 596..625')
    data = bytes(data)
    def integer(offset, size, signed=False):
        return int.from_bytes(data[offset:offset+size], 'little', signed=signed)
    return dict(goal_raw=integer(0,4,True), profile_velocity_raw=integer(4,4,True),
        goal_current_raw=integer(8,2,True), acceleration_raw=integer(10,4),
        moving=bool(integer(14,1)), position_raw=integer(15,4,True),
        velocity_raw=integer(19,4,True), current_raw=integer(25,2,True),
        current_a=integer(25,2,True)*TABLES[35073].current_unit_a,
        voltage_v=integer(27,2)*.1, temperature_c=integer(29,1))


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
    control_mode: str = 'position'
    # Mode 5: resolve the ceiling from the device's existing Current Limit.
    # Mode 0 retains the explicit, bounded signed-current commissioning cap.
    use_hardware_current_limit: bool = True

    def __post_init__(self):
        if self.control_mode not in OPERATING_MODES:
            raise ValueError('control_mode must be position or current')
        if not self.port or type(self.device_id) is not int or not 0 <= self.device_id <= 252:
            raise ValueError('An explicit serial port and unicast device ID are required')
        for key in ('baud','max_step_raw','session_radius_raw','tolerance_raw'):
            if type(getattr(self,key)) is not int or getattr(self,key) <= 0:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('current_cap_a','velocity_rpm','timeout_s'):
            if not math.isfinite(getattr(self,key)) or getattr(self,key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if type(self.use_hardware_current_limit) is not bool:
            raise ValueError('use_hardware_current_limit must be boolean')
        if (self.current_cap_a > .3 and (self.control_mode != 'position' or
                not self.use_hardware_current_limit)) or self.velocity_rpm > 1.:
            raise ValueError('Explicit commissioning cap supports at most .3A and 1rpm; position mode may use the device current limit')


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
    operating_mode: int = 5
    goal_current_a: float = 0.

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


@dataclass(frozen=True)
class GripperCurrentReceipt:
    sent_ns: int
    acknowledged_ns: int
    target_raw: int
    target_current_a: float
    # Current ACK is not position arrival or measured contact force.


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
            if self.config.control_mode == 'position' and self.config.use_hardware_current_limit:
                # Keep existing consumers of config.current_cap_a consistent with
                # the ceiling actually written; never change EEPROM Current Limit.
                cap = self.limits['current_raw']*t.current_unit_a
                if cap <= 0:
                    raise RuntimeError('Invalid hardware current limit')
                self.config = replace(self.config,current_cap_a=cap)
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

    def _read_bytes(self, address, size):
        data,result,error = self._packet.readTxRx(
            self._port,self.config.device_id,address,size)
        self._check(result,error)
        if len(data) != size:
            raise RuntimeError(f'Short register read at {address}: {len(data)} of {size} bytes')
        return bytes(data)

    def _write_bytes(self, address, data):
        """Use SDK encoding/CRC/byte stuffing, with a real unicast ACK."""
        if not self.config.allow_motion:
            raise RuntimeError('Read-only session')
        data = bytes(data)
        attempt = dict(sent_ns=time.monotonic_ns(),address=address,
                       size=len(data),data=list(data))
        self.writes.append(attempt)
        result,error = self._packet.writeTxRx(
            self._port,self.config.device_id,address,len(data),list(data))
        attempt.update(acknowledged_ns=time.monotonic_ns(),result=result,error=error)
        self._check(result,error)

    def _require_mode(self, mode):
        if self.config.control_mode != mode or self.limits['mode'] != OPERATING_MODES[mode]:
            raise RuntimeError(f'Requires {mode} control; position and current actions are distinct')

    def set_mode(self, mode):
        """Explicit EEPROM change only; never called automatically by arm()."""
        if mode not in OPERATING_MODES:
            raise ValueError('Mode must be position or current')
        reason = motion_inhibit_reason()
        if reason:
            raise RuntimeError(f'Physical motion inhibited: {reason}')
        if not self.config.allow_motion or self._motion_started:
            raise RuntimeError('Mode changes require an explicit, inactive motion session')
        if self._read(self.table.torque,1):
            raise RuntimeError('Disable torque before changing operating mode')
        value = OPERATING_MODES[mode]
        if self._read(11,1) != value:
            self._write(11,1,value)
        self.limits['mode'] = self._read(11,1)
        if self.limits['mode'] != value:
            raise RuntimeError('Operating mode readback mismatch')

    def _write_position_profile(self, position, current_raw, velocity_raw):
        """RN 596/600/604 form one 10-byte WRITE; RN(A) has another layout."""
        self._require_mode('position')
        if not self.limits['min_raw'] <= position <= self.limits['max_raw']:
            raise ValueError('Position outside hardware limits')
        if not 0 < current_raw <= self.limits['current_raw']:
            raise ValueError('Position control requires a positive current ceiling')
        if not 0 < velocity_raw <= self.limits['velocity_raw']:
            raise ValueError('Position control requires a positive bounded profile velocity')
        t = self.table
        if self.model_number == 35073:
            data = (position.to_bytes(4,'little',signed=True)
                    + velocity_raw.to_bytes(4,'little')
                    + current_raw.to_bytes(2,'little',signed=True))
            self._write_bytes(t.goal_position,data)
        else:
            self._write(t.goal_current,2,current_raw)
            self._write(t.profile_velocity,4,velocity_raw)
            self._write(t.goal_position,4,position)

    def closure(self, raw):
        return (raw-self.limits['min_raw'])/(self.limits['max_raw']-self.limits['min_raw'])

    def raw_position(self, closure):
        if not math.isfinite(closure) or not 0 <= closure <= 1:
            raise ValueError('Physical gripper action is absolute closure in [0,1]')
        return round(self.limits['min_raw']+closure*(self.limits['max_raw']-self.limits['min_raw']))

    def read(self):
        t = self.table
        if self.model_number == 35073:
            raw = decode_rn_state(self._read_bytes(t.goal_position,30))
            values = {key:raw[key] for key in (
                'position_raw','velocity_raw','current_a','voltage_v','temperature_c','moving')}
            values.update(closure=self.closure(raw['position_raw']),
                goal_position_raw=raw['goal_raw'],
                goal_current_a=raw['goal_current_raw']*t.current_unit_a)
        else:
            values = self._read_individual_state()
        values.update(torque_enabled=bool(self._read(t.torque,1)),
                      hardware_error=self._read(t.hardware_error,1),
                      operating_mode=self.limits['mode'])
        return GripperSample(host_ns=time.monotonic_ns(),**values)

    def _read_individual_state(self):
        """Keep RN(A) addressing separate from the legacy RN block."""
        t = self.table
        position = self._read(t.present_position,4,True)
        return dict(position_raw=position,closure=self.closure(position),
            velocity_raw=self._read(t.present_velocity,4,True),
            current_a=self._read(t.present_current,2,True)*t.current_unit_a,
            voltage_v=self._read(t.voltage,2)*.1,temperature_c=self._read(t.temperature,1),
            moving=bool(self._read(t.moving,1)),
            goal_position_raw=self._read(t.goal_position,4,True),
            goal_current_a=self._read(t.goal_current,2,True)*t.current_unit_a)

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
        self.limits['mode'] = self._read(11,1)
        self._require_mode(self.config.control_mode)
        if self.limits['return_level'] != 2:
            raise RuntimeError('Requires write replies; EEPROM is not changed by arm')
        sample = self.read()
        self._healthy(sample)
        if sample.torque_enabled:
            raise RuntimeError('Another controller may be active; commissioning requires torque initially off')
        t = self.table
        self._origin = sample.position_raw
        self._saved = ({address:(size,self._read(address,size)) for address,size in
            ((t.goal_current,2),(t.profile_velocity,4),(t.profile_acceleration,4))}
            if self.config.control_mode == 'position' else {})
        current = min(self.limits['current_raw'],_raw_magnitude(self.config.current_cap_a,t.current_unit_a))
        velocity_unit = .114 if self.model_number == 35073 else .01
        velocity = min(self.limits['velocity_raw'],max(1,_raw_magnitude(self.config.velocity_rpm,velocity_unit)))
        if current <= 0 or (self.config.control_mode == 'position' and
                (velocity <= 0 or self.limits['acceleration_raw'] <= 0)):
            raise RuntimeError('Invalid current/velocity/acceleration limits')
        self._motion_started = True  # cleanup even if a WRITE reply is lost
        if self.config.control_mode == 'position':
            self._write(t.profile_acceleration,4,1)
            self._write_position_profile(sample.position_raw,current,velocity)
        else:
            # A nonzero Mode 0 preload would start driving as soon as torque is on.
            self._write(t.goal_current,2,0)
        self._write(t.torque,1,1)
        after = self.read()
        self._healthy(after)
        self._check_motion_sample(after)
        margin = (self.config.tolerance_raw if self.config.control_mode == 'position'
                  else self.config.session_radius_raw)
        if abs(after.position_raw-self._origin) > margin or not after.torque_enabled:
            raise RuntimeError('Unexpected position reset or failed torque enable')
        self._armed = True
        return after

    def command(self, proposal):
        self._require_mode('position')
        if not self._armed:
            raise RuntimeError('Motion has not been armed')
        if proposal.normalized or len(proposal.robotis) != 1:
            raise ValueError('Normalized network output cannot be sent to hardware')
        if proposal.time_ns > time.monotonic_ns() or time.monotonic_ns()-proposal.time_ns > 1_000_000_000:
            raise ValueError('Command timestamp is stale or in the future')
        target = self.raw_position(float(proposal.robotis[0]))
        sample = self.read()
        self._healthy(sample)
        self._check_motion_sample(sample)
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

    def command_current(self, current_a):
        """Set signed amperes in Mode 0; no position/force/arrival claim.

        A current target persists until replaced or torque is disabled. The caller
        must continue monitoring feedback and bound the duration of each trial.
        """
        self._require_mode('current')
        if not self._armed:
            raise RuntimeError('Motion has not been armed')
        if not math.isfinite(current_a) or abs(current_a) > self.config.current_cap_a:
            raise ValueError('Target current must be finite and within the configured ampere cap')
        unit = self.table.current_unit_a
        # Truncate magnitude so quantization never exceeds the requested limit.
        raw = _raw_magnitude(current_a,unit) * (-1 if current_a < 0 else 1)
        if current_a and raw == 0:
            raise ValueError('Nonzero target is smaller than one device current unit')
        if abs(raw) > self.limits['current_raw']:
            raise ValueError('Target current exceeds hardware Current Limit')
        sample = self.read()
        self._healthy(sample)
        self._check_motion_sample(sample)
        sent = time.monotonic_ns()
        self._write(self.table.goal_current,2,raw)
        receipt = GripperCurrentReceipt(sent,time.monotonic_ns(),raw,raw*unit)
        if self._read(self.table.goal_current,2,True) != raw:
            raise RuntimeError('Current goal register readback mismatch')
        return receipt

    def _check_motion_sample(self, sample):
        if not sample.torque_enabled or abs(sample.position_raw-self._origin) > self.config.session_radius_raw+3:
            raise RuntimeError('Torque loss or movement outside session range')
        if abs(sample.current_a) > self.config.current_cap_a+.03:
            raise RuntimeError('Measured current exceeds commissioning cap')

    def hold_current(self, current_a, seconds, on_sample=None):
        """Bounded Mode 0 trial: one target, monitored feedback, torque-off at end.

        Host polling is not a hardware watchdog or a hard real-time deadline.
        Current targets never generate a gripper_executed closure observation.
        """
        self._require_mode('current')
        if not self._armed:
            raise RuntimeError('Motion has not been armed')
        if not math.isfinite(seconds) or not 0 < seconds <= self.config.timeout_s:
            raise ValueError('Current trial duration must be positive and within timeout_s')
        try:
            receipt = self.command_current(current_a)
            deadline = time.monotonic()+seconds
            while True:
                last = self.read()
                self._healthy(last)
                self._check_motion_sample(last)
                if on_sample:
                    on_sample(last)
                if time.monotonic() >= deadline:
                    break
                time.sleep(min(.01,max(0.,deadline-time.monotonic())))
            return receipt,last
        finally:
            self.disarm()

    def wait(self, receipt, on_sample=None):
        self._require_mode('position')
        if not isinstance(receipt,GripperReceipt):
            raise TypeError('Position arrival requires a position receipt, not a current ACK')
        deadline = time.monotonic()+self.config.timeout_s
        consecutive = 0
        try:
            while time.monotonic() < deadline:
                sample = self.read()
                if on_sample:
                    on_sample(sample)
                self._healthy(sample)
                self._check_motion_sample(sample)
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
        # Even if zero-current WRITE fails, attempt torque-off in the finally block.
        zero_error = None
        try:
            if self.config.control_mode == 'current':
                try:
                    self._write(self.table.goal_current,2,0)
                except BaseException as error:
                    zero_error = error
        finally:
            self._write(self.table.torque,1,0)
        if self._read(self.table.torque,1):
            raise RuntimeError('Could not confirm torque off; check physical power switch')
        self._motion_started = False
        for address,(size,value) in self._saved.items():
            self._write(address,size,value)
        if zero_error is not None:
            raise zero_error


def confirmed_execution_observation(receipt, sample, state, tolerance_raw=3):
    """Only measured arrival enters executed feedback, never a transport ACK."""
    from ..models.policy import Observation
    if not isinstance(receipt,GripperReceipt) or sample.operating_mode != 5:
        raise ValueError('Only position-mode measured arrival is an executed closure action')
    if sample.host_ns < receipt.acknowledged_ns or sample.hardware_error or abs(sample.position_raw-receipt.target_raw) > tolerance_raw:
        raise ValueError('Execution has not been confirmed by fresh measured position')
    value = state.executed.new_tensor([[sample.closure]])
    return Observation('gripper_executed',max(time.monotonic_ns(),state.time_ns),value,
                       captured_ns=sample.host_ns,capture_time_basis='host_receipt_proxy')
