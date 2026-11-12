"""Causal timer-based ruler slip teacher. Durations are NOT distances."""
from dataclasses import dataclass, asdict
import math


def positive(value, name, *, zero=False):
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')


@dataclass(frozen=True)
class SlipConfig:
    grip_closure: float
    slide_seconds: float = .5
    # Not copied from the paper. Zero means no calibrated compensation yet.
    close_delay_seconds: float = 0.
    expected_direction: str = 'down'
    confidence: float = .8
    confirm_seconds: float = .03
    max_prediction_age: float = .15
    max_feedback_age: float = .15
    max_tick_gap: float = .1
    grip_settle_seconds: float = .2
    release_timeout: float = 3.
    brake_timeout: float = 2.
    open_rate: float = .01
    close_rate: float = .03
    max_open_delta: float = .02
    max_tracking_delta: float = .01
    arrival_tolerance: float = .003

    def __post_init__(self):
        if not math.isfinite(self.grip_closure) or not 0 < self.grip_closure <= 1:
            raise ValueError('grip_closure must be a calibrated physical fraction in (0,1]')
        if self.expected_direction not in ('up','down','left','right'):
            raise ValueError('Map the physical downward axis to an SNN direction explicitly')
        if not 0 < self.confidence <= 1:
            raise ValueError('confidence must be in (0,1]')
        for name in ('slide_seconds','max_prediction_age','max_feedback_age','max_tick_gap',
                     'release_timeout','brake_timeout','open_rate','close_rate','max_open_delta',
                     'max_tracking_delta','arrival_tolerance'):
            positive(getattr(self,name),name)
        for name in ('close_delay_seconds','confirm_seconds','grip_settle_seconds'):
            positive(getattr(self,name),name,zero=True)
        if self.close_delay_seconds >= self.slide_seconds:
            raise ValueError('Compensation must be shorter than the requested sliding interval')
        if self.max_open_delta >= self.grip_closure or self.max_tracking_delta > 1:
            raise ValueError('Invalid release or tracking range')


@dataclass(frozen=True)
class SlipPrediction:
    sequence: int
    available_ns: int
    captured_ns: int
    task: str
    direction: str
    task_confidence: float
    direction_confidence: float
    model_id: str | None = None

    def __post_init__(self):
        if self.task not in ('quiet','slide','rotate','escape_wriggle'):
            raise ValueError('Unknown trained SNN task')
        if self.direction not in ('up','down','left','right'):
            raise ValueError('Unknown SNN direction')
        if any(type(v) is not int or v < 0 for v in (self.sequence,self.available_ns,self.captured_ns)):
            raise ValueError('Nonnegative integer sequence and host timestamps required')
        if self.captured_ns > self.available_ns:
            raise ValueError('Prediction cannot be available before its source')
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (self.task_confidence,self.direction_confidence)):
            raise ValueError('Invalid classification confidence')


@dataclass(frozen=True)
class GripperFeedback:
    host_ns: int
    closure: float

    def __post_init__(self):
        if type(self.host_ns) is not int or self.host_ns < 0 or not math.isfinite(self.closure) or not 0 <= self.closure <= 1:
            raise ValueError('Invalid measured gripper feedback')


@dataclass(frozen=True)
class GripperCommand:
    """Duck-compatible with RobotisGripper.command; no Franka placeholder needed."""
    time_ns: int
    closure: float
    revision: int = 0

    def __post_init__(self):
        if type(self.time_ns) is not int or self.time_ns < 0 or type(self.revision) is not int or self.revision < 0:
            raise ValueError('Command requires nonnegative integer host time and revision')
        if not math.isfinite(self.closure) or not 0 <= self.closure <= 1:
            raise ValueError('Command closure must be in [0,1]')

    @property
    def robotis(self): return (self.closure,)

    @property
    def normalized(self): return False


class SlipController:
    """One controller per episode. Caller supplies periodic monotonic ticks.

    A timer starts at the first *available* qualified SNN prediction, after
    confirmation by distinct observations. It keeps running through quiet gaps;
    quiet is not proof of stopped motion or contact. Faults never resume opening.
    """
    terminal = ('FINISHED','FAULT')

    def __init__(self, config):
        self.config = config
        self.phase = 'IDLE'
        self.reason = None
        self.last_ns = self.phase_ns = None
        self.last_sequence = -1
        self.last_capture_ns = -1
        self.candidate_ns = self.onset_ns = None
        self.stable_ns = None
        self.target = None
        self.revision = 0

    def _phase(self, phase, now):
        self.phase, self.phase_ns, self.stable_ns = phase,now,None

    def _brake(self, now, reason=None):
        if self.phase != 'BRAKING': self._phase('BRAKING',now)
        if reason: self.reason = self.reason or reason

    def step(self, now_ns, feedback, prediction=None):
        c = self.config
        if type(now_ns) is not int or now_ns < 0 or (self.last_ns is not None and now_ns <= self.last_ns):
            raise ValueError('Control ticks must strictly increase on one host clock')
        if feedback.host_ns > now_ns or (prediction and prediction.available_ns > now_ns):
            raise ValueError('Future observations cannot control past actions')
        dt = 0. if self.last_ns is None else (now_ns-self.last_ns)/1e9
        first = self.last_ns is None
        self.last_ns = now_ns
        if first:
            self.target = feedback.closure
            self._phase('GRIPPING',now_ns)
        if self.phase in self.terminal: return self._decision(now_ns,None)
        if (now_ns-feedback.host_ns)/1e9 > c.max_feedback_age:
            self.reason = 'stale_gripper_feedback'
            self._phase('FAULT',now_ns)
            return self._decision(now_ns,None)
        if dt > c.max_tick_gap:
            self._brake(now_ns,'control_tick_gap')
        fresh = prediction is not None and (now_ns-prediction.captured_ns)/1e9 <= c.max_prediction_age
        if not fresh:
            self._brake(now_ns,'stale_or_missing_snn')
        new = False
        if fresh and prediction.sequence != self.last_sequence:
            if prediction.sequence < self.last_sequence or prediction.captured_ns <= self.last_capture_ns:
                self._brake(now_ns,'snn_sequence_or_time_reversed')
            else:
                new = True
                self.last_sequence,self.last_capture_ns = prediction.sequence,prediction.captured_ns
        qualified = bool(fresh and prediction.task == 'slide' and prediction.task_confidence >= c.confidence
                         and prediction.direction_confidence >= c.confidence)
        downward = qualified and prediction.direction == c.expected_direction
        if new and self.phase in ('GRIPPING','OPENING','SLIDING'):
            if qualified and not downward:
                self._brake(now_ns,'unexpected_slide_direction')
            elif prediction.task in ('rotate','escape_wriggle') and prediction.task_confidence >= c.confidence:
                self._brake(now_ns,'unexpected_contact_motion')
        if self.phase == 'GRIPPING':
            if abs(feedback.closure-c.grip_closure) <= c.arrival_tolerance:
                if self.stable_ns is None: self.stable_ns = now_ns
                if (now_ns-self.stable_ns)/1e9 >= c.grip_settle_seconds:
                    self._phase('OPENING',now_ns)
            else: self.stable_ns = None
            if (now_ns-self.phase_ns)/1e9 > c.release_timeout:
                self._brake(now_ns,'grip_timeout')
        if self.phase == 'OPENING':
            if new:
                if downward:
                    if self.candidate_ns is None: self.candidate_ns = prediction.available_ns
                    if (prediction.available_ns-self.candidate_ns)/1e9 >= c.confirm_seconds:
                        self.onset_ns = self.candidate_ns
                        self._phase('SLIDING',now_ns)
                else: self.candidate_ns = None
            if (now_ns-self.phase_ns)/1e9 >= c.release_timeout:
                self._brake(now_ns,'no_slide_timeout')
        if self.phase == 'SLIDING' and (now_ns-self.onset_ns)/1e9 >= c.slide_seconds-c.close_delay_seconds:
            self._brake(now_ns)
        if self.phase == 'BRAKING':
            # Issue the full grip target before accepting arrival; a small
            # release can already lie within the feedback tolerance.
            if (now_ns > self.phase_ns and abs(self.target-c.grip_closure) < 1e-9
                    and abs(feedback.closure-c.grip_closure) <= c.arrival_tolerance):
                if self.stable_ns is None: self.stable_ns = now_ns
                if (now_ns-self.stable_ns)/1e9 >= c.grip_settle_seconds:
                    self._phase('FAULT' if self.reason else 'FINISHED',now_ns)
            else: self.stable_ns = None
            if (now_ns-self.phase_ns)/1e9 >= c.brake_timeout and self.phase == 'BRAKING':
                self.reason = self.reason or 'brake_timeout'
                self._phase('FAULT',now_ns)
        if self.phase in self.terminal: return self._decision(now_ns,None)
        if self.phase == 'OPENING' and self.candidate_ns is None:
            desired = max(c.grip_closure-c.max_open_delta,self.target-c.open_rate*dt)
        elif self.phase in ('GRIPPING','BRAKING'):
            desired = self.target + max(-c.close_rate*dt,min(c.close_rate*dt,c.grip_closure-self.target))
        else:
            desired = self.target  # Keep the current aperture while timing sliding.
        self.target = min(1.,max(0.,min(feedback.closure+c.max_tracking_delta,
                                      max(feedback.closure-c.max_tracking_delta,desired))))
        self.revision += 1
        return self._decision(now_ns,GripperCommand(now_ns,self.target,self.revision))

    def _decision(self, now, command):
        return {'time_ns':now,'phase':self.phase,'fault_reason':self.reason,
                'onset_available_ns':self.onset_ns,'candidate_available_ns':self.candidate_ns,
                'elapsed_since_onset_s':None if self.onset_ns is None else (now-self.onset_ns)/1e9,
                'distance_m':None,'command':asdict(command) if command else None}
