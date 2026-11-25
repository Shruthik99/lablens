from enum import Enum

from pydantic import BaseModel, model_validator


class Action(str, Enum):
    START = "start"
    STOP = "stop"
    INCREASE = "increase"
    DECREASE = "decrease"
    ONE_TIME_DOSE = "one_time_dose"
    NO_CHANGE = "no_change"
    UNKNOWN = "unknown"


class Setting(str, Enum):
    BEFORE_ADMISSION = "before_admission"
    IN_HOSPITAL = "in_hospital"
    AT_DISCHARGE = "at_discharge"
    UNKNOWN = "unknown"


class Medication(BaseModel):
    drug: str
    strength: str | None = None
    dosage: str | None = None
    route: str | None = None
    frequency: str | None = None
    duration: str | None = None
    action: Action = Action.UNKNOWN
    setting: Setting = Setting.UNKNOWN
    evidence: str
    start: int
    end: int

    @model_validator(mode="after")
    def _check_span_and_evidence(self) -> "Medication":
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        if not self.evidence:
            raise ValueError("evidence must not be empty")
        return self
