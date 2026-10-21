"""
app/ai/ml/models/work_order.py

Industrial Maintenance Work Order Models
========================================

Python 3.11+
Pydantic v2

Install:
    pip install "pydantic>=2,<3"

Features:
- Multi-tenant work orders
- Preventive, predictive, corrective and emergency work
- AI prediction and inspection references
- Technician assignments
- Spare parts and inventory reservations
- Safety permits and LOTO verification records
- Approval and execution workflows
- Work-order checklists
- Repair and replacement cost estimates
- Downtime and production loss tracking
- Work-order history
- Create, update, response and summary schemas

Domain models only.

Authorization, inventory reservation, persistence,
notification delivery and actual industrial safety
procedures must be implemented separately.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_datetime(
    value: datetime | None,
) -> datetime | None:
    if value is not None:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "Datetime must be timezone-aware"
            )
    return value


Money = Decimal


# ============================================================
# ENUMS
# ============================================================

class WorkOrderType(str, Enum):
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    CONDITION_BASED = "condition_based"
    EMERGENCY = "emergency"
    INSPECTION = "inspection"
    CALIBRATION = "calibration"
    REPAIR = "repair"
    REPLACEMENT = "replacement"
    OVERHAUL = "overhaul"


class WorkOrderStatus(str, Enum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    PENDING_VERIFICATION = "pending_verification"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class WorkOrderPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class WorkOrderSource(str, Enum):
    MANUAL = "manual"
    SCHEDULE = "schedule"
    SENSOR_ALERT = "sensor_alert"
    ANOMALY_DETECTION = "anomaly_detection"
    FAILURE_PREDICTION = "failure_prediction"
    RUL_PREDICTION = "rul_prediction"
    INSPECTION = "inspection"
    SAFETY_EVENT = "safety_event"
    INCIDENT = "incident"
    AI_AGENT = "ai_agent"


class WorkOrderDecision(str, Enum):
    INSPECT = "inspect"
    MONITOR = "monitor"
    REPAIR = "repair"
    REPLACE = "replace"
    CALIBRATE = "calibrate"
    OVERHAUL = "overhaul"
    ENGINEERING_REVIEW = "engineering_review"


class WorkOrderTaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class WorkOrderSafetyStatus(str, Enum):
    NOT_ASSESSED = "not_assessed"
    REVIEW_REQUIRED = "review_required"
    APPROVED = "approved"
    BLOCKED = "blocked"


class WorkOrderHistoryAction(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    STARTED = "started"
    PAUSED = "paused"
    RESUMED = "resumed"
    VERIFICATION_REQUESTED = "verification_requested"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# ============================================================
# BASE MODEL
# ============================================================

class WorkOrderBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        validate_default=True,
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ============================================================
# WORK ORDER CONTEXT
# ============================================================

class WorkOrderContext(WorkOrderBaseModel):
    company_id: str = Field(min_length=1)
    site_id: str = Field(min_length=1)

    asset_id: str | None = None
    machine_id: str | None = None
    equipment_id: str | None = None
    component_id: str | None = None

    production_line_id: str | None = None
    zone_id: str | None = None

    @model_validator(mode="after")
    def validate_target(self) -> "WorkOrderContext":
        if not any((
            self.asset_id,
            self.machine_id,
            self.equipment_id,
            self.component_id,
        )):
            raise ValueError(
                "Work order requires an asset, machine, "
                "equipment or component reference"
            )
        return self


# ============================================================
# SOURCE REFERENCES
# ============================================================

class WorkOrderSourceReference(WorkOrderBaseModel):
    prediction_id: str | None = None
    anomaly_id: str | None = None
    failure_id: str | None = None

    inspection_id: str | None = None
    incident_id: str | None = None
    safety_event_id: str | None = None

    sensor_id: str | None = None
    telemetry_id: str | None = None

    model_id: str | None = None
    model_version: str | None = None

    failure_probability: float | None = Field(
        default=None,
        ge=0,
        le=1,
        allow_inf_nan=False,
    )

    anomaly_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        allow_inf_nan=False,
    )

    remaining_useful_life_hours: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )

    explanation: str | None = None


# ============================================================
# TECHNICIAN ASSIGNMENT
# ============================================================

class WorkOrderTechnician(WorkOrderBaseModel):
    technician_id: str = Field(min_length=1)

    technician_name: str | None = None

    role: str = Field(min_length=1)

    assigned_at: datetime = Field(
        default_factory=utc_now
    )

    estimated_hours: float = Field(
        default=0,
        ge=0,
        allow_inf_nan=False,
    )

    actual_hours: float = Field(
        default=0,
        ge=0,
        allow_inf_nan=False,
    )

    hourly_rate: Money = Field(
        default=Decimal("0"),
        ge=0,
        allow_inf_nan=False,
    )

    @field_validator("assigned_at")
    @classmethod
    def validate_date(cls, value: datetime) -> datetime:
        return validate_datetime(value)

    @property
    def estimated_labor_cost(self) -> Money:
        return (
            Decimal(str(self.estimated_hours))
            * self.hourly_rate
        )

    @property
    def actual_labor_cost(self) -> Money:
        return (
            Decimal(str(self.actual_hours))
            * self.hourly_rate
        )


# ============================================================
# SPARE PART
# ============================================================

class WorkOrderSparePart(WorkOrderBaseModel):
    part_id: str = Field(min_length=1)

    part_name: str = Field(min_length=1)

    part_number: str | None = None

    quantity_required: float = Field(
        gt=0,
        allow_inf_nan=False,
    )

    quantity_used: float = Field(
        default=0,
        ge=0,
        allow_inf_nan=False,
    )

    unit: str = "pcs"

    unit_cost: Money = Field(
        default=Decimal("0"),
        ge=0,
        allow_inf_nan=False,
    )

    inventory_reference_id: str | None = None
    reservation_reference_id: str | None = None

    supplier_id: str | None = None

    @model_validator(mode="after")
    def validate_quantities(self) -> "WorkOrderSparePart":
        if self.quantity_used > self.quantity_required:
            raise ValueError(
                "Quantity used exceeds quantity required"
            )
        return self

    @property
    def estimated_cost(self) -> Money:
        return (
            Decimal(str(self.quantity_required))
            * self.unit_cost
        )

    @property
    def actual_cost(self) -> Money:
        return (
            Decimal(str(self.quantity_used))
            * self.unit_cost
        )


# ============================================================
# WORK ORDER TASK
# ============================================================

class WorkOrderTask(WorkOrderBaseModel):
    task_id: UUID = Field(default_factory=uuid4)

    title: str = Field(min_length=1)
    description: str | None = None

    status: WorkOrderTaskStatus = (
        WorkOrderTaskStatus.PENDING
    )

    mandatory: bool = True

    assigned_to: str | None = None

    estimated_hours: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )

    completed_by: str | None = None
    completed_at: datetime | None = None

    completion_notes: str | None = None

    @field_validator("completed_at")
    @classmethod
    def validate_date(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_datetime(value)

    @model_validator(mode="after")
    def validate_completion(self) -> "WorkOrderTask":
        if self.status == WorkOrderTaskStatus.COMPLETED.value:
            if not self.completed_by or not self.completed_at:
                raise ValueError(
                    "Completed task requires actor and timestamp"
                )
        return self


# ============================================================
# SAFETY REQUIREMENTS
# ============================================================

class WorkOrderSafety(WorkOrderBaseModel):
    status: WorkOrderSafetyStatus = (
        WorkOrderSafetyStatus.NOT_ASSESSED
    )

    risk_assessment_required: bool = True
    risk_assessment_reference: str | None = None

    work_permit_required: bool = False
    work_permit_id: str | None = None

    lockout_tagout_required: bool = False
    lockout_tagout_verified: bool = False
    lockout_tagout_reference: str | None = None

    required_ppe: list[str] = Field(
        default_factory=list
    )

    identified_hazards: list[str] = Field(
        default_factory=list
    )

    reviewed_by: str | None = None
    reviewed_at: datetime | None = None

    notes: str | None = None

    @field_validator("reviewed_at")
    @classmethod
    def validate_date(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_datetime(value)

    @model_validator(mode="after")
    def validate_safety(self) -> "WorkOrderSafety":
        if self.status == WorkOrderSafetyStatus.APPROVED.value:
            if not self.reviewed_by or not self.reviewed_at:
                raise ValueError(
                    "Safety approval requires reviewer and time"
                )

            if (
                self.risk_assessment_required
                and not self.risk_assessment_reference
            ):
                raise ValueError(
                    "Risk assessment reference is required"
                )

            if (
                self.work_permit_required
                and not self.work_permit_id
            ):
                raise ValueError(
                    "Work permit reference is required"
                )

            if self.lockout_tagout_required and not (
                self.lockout_tagout_verified
                and self.lockout_tagout_reference
            ):
                raise ValueError(
                    "LOTO verification record is required"
                )

        return self

    def is_approved(self) -> bool:
        return self.status == WorkOrderSafetyStatus.APPROVED.value


# ============================================================
# SCHEDULE
# ============================================================

class WorkOrderSchedule(WorkOrderBaseModel):
    scheduled_start_at: datetime | None = None
    scheduled_end_at: datetime | None = None

    actual_start_at: datetime | None = None
    actual_end_at: datetime | None = None

    estimated_duration_hours: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )

    @field_validator(
        "scheduled_start_at",
        "scheduled_end_at",
        "actual_start_at",
        "actual_end_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_datetime(value)

    @model_validator(mode="after")
    def validate_schedule(self) -> "WorkOrderSchedule":
        if (
            self.scheduled_start_at is not None
            and self.scheduled_end_at is not None
            and self.scheduled_end_at <= self.scheduled_start_at
        ):
            raise ValueError(
                "Scheduled end must follow start"
            )

        if (
            self.actual_start_at is not None
            and self.actual_end_at is not None
            and self.actual_end_at < self.actual_start_at
        ):
            raise ValueError(
                "Actual end cannot precede actual start"
            )

        return self


# ============================================================
# WORK ORDER COST
# ============================================================

class WorkOrderCost(WorkOrderBaseModel):
    currency: str = Field(
        default="INR",
        pattern=r"^[A-Z]{3}$",
    )

    labor_cost: Money = Field(
        default=Decimal("0"),
        ge=0,
    )

    spare_parts_cost: Money = Field(
        default=Decimal("0"),
        ge=0,
    )

    repair_cost: Money = Field(
        default=Decimal("0"),
        ge=0,
    )

    replacement_cost: Money = Field(
        default=Decimal("0"),
        ge=0,
    )

    downtime_cost: Money = Field(
        default=Decimal("0"),
        ge=0,
    )

    production_loss_cost: Money = Field(
        default=Decimal("0"),
        ge=0,
    )

    inspection_cost: Money = Field(
        default=Decimal("0"),
        ge=0,
    )

    other_cost: Money = Field(
        default=Decimal("0"),
        ge=0,
    )

    @property
    def total_cost(self) -> Money:
        return sum(
            (
                self.labor_cost,
                self.spare_parts_cost,
                self.repair_cost,
                self.replacement_cost,
                self.downtime_cost,
                self.production_loss_cost,
                self.inspection_cost,
                self.other_cost,
            ),
            Decimal("0"),
        )


# ============================================================
# DOWNTIME
# ============================================================

class WorkOrderDowntime(WorkOrderBaseModel):
    planned_downtime_hours: float = Field(
        default=0,
        ge=0,
        allow_inf_nan=False,
    )

    actual_downtime_hours: float = Field(
        default=0,
        ge=0,
        allow_inf_nan=False,
    )

    production_loss_units: float = Field(
        default=0,
        ge=0,
        allow_inf_nan=False,
    )

    downtime_reason: str | None = None

    production_line_id: str | None = None


# ============================================================
# WORK ORDER HISTORY
# ============================================================

class WorkOrderHistory(WorkOrderBaseModel):
    history_id: UUID = Field(default_factory=uuid4)

    action: WorkOrderHistoryAction

    actor_id: str = Field(min_length=1)

    timestamp: datetime = Field(
        default_factory=utc_now
    )

    previous_status: WorkOrderStatus | None = None
    new_status: WorkOrderStatus | None = None

    notes: str | None = None

    @field_validator("timestamp")
    @classmethod
    def validate_date(cls, value: datetime) -> datetime:
        return validate_datetime(value)


# ============================================================
# MAIN WORK ORDER
# ============================================================

class IndustrialWorkOrder(WorkOrderBaseModel):
    work_order_id: UUID = Field(
        default_factory=uuid4
    )

    work_order_code: str = Field(
        min_length=1,
        max_length=100,
    )

    context: WorkOrderContext

    work_order_type: WorkOrderType

    source: WorkOrderSource = WorkOrderSource.MANUAL

    title: str = Field(
        min_length=1,
        max_length=250,
    )

    description: str | None = None

    status: WorkOrderStatus = WorkOrderStatus.DRAFT

    priority: WorkOrderPriority = WorkOrderPriority.MEDIUM

    decision: WorkOrderDecision | None = None

    source_reference: WorkOrderSourceReference = Field(
        default_factory=WorkOrderSourceReference
    )

    technicians: list[WorkOrderTechnician] = Field(
        default_factory=list
    )

    spare_parts: list[WorkOrderSparePart] = Field(
        default_factory=list
    )

    tasks: list[WorkOrderTask] = Field(
        default_factory=list
    )

    safety: WorkOrderSafety = Field(
        default_factory=WorkOrderSafety
    )

    schedule: WorkOrderSchedule = Field(
        default_factory=WorkOrderSchedule
    )

    estimated_cost: WorkOrderCost = Field(
        default_factory=WorkOrderCost
    )

    actual_cost: WorkOrderCost | None = None

    downtime: WorkOrderDowntime = Field(
        default_factory=WorkOrderDowntime
    )

    approval_required: bool = True

    approved_by: str | None = None
    approved_at: datetime | None = None

    verified_by: str | None = None
    verified_at: datetime | None = None

    completion_notes: str | None = None

    cancellation_reason: str | None = None

    created_by: str = Field(min_length=1)

    history: list[WorkOrderHistory] = Field(
        default_factory=list
    )

    tags: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=utc_now
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )

    @field_validator(
        "approved_at",
        "verified_at",
        "created_at",
        "updated_at",
    )
    @classmethod
    def validate_dates(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return validate_datetime(value)

    @model_validator(mode="after")
    def validate_work_order(self) -> "IndustrialWorkOrder":
        technician_ids = [
            item.technician_id
            for item in self.technicians
        ]

        if len(technician_ids) != len(set(technician_ids)):
            raise ValueError(
                "Duplicate technician assignments"
            )

        task_ids = [item.task_id for item in self.tasks]

        if len(task_ids) != len(set(task_ids)):
            raise ValueError(
                "Duplicate task IDs"
            )

        if (
            self.approved_at is not None
            and not self.approved_by
        ):
            raise ValueError(
                "Approval timestamp requires approver"
            )

        if (
            self.verified_at is not None
            and not self.verified_by
        ):
            raise ValueError(
                "Verification timestamp requires verifier"
            )

        if self.status == WorkOrderStatus.COMPLETED.value:
            if not self.verified_by or not self.verified_at:
                raise ValueError(
                    "Completed work order requires verification"
                )

            if any(
                task.mandatory
                and task.status
                != WorkOrderTaskStatus.COMPLETED.value
                for task in self.tasks
            ):
                raise ValueError(
                    "All mandatory tasks must be completed"
                )

        return self

    # --------------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------------

    def _touch(self) -> None:
        self.updated_at = utc_now()

    def _record_history(
        self,
        action: WorkOrderHistoryAction,
        actor_id: str,
        previous_status: WorkOrderStatus | str,
        notes: str | None = None,
    ) -> None:
        if not actor_id.strip():
            raise ValueError("Actor ID is required")

        self.history.append(
            WorkOrderHistory(
                action=action,
                actor_id=actor_id,
                previous_status=previous_status,
                new_status=self.status,
                notes=notes,
            )
        )

        self._touch()

    # --------------------------------------------------------
    # ASSIGN TECHNICIAN
    # --------------------------------------------------------

    def assign_technician(
        self,
        technician: WorkOrderTechnician,
    ) -> None:
        if self.status in {
            WorkOrderStatus.COMPLETED.value,
            WorkOrderStatus.CANCELLED.value,
        }:
            raise ValueError(
                "Cannot modify a closed work order"
            )

        if any(
            item.technician_id == technician.technician_id
            for item in self.technicians
        ):
            raise ValueError(
                "Technician already assigned"
            )

        self.technicians.append(technician)
        self._touch()

    # --------------------------------------------------------
    # ADD SPARE PART
    # --------------------------------------------------------

    def add_spare_part(
        self,
        spare_part: WorkOrderSparePart,
    ) -> None:
        if self.status in {
            WorkOrderStatus.COMPLETED.value,
            WorkOrderStatus.CANCELLED.value,
        }:
            raise ValueError(
                "Cannot modify a closed work order"
            )

        if any(
            item.part_id == spare_part.part_id
            for item in self.spare_parts
        ):
            raise ValueError(
                "Spare part already exists"
            )

        self.spare_parts.append(spare_part)
        self._touch()

    # --------------------------------------------------------
    # ADD TASK
    # --------------------------------------------------------

    def add_task(
        self,
        task: WorkOrderTask,
    ) -> None:
        if self.status in {
            WorkOrderStatus.COMPLETED.value,
            WorkOrderStatus.CANCELLED.value,
        }:
            raise ValueError(
                "Cannot modify a closed work order"
            )

        if any(
            item.task_id == task.task_id
            for item in self.tasks
        ):
            raise ValueError(
                "Task already exists"
            )

        self.tasks.append(task)
        self._touch()

    # --------------------------------------------------------
    # SUBMIT FOR APPROVAL
    # --------------------------------------------------------

    def submit_for_approval(
        self,
        actor_id: str,
    ) -> None:
        if self.status != WorkOrderStatus.DRAFT.value:
            raise ValueError(
                "Only draft work orders can be submitted"
            )

        previous = self.status
        self.status = WorkOrderStatus.PENDING_APPROVAL

        self._record_history(
            WorkOrderHistoryAction.SUBMITTED,
            actor_id,
            previous,
        )

    # --------------------------------------------------------
    # APPROVE
    # --------------------------------------------------------

    def approve(
        self,
        approver_id: str,
    ) -> None:
        if self.status != (
            WorkOrderStatus.PENDING_APPROVAL.value
        ):
            raise ValueError(
                "Work order is not pending approval"
            )

        if not approver_id.strip():
            raise ValueError(
                "Approver ID is required"
            )

        previous = self.status

        self.approved_by = approver_id
        self.approved_at = utc_now()
        self.status = WorkOrderStatus.APPROVED

        self._record_history(
            WorkOrderHistoryAction.APPROVED,
            approver_id,
            previous,
        )

    # --------------------------------------------------------
    # SCHEDULE
    # --------------------------------------------------------

    def schedule_work(
        self,
        schedule: WorkOrderSchedule,
        actor_id: str,
    ) -> None:
        if self.status != WorkOrderStatus.APPROVED.value:
            raise ValueError(
                "Work order must be approved first"
            )

        if (
            schedule.scheduled_start_at is None
            or schedule.scheduled_end_at is None
        ):
            raise ValueError(
                "Scheduled start and end are required"
            )

        previous = self.status

        self.schedule = schedule
        self.status = WorkOrderStatus.SCHEDULED

        self._record_history(
            WorkOrderHistoryAction.SCHEDULED,
            actor_id,
            previous,
        )

    # --------------------------------------------------------
    # START WORK
    # --------------------------------------------------------

    def start(
        self,
        actor_id: str,
    ) -> None:
        if self.status != WorkOrderStatus.SCHEDULED.value:
            raise ValueError(
                "Only scheduled work orders can start"
            )

        if not self.technicians:
            raise ValueError(
                "At least one technician must be assigned"
            )

        if not self.safety.is_approved():
            raise ValueError(
                "Safety review must be approved before work"
            )

        previous = self.status

        self.schedule.actual_start_at = utc_now()
        self.status = WorkOrderStatus.IN_PROGRESS

        self._record_history(
            WorkOrderHistoryAction.STARTED,
            actor_id,
            previous,
        )

    # --------------------------------------------------------
    # PAUSE WORK
    # --------------------------------------------------------

    def pause(
        self,
        actor_id: str,
        reason: str,
    ) -> None:
        if self.status != WorkOrderStatus.IN_PROGRESS.value:
            raise ValueError(
                "Only in-progress work can be paused"
            )

        if not reason.strip():
            raise ValueError(
                "Pause reason is required"
            )

        previous = self.status
        self.status = WorkOrderStatus.ON_HOLD

        self._record_history(
            WorkOrderHistoryAction.PAUSED,
            actor_id,
            previous,
            reason,
        )

    # --------------------------------------------------------
    # RESUME WORK
    # --------------------------------------------------------

    def resume(
        self,
        actor_id: str,
    ) -> None:
        if self.status != WorkOrderStatus.ON_HOLD.value:
            raise ValueError(
                "Only on-hold work can resume"
            )

        if not self.safety.is_approved():
            raise ValueError(
                "Safety approval is required"
            )

        previous = self.status
        self.status = WorkOrderStatus.IN_PROGRESS

        self._record_history(
            WorkOrderHistoryAction.RESUMED,
            actor_id,
            previous,
        )

    # --------------------------------------------------------
    # SUBMIT FOR VERIFICATION
    # --------------------------------------------------------

    def submit_for_verification(
        self,
        actor_id: str,
        completion_notes: str,
    ) -> None:
        if self.status != WorkOrderStatus.IN_PROGRESS.value:
            raise ValueError(
                "Work order must be in progress"
            )

        incomplete_tasks = [
            task.title
            for task in self.tasks
            if (
                task.mandatory
                and task.status
                != WorkOrderTaskStatus.COMPLETED.value
            )
        ]

        if incomplete_tasks:
            raise ValueError(
                "Mandatory tasks incomplete: "
                + ", ".join(incomplete_tasks)
            )

        if not completion_notes.strip():
            raise ValueError(
                "Completion notes are required"
            )

        previous = self.status

        self.completion_notes = completion_notes
        self.status = WorkOrderStatus.PENDING_VERIFICATION

        self._record_history(
            WorkOrderHistoryAction.VERIFICATION_REQUESTED,
            actor_id,
            previous,
        )

    # --------------------------------------------------------
    # COMPLETE WORK ORDER
    # --------------------------------------------------------

    def complete(
        self,
        verifier_id: str,
        actual_cost: WorkOrderCost | None = None,
    ) -> None:
        if self.status != (
            WorkOrderStatus.PENDING_VERIFICATION.value
        ):
            raise ValueError(
                "Work order is not pending verification"
            )

        if not verifier_id.strip():
            raise ValueError(
                "Verifier ID is required"
            )

        if actual_cost is not None:
            if (
                actual_cost.currency
                != self.estimated_cost.currency
            ):
                raise ValueError(
                    "Estimated and actual currencies differ"
                )

            self.actual_cost = actual_cost

        previous = self.status

        self.verified_by = verifier_id
        self.verified_at = utc_now()
        self.schedule.actual_end_at = self.verified_at
        self.status = WorkOrderStatus.COMPLETED

        self._record_history(
            WorkOrderHistoryAction.COMPLETED,
            verifier_id,
            previous,
        )

    # --------------------------------------------------------
    # CANCEL
    # --------------------------------------------------------

    def cancel(
        self,
        actor_id: str,
        reason: str,
    ) -> None:
        if self.status in {
            WorkOrderStatus.COMPLETED.value,
            WorkOrderStatus.CANCELLED.value,
        }:
            raise ValueError(
                "Cannot cancel a closed work order"
            )

        if not reason.strip():
            raise ValueError(
                "Cancellation reason is required"
            )

        previous = self.status

        self.cancellation_reason = reason
        self.status = WorkOrderStatus.CANCELLED

        self._record_history(
            WorkOrderHistoryAction.CANCELLED,
            actor_id,
            previous,
            reason,
        )

    # --------------------------------------------------------
    # FINANCIAL HELPERS
    # --------------------------------------------------------

    def estimated_spare_parts_cost(self) -> Money:
        return sum(
            (
                part.estimated_cost
                for part in self.spare_parts
            ),
            Decimal("0"),
        )

    def actual_spare_parts_cost(self) -> Money:
        return sum(
            (
                part.actual_cost
                for part in self.spare_parts
            ),
            Decimal("0"),
        )

    def estimated_labor_cost(self) -> Money:
        return sum(
            (
                technician.estimated_labor_cost
                for technician in self.technicians
            ),
            Decimal("0"),
        )

    def actual_labor_cost(self) -> Money:
        return sum(
            (
                technician.actual_labor_cost
                for technician in self.technicians
            ),
            Decimal("0"),
        )

    def cost_variance(self) -> Money | None:
        if self.actual_cost is None:
            return None

        return (
            self.actual_cost.total_cost
            - self.estimated_cost.total_cost
        )

    # --------------------------------------------------------
    # TASK PROGRESS
    # --------------------------------------------------------

    def task_completion_percent(self) -> float:
        if not self.tasks:
            return 0.0

        completed = sum(
            task.status
            == WorkOrderTaskStatus.COMPLETED.value
            for task in self.tasks
        )

        return round(
            completed / len(self.tasks) * 100,
            2,
        )

    # --------------------------------------------------------
    # OVERDUE
    # --------------------------------------------------------

    def is_overdue(
        self,
        reference_time: datetime | None = None,
    ) -> bool:
        if self.status in {
            WorkOrderStatus.COMPLETED.value,
            WorkOrderStatus.CANCELLED.value,
        }:
            return False

        end_at = self.schedule.scheduled_end_at

        if end_at is None:
            return False

        now = reference_time or utc_now()
        validate_datetime(now)

        return now > end_at


# ============================================================
# CREATE SCHEMA
# ============================================================

class WorkOrderCreate(WorkOrderBaseModel):
    work_order_code: str = Field(min_length=1)

    context: WorkOrderContext

    work_order_type: WorkOrderType

    source: WorkOrderSource = WorkOrderSource.MANUAL

    title: str = Field(min_length=1)

    description: str | None = None

    priority: WorkOrderPriority = WorkOrderPriority.MEDIUM

    decision: WorkOrderDecision | None = None

    source_reference: WorkOrderSourceReference = Field(
        default_factory=WorkOrderSourceReference
    )

    created_by: str = Field(min_length=1)


# ============================================================
# UPDATE SCHEMA
# ============================================================

class WorkOrderUpdate(WorkOrderBaseModel):
    title: str | None = None
    description: str | None = None

    priority: WorkOrderPriority | None = None
    decision: WorkOrderDecision | None = None

    tags: list[str] | None = None

    metadata: dict[str, Any] | None = None


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class WorkOrderResponse(IndustrialWorkOrder):
    pass


# ============================================================
# SUMMARY SCHEMA
# ============================================================

class WorkOrderSummary(WorkOrderBaseModel):
    work_order_id: UUID

    work_order_code: str

    company_id: str
    site_id: str

    asset_id: str | None
    machine_id: str | None

    title: str

    work_order_type: WorkOrderType
    priority: WorkOrderPriority
    status: WorkOrderStatus

    decision: WorkOrderDecision | None

    technician_count: int
    task_count: int

    task_completion_percent: float

    estimated_total_cost: Money
    actual_total_cost: Money | None

    currency: str

    overdue: bool

    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_work_order(
        cls,
        order: IndustrialWorkOrder,
    ) -> "WorkOrderSummary":
        return cls(
            work_order_id=order.work_order_id,
            work_order_code=order.work_order_code,
            company_id=order.context.company_id,
            site_id=order.context.site_id,
            asset_id=order.context.asset_id,
            machine_id=order.context.machine_id,
            title=order.title,
            work_order_type=order.work_order_type,
            priority=order.priority,
            status=order.status,
            decision=order.decision,
            technician_count=len(order.technicians),
            task_count=len(order.tasks),
            task_completion_percent=(
                order.task_completion_percent()
            ),
            estimated_total_cost=(
                order.estimated_cost.total_cost
            ),
            actual_total_cost=(
                order.actual_cost.total_cost
                if order.actual_cost is not None
                else None
            ),
            currency=order.estimated_cost.currency,
            overdue=order.is_overdue(),
            created_at=order.created_at,
            updated_at=order.updated_at,
        )


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json
    from datetime import timedelta

    order = IndustrialWorkOrder(
        work_order_code="WO-2026-001",
        context=WorkOrderContext(
            company_id="COMPANY-001",
            site_id="SITE-001",
            asset_id="ASSET-101",
            machine_id="PUMP-101",
            component_id="BEARING-101",
        ),
        work_order_type=WorkOrderType.PREDICTIVE,
        source=WorkOrderSource.FAILURE_PREDICTION,
        title="Replace Pump Motor Bearing",
        description=(
            "Predictive maintenance based on "
            "bearing vibration and failure risk"
        ),
        priority=WorkOrderPriority.HIGH,
        decision=WorkOrderDecision.REPLACE,
        source_reference=WorkOrderSourceReference(
            prediction_id="PRED-101",
            model_id="FAILURE-MODEL-01",
            model_version="1.0.0",
            failure_probability=0.82,
            remaining_useful_life_hours=120,
        ),
        created_by="ENGINEER-001",
        estimated_cost=WorkOrderCost(
            currency="INR",
            labor_cost=Decimal("4000"),
            spare_parts_cost=Decimal("8500"),
            downtime_cost=Decimal("3000"),
        ),
    )

    order.assign_technician(
        WorkOrderTechnician(
            technician_id="TECH-001",
            technician_name="Maintenance Technician",
            role="Mechanical Technician",
            estimated_hours=4,
            hourly_rate=Decimal("1000"),
        )
    )

    order.add_spare_part(
        WorkOrderSparePart(
            part_id="BEARING-6205",
            part_name="Industrial Motor Bearing",
            quantity_required=1,
            unit_cost=Decimal("8500"),
        )
    )

    order.add_task(
        WorkOrderTask(
            title="Inspect bearing condition",
            mandatory=True,
        )
    )

    order.submit_for_approval("ENGINEER-001")
    order.approve("MANAGER-001")

    now = utc_now()

    order.schedule_work(
        WorkOrderSchedule(
            scheduled_start_at=now + timedelta(hours=1),
            scheduled_end_at=now + timedelta(hours=6),
            estimated_duration_hours=5,
        ),
        actor_id="PLANNER-001",
    )

    order.safety = WorkOrderSafety(
        status=WorkOrderSafetyStatus.APPROVED,
        risk_assessment_required=True,
        risk_assessment_reference="RA-001",
        work_permit_required=True,
        work_permit_id="PERMIT-001",
        lockout_tagout_required=True,
        lockout_tagout_verified=True,
        lockout_tagout_reference="LOTO-001",
        required_ppe=[
            "Safety helmet",
            "Safety gloves",
            "Safety shoes",
        ],
        reviewed_by="SAFETY-001",
        reviewed_at=utc_now(),
    )

    order.start("TECH-001")

    task = order.tasks[0]
    task.completed_by = "TECH-001"
    task.completed_at = utc_now()
    task.status = WorkOrderTaskStatus.COMPLETED

    order.submit_for_verification(
        actor_id="TECH-001",
        completion_notes=(
            "Bearing inspection and replacement completed"
        ),
    )

    order.complete(
        verifier_id="ENGINEER-002",
        actual_cost=WorkOrderCost(
            currency="INR",
            labor_cost=Decimal("4200"),
            spare_parts_cost=Decimal("8500"),
            downtime_cost=Decimal("2500"),
        ),
    )

    print(
        json.dumps(
            WorkOrderSummary.from_work_order(
                order
            ).to_dict(),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()