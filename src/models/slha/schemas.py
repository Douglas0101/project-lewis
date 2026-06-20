"""Schemas de dados do SLHA."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class CPUInfo(BaseModel):
    physical_cores: int = Field(..., ge=1)
    logical_cores: int = Field(..., ge=1)
    max_freq_mhz: Optional[float] = None
    architecture: str
    flags: List[str] = Field(default_factory=list)


class GPUDevice(BaseModel):
    index: int
    name: str
    total_memory_mb: int = Field(..., ge=0)
    compute_capability: Optional[str] = None


class GPUInfo(BaseModel):
    available: bool
    count: int = Field(..., ge=0)
    devices: List[GPUDevice] = Field(default_factory=list)


class RAMInfo(BaseModel):
    total_gb: float = Field(..., ge=0.0)
    available_gb: float = Field(..., ge=0.0)
    percent_used: float = Field(..., ge=0.0, le=100.0)


class DiskInfo(BaseModel):
    total_gb: float = Field(..., ge=0.0)
    available_gb: float = Field(..., ge=0.0)


class HardwareSpecs(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    hostname: str
    os: str
    cpu: CPUInfo
    gpu: GPUInfo
    ram: RAMInfo
    disk: DiskInfo


class TrainingConfig(BaseModel):
    accelerator: Literal["cpu", "gpu"]
    strategy: Literal["single_device"]
    devices: int = Field(..., ge=1)
    batch_size: int = Field(..., ge=1)
    precision: Literal["float32", "mixed_float16"]
    num_workers: int = Field(..., ge=0)
    pin_memory: bool
    gradient_clip_val: float = Field(default=1.0)
    accumulate_grad_batches: int = Field(default=1, ge=1)

    @field_validator("devices")
    @classmethod
    def devices_consistent(cls, v: int, info) -> int:
        accelerator = info.data.get("accelerator")
        if accelerator == "cpu" and v != 1:
            raise ValueError("CPU-only só suporta devices=1")
        return v


class ResourceLog(BaseModel):
    epoch: int = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    gpu_utilization_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    cpu_percent: float = Field(..., ge=0.0, le=100.0)
    ram_used_gb: float = Field(..., ge=0.0)
    alerts: List[str] = Field(default_factory=list)
