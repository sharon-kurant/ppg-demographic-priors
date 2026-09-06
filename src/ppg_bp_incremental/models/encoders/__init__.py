"""Common interfaces and cache contracts for frozen PPG encoders."""

from ppg_bp_incremental.models.encoders.base import (
    EncoderFingerprint,
    NativeEncoderOutput,
    PPGEncoder,
)

__all__ = ["EncoderFingerprint", "NativeEncoderOutput", "PPGEncoder"]
