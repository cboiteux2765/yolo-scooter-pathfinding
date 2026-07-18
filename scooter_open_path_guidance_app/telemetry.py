from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass(slots=True)
class TelemetryConfig:
    service_name: str = "scooter-open-path-guidance"
    exporter: str = "console"
    endpoint: str | None = None
    metric_export_interval_ms: int = 5000


class GuidanceTelemetry:
    """Manual OpenTelemetry hooks for the OpenCV guidance loop."""

    def __init__(self, config: TelemetryConfig) -> None:
        self.config = config

        if config.exporter not in {"console", "otlp_http"}:
            raise ValueError(f"Unsupported telemetry exporter: {config.exporter}")

        if config.endpoint and config.exporter == "otlp_http":
            os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = config.endpoint

        try:
            from opentelemetry import metrics, trace
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import (
                ConsoleMetricExporter,
                PeriodicExportingMetricReader,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        except ImportError as exc:
            raise RuntimeError(
                "OpenTelemetry support requires `opentelemetry-api`, `opentelemetry-sdk`, "
                "and `opentelemetry-exporter-otlp-proto-http` in your environment."
            ) from exc

        if config.exporter == "otlp_http":
            try:
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            except ImportError as exc:
                raise RuntimeError(
                    "OTLP export requires `opentelemetry-exporter-otlp-proto-http`."
                ) from exc
            span_exporter = OTLPSpanExporter()
            metric_exporter = OTLPMetricExporter()
        else:
            span_exporter = ConsoleSpanExporter()
            metric_exporter = ConsoleMetricExporter()

        resource = Resource.create(
            {
                "service.name": config.service_name,
                "service.namespace": "scooter_open_path_guidance_app",
            }
        )

        trace_provider = TracerProvider(resource=resource)
        trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(trace_provider)

        metric_reader = PeriodicExportingMetricReader(
            metric_exporter,
            export_interval_millis=config.metric_export_interval_ms,
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)

        self._trace_provider = trace_provider
        self._meter_provider = meter_provider
        self._trace = trace
        self.tracer = trace.get_tracer(config.service_name)
        self.meter = metrics.get_meter(config.service_name)

        self.frames_total = self.meter.create_counter(
            "guidance.frames.total",
            description="Total processed frames",
        )
        self.capture_failures_total = self.meter.create_counter(
            "guidance.capture_failures.total",
            description="Failed OpenCV capture reads",
        )
        self.commands_total = self.meter.create_counter(
            "guidance.commands.total",
            description="Emitted guidance commands by type",
        )
        self.detections_per_frame = self.meter.create_histogram(
            "guidance.detections_per_frame",
            description="Detections seen in each processed frame",
        )
        self.obstacles_per_frame = self.meter.create_histogram(
            "guidance.obstacles_per_frame",
            description="Tracked obstacles considered by the planner per frame",
        )
        self.risk_score_histogram = self.meter.create_histogram(
            "guidance.risk_score",
            description="Planner risk score per frame",
        )
        self.frame_latency_ms = self.meter.create_histogram(
            "guidance.frame_latency.ms",
            unit="ms",
            description="End-to-end processing time per frame",
        )
        self.capture_latency_ms = self.meter.create_histogram(
            "guidance.capture_latency.ms",
            unit="ms",
            description="OpenCV capture read latency",
        )
        self.inference_latency_ms = self.meter.create_histogram(
            "guidance.inference_latency.ms",
            unit="ms",
            description="YOLO tracking and detection latency",
        )
        self.planning_latency_ms = self.meter.create_histogram(
            "guidance.planning_latency.ms",
            unit="ms",
            description="Planner latency per frame",
        )
        self.overlay_latency_ms = self.meter.create_histogram(
            "guidance.overlay_latency.ms",
            unit="ms",
            description="Overlay rendering latency per frame",
        )
        self.publish_latency_ms = self.meter.create_histogram(
            "guidance.publish_latency.ms",
            unit="ms",
            description="HTTP payload publication latency per frame",
        )
        self.speech_enqueue_latency_ms = self.meter.create_histogram(
            "guidance.speech_enqueue_latency.ms",
            unit="ms",
            description="Local TTS enqueue latency per frame",
        )

        self._stage_histograms = {
            "capture": self.capture_latency_ms,
            "inference": self.inference_latency_ms,
            "planning": self.planning_latency_ms,
            "overlay": self.overlay_latency_ms,
            "publish": self.publish_latency_ms,
            "speech_queue": self.speech_enqueue_latency_ms,
        }

    @contextmanager
    def frame_span(
        self,
        *,
        frame_index: int,
        source: str,
    ) -> Iterator[object]:
        with self.tracer.start_as_current_span(
            "guidance.frame",
            attributes={
                "frame.index": frame_index,
                "input.source": source,
            },
        ) as span:
            yield span

    @contextmanager
    def stage(self, name: str, attributes: dict[str, object] | None = None) -> Iterator[object]:
        attrs = dict(attributes or {})
        started_at = time.perf_counter()
        with self.tracer.start_as_current_span(f"guidance.{name}", attributes=attrs) as span:
            try:
                yield span
            finally:
                duration_ms = (time.perf_counter() - started_at) * 1000.0
                histogram = self._stage_histograms.get(name)
                if histogram is not None:
                    histogram.record(duration_ms, attrs)
                span.set_attribute("duration.ms", duration_ms)

    def record_capture_failure(self, *, source: str) -> None:
        self.capture_failures_total.add(1, {"input.source": source})

    def record_frame(
        self,
        *,
        frame_span: object,
        frame_index: int,
        frame_width: int,
        frame_height: int,
        detections: int,
        obstacles: int,
        command: str,
        risk_score: float,
        latency_ms: float,
    ) -> None:
        attrs = {
            "command": command,
            "frame.width": frame_width,
            "frame.height": frame_height,
        }
        self.frames_total.add(1, attrs)
        self.commands_total.add(1, {"command": command})
        self.detections_per_frame.record(detections, attrs)
        self.obstacles_per_frame.record(obstacles, attrs)
        self.risk_score_histogram.record(risk_score, attrs)
        self.frame_latency_ms.record(latency_ms, attrs)

        frame_span.set_attribute("frame.index", frame_index)
        frame_span.set_attribute("frame.width", frame_width)
        frame_span.set_attribute("frame.height", frame_height)
        frame_span.set_attribute("detections.count", detections)
        frame_span.set_attribute("obstacles.count", obstacles)
        frame_span.set_attribute("guidance.command", command)
        frame_span.set_attribute("guidance.risk_score", risk_score)
        frame_span.set_attribute("guidance.frame_latency_ms", latency_ms)

    def shutdown(self) -> None:
        self._trace_provider.force_flush()
        self._meter_provider.force_flush()
        self._trace_provider.shutdown()
        self._meter_provider.shutdown()
