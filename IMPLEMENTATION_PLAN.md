# Aerie — Implementation Plan

This document describes the step-by-step build plan for Aerie. Each phase produces a testable, working increment. Phases are designed so that later phases never invalidate earlier work.

## Guiding Principles

- **SITL-first**: Every feature is developed and tested against ArduPilot SITL before touching hardware.
- **Interfaces before internals**: Define the message/event contracts between components first, then implement.
- **Each phase has a "done" test**: A concrete, automatable check that the phase works.

---

## Phase 0: Development Environment

**Goal**: Everyone on the project can run ArduPilot SITL, connect QGroundControl, and execute a scripted mission from Python.

### Steps

0.1. Install ArduPilot SITL build dependencies and build `sim_vehicle.py` for ArduCopter.

0.2. Install MAVProxy, pymavlink, QGroundControl.

0.3. Write a bootstrap script (`scripts/dev-setup.sh`) that starts SITL + MAVProxy with the correct output ports:
  - `udp:127.0.0.1:14550` → QGC
  - `udp:127.0.0.1:14551` → aerie-vision (future)
  - `udp:127.0.0.1:14552` → aerie-agent

0.4. Write a minimal Python script (`scripts/smoke_test.py`) that connects via pymavlink to `udp:127.0.0.1:14552`, waits for a heartbeat, reads GPS position, and prints it.

0.5. Verify QGC connects to `udp:127.0.0.1:14550` and displays the simulated vehicle.

### Done Test

```
scripts/dev-setup.sh starts cleanly
scripts/smoke_test.py prints a valid GPS coordinate
QGC shows the vehicle on its map
```

### Testing Approach

- `smoke_test.py` can be run in CI (headless, no QGC needed) to validate the SITL + pymavlink path.
- Add a timeout: if no heartbeat within 10 seconds, fail.

---

## Phase 1: Agent Telemetry Interface (`aerie-agent/telemetry`)

**Goal**: A Python module that connects to ArduPilot via pymavlink, subscribes to telemetry, and exposes a clean async interface for reading vehicle state.

### Steps

1.1. Define the `VehicleState` dataclass:
```python
@dataclass
class VehicleState:
    timestamp: float
    lat: float          # degrees
    lon: float          # degrees
    alt_msl: float      # meters
    alt_rel: float      # meters (relative to home)
    heading: float      # degrees
    groundspeed: float  # m/s
    airspeed: float     # m/s
    battery_pct: float  # 0-100
    flight_mode: str
    armed: bool
    gps_fix: int
    satellites: int
    mission_current: int    # current waypoint index
    mission_total: int      # total waypoints
```

1.2. Implement `TelemetryClient` class:
  - Constructor takes a pymavlink connection string (e.g., `udpin:127.0.0.1:14552`)
  - Runs an async loop that reads MAVLink messages and updates `VehicleState`
  - Key messages to handle: `HEARTBEAT`, `GLOBAL_POSITION_INT`, `GPS_RAW_INT`, `VFR_HUD`, `BATTERY_STATUS`, `MISSION_CURRENT`
  - Exposes `get_state() → VehicleState` (returns latest snapshot)
  - Exposes `subscribe(callback)` for push-based updates

1.3. Write unit tests using a mock MAVLink connection that replays recorded message sequences.

1.4. Write an integration test that connects to SITL, arms the vehicle, starts a simple 4-waypoint AUTO mission, and asserts that `mission_current` increments as waypoints are reached.

### Done Test

```
pytest tests/unit/test_telemetry.py        — passes with mock data
pytest tests/sitl/test_telemetry_live.py   — passes against running SITL
```

### Testing Approach

- Unit tests: mock the pymavlink `recv_match()` to return canned message sequences. Validate that `VehicleState` is populated correctly, including edge cases (GPS fix loss, mode changes).
- SITL integration test: script a mission via pymavlink, then read back telemetry and assert expected state transitions (GUIDED → AUTO → waypoint progression → RTL).

---

## Phase 2: Agent Command Interface (`aerie-agent/actions`)

**Goal**: A Python module that can send MAVLink commands to control the drone — mode changes, GUIDED targets, ROI, mission upload, and STATUSTEXT messages.

### Steps

2.1. Implement `CommandClient` class with methods:
```python
class CommandClient:
    async def set_mode(self, mode: str) → bool
    async def goto(self, lat: float, lon: float, alt: float) → bool
    async def set_roi(self, lat: float, lon: float, alt: float) → bool
    async def upload_mission(self, waypoints: list[Waypoint]) → bool
    async def set_current_waypoint(self, seq: int) → bool
    async def resume_mission(self) → bool  # convenience: set mode AUTO
    async def send_statustext(self, text: str, severity: int) → None
```

2.2. Each method sends the appropriate MAVLink message(s) and waits for ACK where applicable. Include timeout handling and retry logic.

2.3. Write unit tests with a mock MAVLink connection that validates the correct messages are sent with correct field values.

2.4. Write SITL integration tests:
  - `test_guided_goto`: Arm → takeoff → switch to GUIDED → send goto → verify vehicle moves toward target within tolerance.
  - `test_roi`: Set ROI → verify `MOUNT_STATUS` or `GIMBAL_DEVICE_ATTITUDE_STATUS` reflects the commanded orientation (in SITL with simulated gimbal).
  - `test_mission_upload`: Upload a 4-waypoint mission → read it back → verify it matches.
  - `test_statustext`: Send a STATUSTEXT → verify it appears in the MAVLink message stream (simulating what QGC would see).
  - `test_mission_interrupt_resume`: Start AUTO mission → interrupt with GUIDED goto → resume AUTO → verify vehicle continues from the correct waypoint.

### Done Test

```
pytest tests/unit/test_commands.py
pytest tests/sitl/test_commands_live.py
```

### Testing Approach

- Unit tests: mock the connection, assert correct MAVLink message types and field values.
- SITL tests: actually fly the simulated vehicle. Use `TelemetryClient` from Phase 1 to verify outcomes. This validates the two modules work together.

---

## Phase 3: Detection Event Interface (`aerie-common/events`)

**Goal**: Define the shared event schema that `aerie-vision` publishes and `aerie-agent` consumes. Implement the pub/sub transport.

### Steps

3.1. Define the `DetectionEvent` dataclass:
```python
@dataclass
class DetectionEvent:
    timestamp: float
    event_id: str           # UUID
    detection_type: str     # e.g., "fire", "person", "vehicle"
    confidence: float       # 0.0 - 1.0
    bbox: tuple[int, int, int, int]  # pixel coords in source frame
    geo: GeoPoint | None    # geocoded ground position (lat, lon, alt)
    frame_index: int
    source_telemetry: VehicleState  # drone state at detection time

@dataclass
class GeoPoint:
    lat: float
    lon: float
    alt: float              # estimated ground elevation
    accuracy_m: float       # estimated position uncertainty radius
```

3.2. Implement transport using ZeroMQ PUB/SUB (lightweight, language-agnostic, no broker needed):
  - `aerie-vision` publishes serialized `DetectionEvent` on a ZMQ PUB socket.
  - `aerie-agent` subscribes on a ZMQ SUB socket.
  - Serialization: JSON for simplicity and debuggability. Msgpack as a future optimization.

3.3. Write a mock publisher (`tests/mocks/mock_vision.py`) that emits fake detection events on a timer. This decouples agent development from the vision pipeline.

3.4. Write tests that validate serialization roundtrips and that the subscriber receives events within latency bounds.

### Done Test

```
pytest tests/unit/test_events.py                — schema validation, serde roundtrip
pytest tests/integration/test_event_pubsub.py   — mock publisher → subscriber receives events
```

### Testing Approach

- Unit: serialize/deserialize edge cases (None geo, boundary confidence values).
- Integration: spin up mock publisher in a subprocess, connect subscriber, assert N events received within T seconds.

---

## Phase 4: Video Ingest + ML Detection (`aerie-vision`)

**Goal**: Receive a video stream, run object detection, geocode detections, and publish `DetectionEvent`s.

### Steps

4.1. **Video ingest** (`aerie-vision/ingest/`):
  - GStreamer pipeline to receive RTSP/UDP video stream.
  - Decode frames, expose as numpy arrays to the detection module.
  - For SITL testing: use a pre-recorded video file or synthetic test frames (a known image with objects at known positions).

4.2. **Detection** (`aerie-vision/detection/`):
  - Define a `Detector` protocol/interface:
    ```python
    class Detector(Protocol):
        def detect(self, frame: np.ndarray) -> list[RawDetection]: ...
    ```
  - Implement `YoloDetector` using ultralytics or ONNX runtime.
  - Implement `MockDetector` that returns hardcoded detections for specific test frames (used in integration tests).

4.3. **Geocoding** (`aerie-vision/geocoding/`):
  - Given a pixel bounding box center, camera intrinsics (FOV, resolution), drone telemetry (position, attitude), and gimbal angle → compute the ground intersection point.
  - This is a ray-casting problem: pixel → camera ray → world ray (using drone attitude + gimbal) → ground plane intersection.
  - Start with flat-earth approximation (sufficient for sub-1km AGL operations). Terrain-aware version is a future enhancement.

4.4. **Pipeline orchestration**: For each frame:
  - Run detector
  - For each detection, look up the closest telemetry snapshot (from the `TelemetryClient` connected to port 14551)
  - Geocode the detection
  - Publish `DetectionEvent` on ZMQ

4.5. Write tests:
  - Geocoding unit tests with known camera geometry and expected ground points.
  - Pipeline integration test: feed a test video with known objects at known frame numbers → assert correct `DetectionEvent`s are published.

### Done Test

```
pytest tests/unit/test_geocoding.py
pytest tests/unit/test_detector.py
pytest tests/integration/test_vision_pipeline.py  — test video → expected detection events
```

### Testing Approach

- Geocoding: pure math, highly unit-testable. Create test fixtures with known camera params, drone position/attitude, pixel coordinates, and expected ground points. Test edge cases (straight-down camera, oblique angles, horizon cases).
- Detection: test the `Detector` interface with a few known images. Don't test YOLO itself — test that your wrapper correctly translates its output.
- Pipeline: use `MockDetector` + a test video + canned telemetry to validate the full pipeline end-to-end.

---

## Phase 5: AI Agent Reasoning Engine (`aerie-agent/reasoning`)

**Goal**: The agent consumes telemetry + detections, uses an LLM to reason about what's happening, and decides on actions.

### Steps

5.1. Define the `AgentAction` type — the set of things the agent can decide to do:
```python
@dataclass
class AgentAction:
    action_type: str        # "alert", "goto", "set_roi", "replan", "resume", "no_op"
    parameters: dict        # action-specific params
    reasoning: str          # LLM's explanation (for logging/audit)
    priority: int           # 0 = informational, 1 = important, 2 = critical
```

5.2. Define the `AgentContext` — the input to each reasoning cycle:
```python
@dataclass
class AgentContext:
    vehicle_state: VehicleState
    recent_detections: list[DetectionEvent]   # last N seconds
    mission_summary: dict                     # waypoints, progress, objective
    standing_orders: list[str]                # user-defined rules/priorities
    action_history: list[AgentAction]         # what the agent has done recently
```

5.3. Implement the reasoning engine:
  - Construct a prompt from `AgentContext` with a structured system prompt defining the agent's role, available actions, and decision criteria.
  - Call the LLM with tool-use / function-calling to select an `AgentAction`.
  - Parse the response into an `AgentAction`.
  - Include a confidence threshold: below the threshold, default to `no_op` and log for human review.

5.4. Implement the `ActionExecutor` that takes an `AgentAction` and calls the appropriate `CommandClient` method(s) from Phase 2, plus dispatches alerts.

5.5. Implement the agent main loop:
```
while running:
    state = telemetry_client.get_state()
    detections = get_recent_detections(detection_subscriber)
    context = build_context(state, detections, mission, orders, history)
    action = reasoning_engine.decide(context)
    action_executor.execute(action)
    history.append(action)
    sleep(agent_cycle_interval)
```

5.6. Write tests:
  - Reasoning unit tests with canned contexts and mocked LLM responses. Validate that specific scenarios produce expected action types.
  - Action executor unit tests: mock `CommandClient`, verify correct methods called for each action type.
  - Scenario tests: combine a mock telemetry stream, mock detection events, and a mock LLM → verify the full agent loop produces the correct sequence of actions.

### Done Test

```
pytest tests/unit/test_reasoning.py
pytest tests/unit/test_action_executor.py
pytest tests/scenarios/test_fire_scenario.py    — fire detection → alert + goto + roi
pytest tests/scenarios/test_no_action.py        — low confidence detection → no_op
```

### Testing Approach

- **Reasoning is the hardest to test deterministically.** Strategy:
  - Mock the LLM to return scripted tool-call responses for specific context fingerprints.
  - Test the prompt construction separately: given a context, assert the prompt contains the expected information.
  - Test the response parsing separately: given an LLM response, assert the correct `AgentAction` is produced.
  - Scenario tests tie it together but still use mocked LLM responses for determinism.
- **Evaluation harness** (post-MVP): Run the real LLM against a bank of scenario contexts and grade outputs. This is non-blocking for initial development.

---

## Phase 6: Alert System (`aerie-agent/alerts`)

**Goal**: When the agent decides to alert humans, deliver notifications through configurable channels.

### Steps

6.1. Define the `Alert` dataclass:
```python
@dataclass
class Alert:
    alert_id: str
    severity: str           # "info", "warning", "critical"
    title: str
    body: str
    detection: DetectionEvent | None
    vehicle_state: VehicleState
    agent_reasoning: str
    timestamp: float
```

6.2. Implement alert dispatchers (each behind a common interface):
  - `WebhookDispatcher`: POST JSON to a configurable URL.
  - `SMSDispatcher`: Send via Twilio (or similar) API.
  - `StatusTextDispatcher`: Send MAVLink STATUSTEXT so it appears in QGC.
  - Future: email, Slack, push notification.

6.3. Implement `AlertManager` that routes alerts to configured dispatchers based on severity rules.

6.4. Write tests: mock HTTP endpoints, verify correct payloads are sent for each severity level.

### Done Test

```
pytest tests/unit/test_alerts.py
```

### Testing Approach

- Mock HTTP servers for webhook/SMS tests.
- Capture MAVLink STATUSTEXT in a pymavlink listener for STATUSTEXT dispatcher tests.

---

## Phase 7: End-to-End Integration

**Goal**: All components running together against SITL, demonstrating the fire scenario end-to-end.

### Steps

7.1. Write a `docker-compose.yml` (or a single orchestration script) that starts:
  - ArduPilot SITL
  - MAVProxy with the correct port fan-out
  - `aerie-vision` (with `MockDetector` injecting a fire detection at a scripted time)
  - `aerie-agent` (with real or mocked LLM)
  - A webhook receiver that logs alerts

7.2. Write a scenario test script that:
  - Uploads a survey mission via pymavlink
  - Arms and launches the vehicle in AUTO mode
  - Waits for the mock fire detection to trigger
  - Asserts:
    - Agent switches vehicle to GUIDED mode
    - Vehicle moves toward the fire coordinates
    - ROI is set
    - STATUSTEXT message appears in the MAVLink stream
    - Webhook receives the alert
    - After a timeout, agent (or test script) resumes AUTO mission
    - Vehicle continues to subsequent waypoints

7.3. Document the manual QGC verification: start the full stack, open QGC on port 14550, visually confirm the scenario plays out on the map.

### Done Test

```
pytest tests/e2e/test_fire_scenario_e2e.py    — fully automated against SITL
Manual: visual confirmation in QGC
```

---

## Phase 8: Hardware Integration (Future)

**Goal**: Transition from SITL to a real ArduPilot vehicle.

This phase is documented for planning but not detailed here, as it depends on specific hardware choices (flight controller, companion computer, camera, telemetry radio). Key considerations:

- Replace SITL connection string with serial/UDP telemetry radio link.
- Replace test video with real GStreamer pipeline from the onboard camera.
- Replace `MockDetector` with the trained YOLO (or custom) model.
- Validate latencies: detection-to-action loop must complete within acceptable bounds for the flight speed and observation altitude.
- Add watchdog: if the agent crashes or loses connection, the drone must continue its mission safely (ArduPilot does this by default — if no GUIDED commands arrive, it holds position).

---

## Summary: Build Order and Dependencies

```
Phase 0: Dev Environment          ← no dependencies
Phase 1: Telemetry Interface      ← Phase 0
Phase 2: Command Interface        ← Phase 0, uses Phase 1 for test verification
Phase 3: Event Schema + PubSub    ← no dependencies (pure Python)
Phase 4: Vision Pipeline          ← Phase 1 (for telemetry), Phase 3 (for events)
Phase 5: Agent Reasoning          ← Phase 1, 2, 3
Phase 6: Alert System             ← Phase 3 (for event types)
Phase 7: E2E Integration          ← all above
Phase 8: Hardware                 ← Phase 7
```

Phases 1-3 can be developed in parallel by different contributors. Phase 4 and 5 can overlap once Phase 3's event schema is stable.
