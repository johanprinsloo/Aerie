# Aerie
High-level semi-autonomous drone supervison with AI agent assist

**Autonomous observation platform for ArduPilot drones.**

Aerie is a modular base station system that combines mission planning, real-time video analysis with ML object detection, and an AI agent that can evaluate telemetry, interpret detections, alert humans, and autonomously modify drone behavior in response to what it sees.

## The Problem

Existing ground control stations (QGroundControl, Mission Planner) are excellent at mission planning and manual flight monitoring, but they have no native support for:

- Real-time ML inference on video feeds with detection-to-action pipelines
- An AI agent that can reason about detections, telemetry, and mission context to make autonomous decisions
- Programmatic mission modification in response to detected events (e.g., re-routing a drone to observe a fire)

Bolting these capabilities onto a C++/Qt or C#/.NET GCS means maintaining a heavy fork against an upstream you don't control.

## Use Case: Building Inspection with Autonomous Fire Detection

A property management company operates a drone to perform routine visual inspections of a large commercial building complex. An operator at the base station plans a survey mission in QGroundControl — a series of waypoints that fly the drone in a grid pattern over the rooftops and along the building facades, capturing video throughout.

The operator launches the mission. The drone takes off and begins its planned route. On the base station, Aerie's vision pipeline is processing the live video feed in real time, running a detection model trained to recognize structural damage, water pooling, HVAC anomalies, and fire or smoke.

Midway through the survey, as the drone passes a section of rooftop near waypoint 5 of 12, the vision pipeline detects what appears to be a small fire near an HVAC unit — smoke and visible flame, confidence 94%. Aerie geocodes the detection: the fire is at a specific latitude and longitude on the southeast corner of Building C's roof.

Aerie's AI agent receives this detection along with the current mission context. It evaluates the situation: a fire is an immediate safety concern, high confidence, and the drone is about to fly away from it on its planned route. The agent decides this is critical and takes several actions within seconds:

- It sends an alert to the facility manager and the operator via webhook and SMS, including the geocoded location, a description of the detection, and a frame capture.
- It commands the drone to leave its planned route and reposition to a loiter point with a clear view of the fire.
- It sets a Region of Interest on the fire's coordinates so the gimbal keeps the camera pointed at it.
- It sends a status message that appears in QGroundControl's message feed: *"AERIE: Fire detected at Building C SE roof — observing."*

The operator in QGroundControl sees the drone depart its planned path and begin loitering. The original mission is still loaded — the drone hasn't forgotten it. The operator can watch the live video to confirm the fire, take manual control if needed, or let the agent continue observing while emergency services respond.

Twenty minutes later, the situation is handled. The operator tells the agent to "Resume Mission" in QGroundControl (or the agent does so autonomously based on a configured timeout). The drone returns to waypoint 6 and continues the survey as if nothing happened — except now there's a logged, geocoded detection event with video, telemetry, and the agent's reasoning for the decision, all available for post-flight review.

Without Aerie, the operator would have had to watch the video feed continuously and react manually — or worse, review the footage after landing and discover the fire too late.

With Aerie the operator does not need to be at the base station at all, they can communicate with the agent over Text or Voice on their phone.

## Architecture

Aerie doesn't replace your GCS — it runs alongside it. The key insight is that ArduPilot's autopilot is the single source of truth, and any number of MAVLink peers can observe and command it simultaneously. QGroundControl (or Mission Planner) remains the human operator's interface; Aerie adds the intelligence layer.

The drone has two independent downlinks to the base station:

- **MAVLink telemetry/command link** — Low-bandwidth, bidirectional. Carries vehicle state, GPS, mission progress, and accepts commands. Typically over a SiK radio, Herelink, or IP radio.
- **Digital video link** — High-bandwidth, typically unidirectional (drone → ground). Carries the camera feed. Received at the base station via HDMI framegrabber, RTSP/UDP stream, or proprietary receiver (DJI, Herelink, etc.). This path is completely out-of-band from MAVLink.


```
                              ┌──────────────────────────────────┐
                              │           DRONE                  │
                              │                                  │
                              │  ┌──────────┐    ┌────────────┐  │
                              │  │ ArduPilot│    │   Camera / │  │
                              │  │    FC    │    │   Gimbal   │  │
                              │  └────┬─────┘    └─────┬──────┘  │
                              └───────┼────────────────┼─────────┘
                                      │                │
                           MAVLink    │                │  Video
                        (telemetry +  │                │  (HDMI/IP/
                          commands)   │                │   proprietary)
                                      │                │
                              ┌───────┴──┐        ┌────┴─────────┐
                              │Telemetry │        │  Video Rx /  │
                              │  Radio   │        │ Framegrabber │
                              └───────┬──┘        └────┬─────────┘
                                      │                │
            ══════════════════════════════════════════════════════════════════════
                                          BASE STATION
            ══════════════════════════════════════════════════════════════════════
                                      │                │
                             ┌────────┴──────────┐     │
                             │    MAVProxy /     │     │
                             │  mavlink-router   │     │
                             │  (message router) │     │
                             └──┬─────┬──────┬───┘     │
                                │     │      │         │
               ┌────────────────┘     │      │         │
               │                      │      │         │
               ▼                      │      ▼         ▼
      ┌──────────────┐                │  ┌────────────────┐
      │    QGC /     │                │  │  Video + ML    │
      │   Mission    │                │  │  Pipeline      │
      │   Planner    │                │  │ (aerie-vision) │
      │              │                │  │                │
      │  (human      │  telemetry     │  │ Video in ──►   │
      │   operator)  │  for geocoding │  │ Detection ──►  │
      └──────────────┘                │  │ Geocoding ──►  │
                                      │  │ Events out     │
                                      │  └──────┬─────────┘
                                      │         │
                                      ▼         │ detections
                                ┌────────────────┐
                                │   AI Agent     │
                                │ (aerie-agent)  │
                                │                │
                                │ Consumes:      │
                                │ - telemetry    │
                                │ - detections   │
                                │                │
                                │ Emits:         │
                                │ - MAV commands │
                                │ - alerts       │
                                │ - STATUSTEXT   │
                                └────────────────┘
```


`aerie-vision` receives **two independent inputs**: the video stream directly from the video receiver, and a MAVLink telemetry feed from MAVProxy (used to correlate each video frame with the drone's position, altitude, and gimbal angle for geocoding detections). It does not send or receive drone commands — it is a pure observation pipeline.

`aerie-agent` also receives **two inputs**: the MAVLink telemetry stream and the detection events published by `aerie-vision`. Unlike the vision pipeline, the agent sends commands back through MAVProxy to the autopilot.


### Component Responsibilities

**MAVProxy (Message Router)** — Connects to the flight controller and fans out MAVLink UDP streams to all consumers. Every component receives the same telemetry. Every component can send commands back through the router to the autopilot.

**Video + ML Pipeline (`aerie-vision`)** — Receives the drone's video stream (typically via GStreamer/RTSP), runs object detection inference (YOLO or a custom model), geocodes detections using synchronized telemetry (drone position, gimbal angle, camera FOV → ground coordinates), and publishes structured detection events.

**AI Agent (`aerie-agent`)** — The decision-making core. Subscribes to both the telemetry stream (via pymavlink) and the detection event stream from `aerie-vision`. Uses an LLM to reason about what's happening, decide whether action is needed, and execute that action — which can include alerting humans, re-routing the drone, or modifying mission parameters.

**QGroundControl / Mission Planner (Human Interface)** — Unchanged, off-the-shelf. The operator sees everything the agent does because all state changes flow through ArduPilot's MAVLink bus. The operator can override the agent at any time — their commands are just more MAVLink messages, and the last command wins.

### How the Agent Controls the Drone

The agent has three main control mechanisms, all via standard MAVLink:

**GUIDED mode commands (immediate reaction)** — The agent sets the flight mode to GUIDED, then sends position/velocity targets. The drone responds immediately. The original AUTO mission remains loaded in the autopilot and can be resumed. This is the right tool for "something urgent happened, react now."

**Mission rewrite (replanning)** — The agent uploads a new mission or modifies the current one using the MAVLink mission protocol (`MISSION_COUNT` → `MISSION_REQUEST_INT` → `MISSION_ITEM_INT` → `MISSION_ACK`). QGC picks this up and displays the new plan. Best for "the rest of this route needs to change."

**Region of Interest (camera pointing)** — `MAV_CMD_DO_SET_ROI` tells the autopilot to aim the gimbal at a specific ground coordinate. Can be combined with either approach above. Best for "keep watching that spot."

### Example Scenario: Fire Detection

```
1. Drone is executing AUTO mission, waypoint 5 of 12
2. aerie-vision detects fire (confidence 0.94) at video frame T
3. aerie-vision geocodes detection → {type: "fire", lat: 47.123, lon: -122.456}
4. aerie-agent receives detection event
5. Agent evaluates: fire + high confidence + mission context → critical event
6. Agent actions:
   a. Sends alert (webhook/SMS) to designated humans
   b. pymavlink → MAV_CMD_DO_SET_MODE: GUIDED
   c. pymavlink → SET_POSITION_TARGET_GLOBAL_INT: loiter near fire
   d. pymavlink → MAV_CMD_DO_SET_ROI: fire coordinates (camera tracks it)
   e. pymavlink → STATUSTEXT: "AERIE: Fire detected at 47.123,-122.456 — observing"
7. QGC shows: drone departing mission path, loitering, STATUSTEXT in message feed
8. Operator acknowledges, can take manual control or let agent continue
9. When resolved: agent or operator → MAV_CMD_DO_SET_MODE: AUTO → mission resumes
```

## Project Structure

```
aerie/
├── aerie-router/        # MAVProxy configuration and startup scripts
├── aerie-vision/        # Video ingest + ML detection pipeline
│   ├── ingest/          # GStreamer video capture
│   ├── detection/       # Model inference (YOLO, custom)
│   ├── geocoding/       # Detection → ground coordinates
│   └── events/          # Detection event publishing
├── aerie-agent/         # AI decision-making agent
│   ├── telemetry/       # pymavlink telemetry consumer
│   ├── reasoning/       # LLM-based decision engine
│   ├── actions/         # MAVLink command execution
│   └── alerts/          # Human notification (webhook, SMS)
├── aerie-common/        # Shared types, config, logging
├── tests/
│   ├── sitl/            # ArduPilot SITL integration tests
│   ├── unit/            # Per-component unit tests
│   └── scenarios/       # End-to-end scenario tests
└── docs/
```

## Requirements

- Python 3.11+
- ArduPilot SITL (for development/testing)
- MAVProxy
- GStreamer 1.x
- pymavlink
- An ML model (YOLO recommended for starting, swappable)
- LLM API access (for the agent reasoning engine)

## Development

All development and testing uses ArduPilot SITL (Software In The Loop), which simulates a complete autopilot. No hardware needed until integration testing.

```bash
# Start SITL
sim_vehicle.py -v ArduCopter --map --console

# Start MAVProxy router (fans out to QGC + Aerie components)
mavproxy.py --master=tcp:127.0.0.1:5760 \
  --out=udp:127.0.0.1:14550 \   # QGC
  --out=udp:127.0.0.1:14551 \   # aerie-vision (telemetry for geocoding)
  --out=udp:127.0.0.1:14552      # aerie-agent
```

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the step-by-step build plan.

## License

TBD

## Status

Pre-alpha. Architecture defined, implementation in progress.

