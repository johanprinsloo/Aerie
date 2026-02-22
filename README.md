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


### The AI Agent in Detail

The Aerie agent is not a request-response service — it is a **persistent, always-on autonomous process** that runs continuously alongside the drone operation. It observes, remembers, reasons, and acts across the full lifecycle of a mission, and it retains knowledge across missions over time.

#### Runtime: ZeroClaw

The agent runs on [ZeroClaw](https://github.com/zeroclaw-labs/zeroclaw), a lightweight Rust-based autonomous agent runtime. ZeroClaw provides the infrastructure that makes the agent persistent and capable:

- **Daemon process**: Runs as an always-on system service with automatic restart on failure. The agent is alive before the drone launches and remains alive after it lands.
- **Persistent memory**: ZeroClaw's built-in memory system (backed by SQLite) gives the agent durable, searchable recall across sessions. The agent remembers what it has seen on previous missions, what decisions it made and why, what the operator corrected, and what standing orders it has been given. Memory is stored as human-readable files and database entries — fully auditable and version-controllable.
- **Learning through memory**: Over time, the agent accumulates operational knowledge. It learns that a particular building always has HVAC heat signatures that are not fires. It learns that the operator prefers to be alerted about water pooling even at lower confidence. It learns the flight characteristics of the specific airframe. This is not model fine-tuning — it is persistent context that the agent retrieves and injects into its reasoning on every decision cycle.
- **Operator communication channels**: ZeroClaw has native Telegram, Discord, Slack, and WhatsApp integrations. The operator communicates with the agent over their preferred messaging app — receiving alerts, asking about mission status, issuing commands ("resume mission", "return to base", "ignore that detection"), and reviewing the agent's reasoning. The operator does not need to be at the base station.
- **LLM provider flexibility**: ZeroClaw supports 22+ AI providers (OpenAI, Anthropic, Ollama, etc.) and can be switched between them via configuration. For field operations with limited connectivity, a local model via Ollama can run on the base station hardware. For full capability, cloud APIs provide the strongest reasoning.

#### Agent Tools: The pymavlink Bridge

ZeroClaw executes actions through **tools** — sandboxed commands the LLM can invoke during its reasoning loop. Aerie exposes drone control as a set of Python scripts that wrap pymavlink, registered as ZeroClaw tools:

```
aerie-agent/
├── tools/
│   ├── drone_set_mode.py       # Set flight mode (GUIDED, AUTO, RTL, LOITER)
│   ├── drone_goto.py           # Fly to lat/lon/alt in GUIDED mode
│   ├── drone_set_roi.py        # Point camera at ground coordinate
│   ├── drone_upload_mission.py # Upload/replace mission waypoints
│   ├── drone_resume_mission.py # Resume AUTO mission from current waypoint
│   ├── drone_get_state.py      # Read current telemetry snapshot
│   └── drone_send_status.py    # Send STATUSTEXT visible in QGC
├── identity/
│   ├── IDENTITY.md             # Agent persona and operational role
│   └── SOUL.md                 # Core behavioral constraints and safety rules
├── memory/
│   └── ...                     # Persistent memory (auto-managed by ZeroClaw)
└── config.toml                 # ZeroClaw configuration
```

Each tool is a thin Python script that connects to the MAVProxy UDP endpoint, executes a single MAVLink action, and returns a structured result. ZeroClaw invokes them as sandboxed shell commands and feeds the output back into the agent's reasoning context.

Example tool invocation in the agent's reasoning loop:

```
Agent thinks: "Fire detected with 94% confidence. This is critical. I need to
               redirect the drone and alert the operator."

Agent calls:  drone_set_mode --mode GUIDED
              → "Mode changed to GUIDED successfully"

Agent calls:  drone_goto --lat 47.123 --lon -122.456 --alt 45
              → "Navigating to 47.123, -122.456 at 45m"

Agent calls:  drone_set_roi --lat 47.123 --lon -122.456 --alt 0
              → "ROI set, gimbal tracking 47.123, -122.456"

Agent sends Telegram message to operator:
              "🔥 Fire detected on Building C (SE roof) with 94% confidence.
               Drone redirected to observe. Live position: [map link]"
```

#### Detection Event Ingestion

`aerie-vision` delivers detection events to the agent via ZeroClaw's HTTP gateway. When the vision pipeline geocodes a detection, it POSTs a JSON payload to the agent's gateway endpoint:

```
POST http://localhost:3000/webhook/aerie-vision
{
  "type": "fire",
  "confidence": 0.94,
  "lat": 47.123,
  "lon": -122.456,
  "timestamp": "2026-02-21T14:32:01Z",
  "frame_index": 8847
}
```

ZeroClaw routes this into the agent's reasoning loop as an incoming event, alongside the ongoing telemetry context from memory.

#### Where Does the Agent Run?

The agent runtime is decoupled from the base station hardware. Three deployment models are supported:

**Local (base station machine)** — ZeroClaw runs on the same machine as MAVProxy and `aerie-vision`. Lowest latency for drone commands. Works fully offline if using a local LLM via Ollama. This is the default for field operations.

```
┌─────────────────────────────────────────────────┐
│                 BASE STATION                    │
│                                                 │
│  MAVProxy ──► aerie-vision ──► ZeroClaw Agent   │
│     ▲                              │            │
│     └──────── drone commands ──────┘            │
│                                                 │
│  LLM: local (Ollama) or cloud API               │
└─────────────────────────────────────────────────┘
```

**Remote (cloud/VPS)** — ZeroClaw runs on a remote server. The base station forwards telemetry and detection events over the network (via ZeroClaw's tunnel support with Tailscale or Cloudflare). Agent commands route back to MAVProxy on the base station. More compute available for the LLM, but adds network latency and requires connectivity.

```
┌────────────────────────┐        ┌──────────────────────┐
│     BASE STATION       │        │     REMOTE SERVER    │
│                        │  net   │                      │
│  MAVProxy ◄────────────┼───────►│  ZeroClaw Agent      │
│  aerie-vision ─────────┼───────►│  LLM: cloud API      │
│                        │        │                      │
└────────────────────────┘        └──────────────────────┘
```

**Hybrid** — ZeroClaw runs locally for minimum command latency, but uses a cloud LLM API for reasoning. This gives fast drone control with strong AI capability, at the cost of requiring internet for the reasoning step (but not for pre-programmed safety behaviors).

```
┌─────────────────────────────────────────────────┐
│                 BASE STATION                    │
│                                                 │
│  MAVProxy ──► aerie-vision ──► ZeroClaw Agent   │
│     ▲                              │      │     │
│     └──────── drone commands ──────┘      │     │
│                                           │     │
└───────────────────────────────────────────┼─────┘
                                            │ LLM API calls
                                     ┌──────┴───────┐
                                     │  Cloud LLM   │
                                     │  (Anthropic, │
                                     │   OpenAI)    │
                                     └──────────────┘
```

The hybrid model is recommended for most operations — it keeps the safety-critical command path local while leveraging the best available reasoning models.

#### Agent Identity and Safety

ZeroClaw's identity system (`IDENTITY.md` and `SOUL.md`) defines the agent's operational role and hard constraints:

- The agent always prioritizes human safety and drone safety over mission completion.
- The agent will never fly the drone into a geofence violation or below minimum altitude.
- The agent defaults to conservative action (loiter, alert, wait for human) when uncertain.
- The agent logs every decision with its reasoning, creating an audit trail.
- The operator can always override the agent — operator commands take absolute priority.

These constraints are loaded into every reasoning cycle as part of the system prompt, independent of the LLM provider.

#### Agent Memory: What It Learns

The agent's memory accumulates across missions:

| Memory Type | Example | How It's Used |
|---|---|---|
| **Mission history** | "Building C inspection on Feb 14 found water pooling on north roof" | Contextualizes current observations against past findings |
| **Operator preferences** | "Operator wants alerts for water pooling even at 60% confidence" | Adjusts decision thresholds per operator feedback |
| **Environmental knowledge** | "HVAC Unit 7 on Building C always reads as thermal anomaly — not a fire" | Reduces false positives over time |
| **Decision outcomes** | "Redirected for suspected fire on Jan 20 — operator confirmed false alarm from steam vent" | Calibrates future decisions |
| **Standing orders** | "Always complete full survey grid before RTL unless critical event" | Shapes mission-level behavior |

Memory is searchable via hybrid vector + keyword retrieval. On each decision cycle, the agent automatically recalls relevant memories based on the current context (detection type, location, mission phase, etc.).

#### Memory Backup, Sync, and Cloning with agent-life

Agent memory is a compounding asset — months of learned behavior, calibrated thresholds, and site-specific knowledge that becomes more valuable with every mission and every LLM upgrade. Losing it to a disk failure, a framework migration, or a corrupted update is operationally costly. Keeping it locked to a single deployment site is a missed opportunity.

Aerie uses [agent-life](https://agent-life.ai) to manage agent memory as portable, protected infrastructure:

**Backup and disaster recovery** — agent-life incrementally syncs the agent's complete state (memory, identity, workspace artifacts, encrypted credentials) to a neutral format. Every sync creates an immutable restore point. If the base station fails, the agent can be restored to any moment in its history on replacement hardware.

**Cloning learned behavior across sites** — This is where agent-life becomes especially powerful for Aerie. An agent that has spent months inspecting Building Complex A has accumulated valuable knowledge: which thermal signatures are false positives, what confidence thresholds the operator trusts, how specific building features appear from the air, and what environmental conditions affect detection quality. When the operator begins inspection operations at a new Building Complex B, they don't start from zero. They **clone** the Site A agent's memory into a new agent instance for Site B. The new agent inherits all the general operational knowledge — detection calibration, operator preferences, mission patterns — while accumulating its own site-specific memories as it flies.

**Framework portability** — agent-life uses an open neutral format with per-framework adapters. If ZeroClaw is ever superseded by a better agent runtime, the agent's accumulated knowledge migrates with it rather than being rebuilt from scratch. Export from ZeroClaw, import to the next runtime — one command, no manual reconstruction.

**Credential protection** — API keys and tokens are encrypted locally before leaving the base station. The sync service stores only ciphertext it cannot decrypt, using a zero-knowledge architecture.

The cloning workflow:

```
Site A agent (6 months of inspections)
    │
    ▼
agent-life export → neutral format snapshot
    │
    ├──► agent-life import → Site B agent (inherits learned behavior)
    ├──► agent-life import → Site C agent (same knowledge, new site)
    └──► agent-life import → Training/test agent (for SITL validation)
```

Over time, this creates a **fleet-wide learning loop**: insights from any site can be reviewed, curated, and distributed to agents operating elsewhere. A false-positive pattern discovered at one site immediately benefits every other deployment.



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

