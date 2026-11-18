"""All tunable values for Predator-Pod live here."""

# --- Window -----------------------------------------------------------------
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 640
FPS = 60
TITLE = "Predator-Pod"

# --- Arena (bordered play area inside the window) ---------------------------
ARENA_X = 30
ARENA_Y = 70           # leaves a strip at the top for the HUD
ARENA_WIDTH = 900
ARENA_HEIGHT = 540
ARENA_BORDER = 3

# --- Fish -------------------------------------------------------------------
FISH_MIN_COUNT = 10
FISH_MAX_COUNT = 30
FISH_DEFAULT_COUNT = 20
FISH_RADIUS = 7                # body radius used for collisions
FISH_SPEED = 120.0             # px/s, constant for every fish
FISH_TURN_RATE = 200.0         # deg/s, how fast a fish can change heading
# Fast-start (C-start escape reflex): instant turn + short burst, then a cooldown.
FISH_BURST_FACTOR = 2.5        # x FISH_SPEED while bursting
FISH_BURST_TIME = 0.35         # s
FISH_BURST_COOLDOWN = 3.0      # s before the next burst (muscle fatigue)

# --- Shark ------------------------------------------------------------------
SHARK_RADIUS = 22              # ~3x the fish radius, never grows
SHARK_SPEED = 200.0            # px/s, cruising speed (it never stops swimming)
SHARK_MIN_SPEED_FACTOR = 0.45  # slows to 45% while turning hard -> tighter turns
SHARK_TURN_RATE = 300.0        # deg/s, max turn speed
SHARK_STEER_GAIN = 5.0         # turn speed per degree of error (smooth homing)
SHARK_STEER_SMOOTHING = 10.0   # how fast turn speed and swim speed ease in (1/s)
SHARK_MOUSE_MOVE_EPS = 2.0     # px, cursor movement that counts as a new command

# --- Physics ----------------------------------------------------------------
OVERLAP_PASSES = 3             # overlap resolution iterations per frame
MAX_DT = 0.05                  # clamp frame time to avoid huge jumps

# --- Colors -----------------------------------------------------------------
COLOR_BG = (6, 7, 10)
COLOR_ARENA = (13, 15, 20)
COLOR_BORDER = (70, 78, 92)
COLOR_FISH = (240, 242, 245)
COLOR_FISH_TAIL = (175, 180, 188)
COLOR_SHARK = (250, 250, 252)
COLOR_SHARK_BELLY = (200, 205, 212)
COLOR_EYE = (10, 10, 12)
COLOR_TEXT = (230, 233, 238)
COLOR_TEXT_DIM = (120, 128, 140)
COLOR_BUTTON = (28, 32, 40)
COLOR_BUTTON_HOVER = (48, 54, 66)
COLOR_ACCENT = (240, 242, 245)
