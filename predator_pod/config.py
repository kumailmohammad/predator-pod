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
FISH_TURN_RATE = 240.0         # deg/s, how fast a fish can change heading
FISH_WALL_MARGIN = 50.0        # px, start steering away from walls inside this

# --- Shark ------------------------------------------------------------------
SHARK_RADIUS = 32              # ~4.5x the fish radius, never grows
SHARK_MAX_SPEED = 200.0        # px/s, shark chases the cursor at this speed
SHARK_TURN_RATE = 360.0        # deg/s, visual heading smoothing

# --- Physics ----------------------------------------------------------------
OVERLAP_PASSES = 3             # overlap resolution iterations per frame
MAX_DT = 0.05                  # clamp frame time to avoid huge jumps

# --- Colors -----------------------------------------------------------------
COLOR_BG = (14, 42, 71)
COLOR_ARENA = (22, 74, 115)
COLOR_BORDER = (200, 225, 240)
COLOR_FISH = (255, 196, 61)
COLOR_FISH_TAIL = (232, 150, 40)
COLOR_SHARK = (130, 145, 160)
COLOR_SHARK_BELLY = (205, 215, 222)
COLOR_EYE = (20, 20, 25)
COLOR_TEXT = (235, 242, 248)
COLOR_TEXT_DIM = (150, 175, 195)
COLOR_BUTTON = (36, 104, 156)
COLOR_BUTTON_HOVER = (52, 136, 196)
COLOR_ACCENT = (255, 196, 61)
