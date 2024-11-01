"""Configuration settings for the Jupiter Project."""

# AWS S3 Settings
S3_BUCKET = "astro-data-io"
INPUT_PREFIX = "voyager_all/"
PNG_PREFIX = "png/"
OUTPUT_PREFIX = "sorted_png/"

# Processing Settings
TEMP_DIR = "/tmp/voyager_processing"
PARALLEL_WORKERS = 4
META_FILE_PATH = '/home/ubuntu/jupiter_project/meta_file.csv'

# Detection Parameters
BRIGHTNESS_THRESHOLD = 75
MIN_SIZE = 20
MAX_SIZE = 39550  # Maximum size in pixels for detected objects
CIRCULARITY_THRESHOLD = 0.7

# Visualization Settings
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SIZE = 30
VIDEO_FPS = 3
VIDEO_BITRATE = "2000k"

# Visualization Colors (RGB)
COLORS = {
    'BOX_COLOR': (0, 255, 0),      # Green
    'CENTER_COLOR': (255, 0, 0),    # Red
    'TEXT_COLOR': (255, 255, 0),    # Yellow
    'OVERLAY_TEXT': (255, 255, 255) # White
}

# Visualization Styles
VIS_STYLES = {
    'BOX_THICKNESS': 2,
    'CENTER_RADIUS': 3,
    'TEXT_SCALE': 0.5,
    'TEXT_THICKNESS': 1
}

# Major celestial targets
MAJOR_TARGETS = [
    'CALLISTO',
    'EUROPA',
    'GANYMEDE',
    'IO',
    'TITAN',
    'JUPITER',
    'SATURN',
    'URANUS',
    'NEPTUNE'
]

# Existing videos to skip
EXISTING_VIDEOS = {
    'CALLISTO',
    'IO',
    'MIRANDA',
    'TITAN',
    'EUROPA'
}

# All available targets
ALL_TARGETS = [
    'ADRASTEA', 'AMALTHEA', 'ARCTURUS', 'ARIEL', 'BETACMA', 'CAL LAMPS',
    'CALYPSO', 'DARK', 'DIONE', 'ENCELADUS', 'EPIMETHEUS', 'GANYMEDE',
    'HELENE', 'HYPERION', 'IAPETUS', 'J RINGS', 'JANUS', 'JUPITER',
    'METIS', 'MIMAS', 'N RINGS', 'NEPTUNE', 'NEREID', 'OBERON', 'ORION',
    'PANDORA', 'PHOEBE', 'PLAQUE', 'PLEIADES', 'PROMETHEUS', 'PROTEUS',
    'PUCK', 'RHEA', 'S RINGS', 'SATURN', 'SCORPIUS', 'SIGMA SGR', 'SKY',
    'STAR', 'SYSTEM', 'TAURUS', 'TELESTO', 'TETHYS', 'THEBE', 'THETA CAR',
    'TITANIA', 'TRITON', 'U RINGS', 'UMBRIEL', 'UNK SAT', 'URANUS', 'VEGA'
]
