"""
Biomechanical conventions extracted from OpenBiomechanics Project.
Reference: baseball_hitting/README.md and baseball_pitching/README.md

These conventions define:
- Joint angle sign conventions (what's positive/negative)
- Coordinate systems
- Event definitions
"""

# =============================================================================
# COORDINATE SYSTEM (Hitting - from baseball_hitting/README.md)
# =============================================================================
# Global (laboratory) reference frame:
#   (+) X = toward pitcher (from home plate to mound)
#   (+) Y = toward right-handed batter's box (posterior for RHH)
#   (+) Z = upward

# =============================================================================
# JOINT ANGLE CONVENTIONS (Hitting)
# =============================================================================
# Format: Joint: (X component, Y component, Z component)
#
# These follow the convention where:
#   - First rotation (X) is typically flexion/extension
#   - Second rotation (Y) is typically ab/adduction or lateral
#   - Third rotation (Z) is typically rotation

HITTING_JOINT_CONVENTIONS = {
    'wrist': {
        'x': ('Flexion (-)', 'Extension (+)'),
        'y': ('Ulnar (-)', 'Radial (+)', 'Deviation'),
        'z': 'Constrained'
    },
    'elbow': {
        'x': ('Flexion (+)', 'Extension (-)'),
        'y': 'Constrained',
        'z': ('Pronation (+)', 'Supination (-)')
    },
    'shoulder': {
        'x': ('Horizontal Abduction (+)', 'Horizontal Adduction (-)'),
        'y': ('Abduction (+)', 'Adduction (-)'),
        'z': ('External Rotation (+)', 'Internal Rotation (-)')
    },
    'pelvis': {
        'x': ('Posterior Tilt (+)', 'Anterior Tilt (-)'),
        'y': ('Lateral Tilt Toward Rear Leg (+)', 'Lateral Tilt Away (-)'),
        'z': ('Rotation Toward Mound (+)', 'Rotation Away (-)')
    },
    'torso': {
        'x': ('Extension (+)', 'Flexion (-)'),
        'y': ('Lateral Tilt Toward Rear Leg (+)', 'Lateral Tilt Away (-)'),
        'z': ('Rotation Toward Mound (+)', 'Rotation Away (-)')
    },
    'torso_pelvis': {  # X-factor / hip-shoulder separation
        'x': ('Extension (+)', 'Flexion (-)'),
        'y': ('Lateral Tilt Toward Rear Leg (+)', 'Lateral Tilt Away (-)'),
        'z': ('Torso Toward Mound / Pelvis Toward Catcher (+)',
              'Torso Toward Catcher / Pelvis Toward Mound (-)')
    },
    'hip': {
        'x': ('Flexion (+)', 'Extension (-)'),
        'y': ('Abduction (+)', 'Adduction (-)'),
        'z': ('Internal Rotation (+)', 'External Rotation (-)')
    },
    'knee': {
        'x': ('Flexion (+)', 'Extension (-)'),
        'y': 'Constrained',
        'z': 'Constrained'
    },
    'ankle': {
        'x': ('Dorsiflexion (+)', 'Plantarflexion (-)'),
        'y': ('Eversion (+)', 'Inversion (-)'),
        'z': ('Lateral Rotation (+)', 'Medial Rotation (-)')
    }
}

# =============================================================================
# HITTING EVENTS
# =============================================================================
# Key events in a baseball swing, used for point-of-interest (POI) extraction

class HittingEvents:
    """Definitions of key events in a baseball swing."""

    # First Move (FM): Initial movement toward the pitch
    FIRST_MOVE = 'fm'

    # Load / Loaded Position: Maximum coil/separation before stride
    LOAD = 'load'

    # Heel Strike (HS): Front heel contacts ground
    HEEL_STRIKE = 'hs'

    # Foot Plant (FP): Front foot fully planted
    # In OBP data: 10% bodyweight = foot contact, 100% bodyweight = foot plant
    FOOT_PLANT = 'fp'

    # Maximum Hip-Shoulder Separation
    MAX_HIP_SHOULDER_SEP = 'maxhss'

    # Down Swing (DS): Start of forward bat movement
    DOWN_SWING = 'ds'

    # Contact: Bat contacts ball
    CONTACT = 'contact'


# =============================================================================
# PITCHING EVENTS (for reference)
# =============================================================================

class PitchingEvents:
    """Definitions of key events in a pitch."""

    # Peak Knee Height
    PEAK_KNEE_HEIGHT = 'pkh'

    # Foot Contact (10% bodyweight)
    FOOT_CONTACT = 'fc'

    # Foot Plant (100% bodyweight)
    FOOT_PLANT = 'fp'

    # Maximum External Rotation (layback)
    MAX_EXTERNAL_ROTATION = 'mer'

    # Ball Release
    BALL_RELEASE = 'br'

    # Maximum Internal Rotation
    MAX_INTERNAL_ROTATION = 'mir'


# =============================================================================
# HANDEDNESS MAPPING
# =============================================================================
# For hitting, need to map left/right body sides to lead/rear leg concepts

def get_side_mapping(bats: str) -> dict:
    """
    Get mapping from anatomical sides to functional sides based on handedness.

    Args:
        bats: 'L' for left-handed hitter, 'R' for right-handed hitter

    Returns:
        Dictionary mapping 'lead'/'rear' to 'left'/'right'
    """
    if bats.upper() == 'R':
        return {
            'lead_leg': 'left',
            'rear_leg': 'right',
            'lead_arm': 'left',
            'rear_arm': 'right'
        }
    else:  # Left-handed
        return {
            'lead_leg': 'right',
            'rear_leg': 'left',
            'lead_arm': 'right',
            'rear_arm': 'left'
        }


# =============================================================================
# UNITS
# =============================================================================

class Units:
    """Standard units used in biomechanical analysis."""

    # Angles
    ANGLE_DEG = 'deg'
    ANGLE_RAD = 'rad'

    # Angular velocity
    ANGULAR_VELOCITY = 'deg/s'

    # Linear velocity
    LINEAR_VELOCITY_MS = 'm/s'
    LINEAR_VELOCITY_MPH = 'mph'

    # Force
    FORCE_N = 'N'
    FORCE_N_PER_KG = 'N/kg'

    # Moment/Torque
    MOMENT_NM = 'Nm'

    # Length
    LENGTH_M = 'm'
    LENGTH_IN = 'in'

    # Mass
    MASS_KG = 'kg'
    MASS_LBS = 'lbs'


# Conversion factors
def mph_to_ms(mph: float) -> float:
    """Convert miles per hour to meters per second."""
    return mph * 0.44704

def ms_to_mph(ms: float) -> float:
    """Convert meters per second to miles per hour."""
    return ms / 0.44704

def deg_to_rad(deg: float) -> float:
    """Convert degrees to radians."""
    import math
    return deg * math.pi / 180.0

def rad_to_deg(rad: float) -> float:
    """Convert radians to degrees."""
    import math
    return rad * 180.0 / math.pi

def inches_to_meters(inches: float) -> float:
    """Convert inches to meters."""
    return inches * 0.0254

def lbs_to_kg(lbs: float) -> float:
    """Convert pounds to kilograms."""
    return lbs * 0.453592
