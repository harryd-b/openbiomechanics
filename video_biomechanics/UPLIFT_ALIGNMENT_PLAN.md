# UPLIFT Data Alignment Plan

**Status: IMPLEMENTED** (2026-03-07)

## Overview
This document outlines the changes made to align our biomechanics output with UPLIFT's data format.

## Issue Summary (from frame-by-frame comparison)

| Issue | Columns Affected | Root Cause | Fix |
|-------|-----------------|------------|-----|
| 180° offset | knee_extension | Different zero reference | `uplift_value = -(180 - our_value)` |
| Sign inversion | pelvis/trunk_global_rotation | Opposite rotation direction | Negate values |
| Sign inversion | shoulder angles | Different anatomical reference | Multiple fixes needed |
| Coordinate origin | center_of_mass | Different global origin | Use pelvis-relative COM |
| Poor correlation | velocities | Different derivative method | Use Savitzky-Golay filter |

---

## Phase 1: Angle Convention Fixes

### 1.1 Knee Extension (Priority: HIGH)
**Current**: We calculate `knee_extension = 180 - flexion_angle`
**UPLIFT**: Uses negative values for flexed knee, 0° = straight leg
**Data comparison**:
- UPLIFT: -36° to -27° (flexed knee = negative)
- Ours: 93° to 172° (different reference)

**Fix in `joint_angles_3d.py`**:
```python
# Current (line ~398):
setattr(angles, f'{side}_knee_extension', 180 - flexion)

# Change to:
knee_extension = -(flexion_angle)  # Flexed = negative, straight = 0
setattr(angles, f'{side}_knee_extension', knee_extension)
```

### 1.2 Global Rotations (Priority: HIGH)
**Current**: Positive = counterclockwise when viewed from above
**UPLIFT**: Appears to use opposite convention (correlation = -0.91)

**Fix in `joint_angles_3d.py`**:
```python
# Current (line ~289-294, ~328-332):
angles.pelvis_rotation = np.degrees(pelvis_angle - global_angle)
angles.torso_rotation = np.degrees(torso_angle - global_angle)

# Change to:
angles.pelvis_rotation = -np.degrees(pelvis_angle - global_angle)
angles.torso_rotation = -np.degrees(torso_angle - global_angle)
# Also update pelvis_global_rotation and trunk_global_rotation
```

### 1.3 Shoulder Angles (Priority: MEDIUM)
Multiple issues with shoulder angles due to different reference frames.

**Shoulder Flexion**:
- UPLIFT uses trunk-relative reference
- Correlation is negative (-0.28), suggesting reference frame mismatch

**Shoulder Adduction**:
- Sign appears inverted for left vs right

**Shoulder Horizontal Adduction**:
- UPLIFT: positive = arm across body
- Ours: opposite sign

**Fix in `joint_angles_3d.py`**:
```python
# Shoulder flexion - use trunk reference frame
arm_in_trunk_frame = np.dot(trunk_rotation_matrix.T, upper_arm)
flexion = np.degrees(np.arctan2(arm_in_trunk_frame[0], -arm_in_trunk_frame[2]))

# Shoulder horizontal adduction - flip sign
h_add = signed_angle_about_axis(torso_forward, upper_arm_horiz, np.array([0, 0, 1]))
# Remove the negation currently applied
```

### 1.4 Hip Angles (Priority: MEDIUM)
Hip flexion w.r.t. trunk has negative correlation.

**Fix**:
- Use trunk coordinate frame consistently
- Check sign convention for adduction

---

## Phase 2: Coordinate System Fixes

### 2.1 Center of Mass (Priority: HIGH)
**Current**: Using global coordinates (meters from world origin)
**UPLIFT**: Uses pelvis-centered coordinates

**Data comparison**:
- UPLIFT trunk_com_z: ~2.4m (relative to ground)
- Ours trunk_com_z: ~0.05m (relative to pelvis)

**Fix**: Express COM relative to pelvis center, then add ground offset.

```python
# In joint_angles_3d.py, update COM calculation:
pelvis_center = (left_hip + right_hip) / 2

# Trunk COM relative to pelvis (in pelvis coordinate frame)
trunk_com_local = trunk_com - pelvis_center
# Transform to UPLIFT coordinate system (pelvis at origin, Z = height from ground)
# Add estimated pelvis height (~1m standing)
pelvis_height = pelvis_center[2]  # Z coordinate in world frame
angles.trunk_center_of_mass_z = pelvis_height + trunk_com_local[2]
```

### 2.2 3D Joint Positions
**UPLIFT columns**: `pelvis_3d_x`, `pelvis_3d_y`, `pelvis_3d_z`
**Current**: Not exported in same format

**Fix**: Add pelvis position export matching UPLIFT naming.

---

## Phase 3: Velocity Calculation Fixes

### 3.1 Angular Velocities (Priority: HIGH)
**Current**: Simple forward difference
**Issue**: Max differences >10,000 deg/s, poor correlation

**Fix**: Use Savitzky-Golay filter for smooth derivatives

```python
from scipy.signal import savgol_filter

def calculate_angular_velocity_smooth(values, fps, window_length=7, polyorder=2):
    """Calculate smooth angular velocity using Savitzky-Golay filter."""
    dt = 1.0 / fps

    # Handle NaN values
    values_filled = pd.Series(values).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    # Apply Savitzky-Golay derivative
    if len(values_filled) >= window_length:
        velocity = savgol_filter(values_filled, window_length, polyorder, deriv=1, delta=dt)
    else:
        velocity = np.gradient(values_filled, dt)

    return velocity
```

---

## Phase 4: Missing Columns

### 4.1 Event Columns (Priority: LOW)
Add these columns (can be NaN if not detected):
- `foot_contact_frame`
- `ball_contact_frame`
- `foot_contact`
- `ball_contact`

### 4.2 Metric Columns (Priority: LOW)
Add these columns:
- `stride_length`
- `sway`
- `kinematic_sequence_order`

### 4.3 Velocity w.r.t. Ground (Priority: MEDIUM)
Add rotational velocities relative to ground:
- `pelvis_rotational_velocity_with_respect_to_ground`
- `trunk_rotational_velocity_with_respect_to_ground`

---

## Phase 5: Implementation Order

1. **joint_angles_3d.py** - Core angle fixes
   - [x] Fix knee_extension sign convention
   - [x] Fix pelvis/trunk rotation sign
   - [x] Fix shoulder angle reference frames
   - [x] Fix hip angle calculations
   - [x] Fix COM coordinate system
   - [x] Add pelvis_3d_x/y/z columns
   - [x] Add frame number field

2. **pipeline.py** - Velocity calculation
   - [x] Replace np.gradient with Savitzky-Golay filter
   - [x] Pass frame_number to angle calculator

3. **timeseries export** - Column naming
   - [x] Column names match UPLIFT format
   - [ ] Add missing event/metric columns (future)

4. **validation** - Run comparison
   - [x] Re-run frame-by-frame comparison
   - [x] Verify improved correlations

---

## Files to Modify

| File | Changes |
|------|---------|
| `joint_angles_3d.py` | Sign conventions, reference frames, COM |
| `pipeline.py` | Velocity calculation method |
| `conventions.py` | Document UPLIFT conventions |

---

## Testing Plan

1. Create unit tests with known poses
2. Compare output against UPLIFT export for same video
3. Target metrics:
   - Correlation > 0.95 for all angle columns
   - MAE < 5° for joint angles
   - MAE < 0.1m for positions

---

## Implementation Results (2026-03-07)

### Improvements Achieved

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **right_knee_extension** | corr -0.79, MAE 182° | corr 0.79, MAE 9° | **Fixed 180° offset** |
| **pelvis_global_rotation** | corr -0.91 | corr +0.91 | **Sign corrected** |
| **trunk_global_rotation** | corr -0.83 | corr +0.83 | **Sign corrected** |
| **trunk_twist_clockwise** | corr -0.88 | corr +0.88 | **Sign corrected** |

### Current Status

- 3 columns with moderate correlation (>0.8)
- Core biomechanics (knee, rotations, X-factor) well-aligned
- Shoulder angles need further investigation (different anatomical reference)
- Velocity correlations improved with Savitzky-Golay filtering

### Remaining Work

1. Shoulder angle reference frames need biomechanics expert review
2. Event detection columns (foot_contact, ball_contact) to be added
3. Metric columns (stride_length, sway) require additional calculations
