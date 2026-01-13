# Spectral Angle Mapper (SAM) Implementation Report

## Problem Summary
Current SAM classification is returning **0% classified pixels**, indicating a fundamental issue with data preprocessing or implementation rather than a problem with the SAM algorithm itself.

---

## Root Cause Analysis

### Primary Issues Identified

#### 1. **Scale Mismatch (MOST LIKELY CAUSE)**
**Symptoms:**
- 0% classification despite reasonable threshold (1.40 radians = 80°)
- Very large spectral angles (approaching π radians)
- Invalid cosine values in spectral angle calculations

**Problem:**
Your Hyperion cube and USGS endmember library may have different reflectance scales:
- **Expected:** Both should be 0-1 reflectance
- **Common mistakes:**
  - Hyperion data in 0-10000 scale (integer reflectance)
  - Hyperion data in 0-65535 scale (16-bit DN values)
  - USGS library already in 0-1, but cube not scaled
  - Or vice versa: cube scaled but library not scaled

**Why this causes 0% classification:**
SAM computes: `cos(θ) = (pixel · endmember) / (||pixel|| × ||endmember||)`

If pixel values are 0-10000 and endmember values are 0-1:
- Dot products become massive
- Norms are mismatched by orders of magnitude
- Computed angles are invalid or meaningless
- All angles exceed threshold → 0% classification

#### 2. **Band Count Mismatch**
**Problem:**
- Hyperion after bad band removal: typically 196 usable bands
- USGS library loaded: may have different number of bands (e.g., full 242 or resampled to different wavelengths)
- **SAM requires EXACT band count match**

**Check in your code:**
```python
cube_bands = cube.shape[2]  # e.g., 196
endmember_bands = len(list(endmembers.values())[0])  # e.g., 224
```

If these don't match, SAM will either crash or produce garbage results.

#### 3. **Invalid/Zero Pixels**
**Problem:**
- Pixels with all zeros (masked areas, no-data regions)
- These produce zero-length vectors
- Division by zero in norm calculation
- Results in NaN angles or π angles (maximum)

**Your code handles this partially:**
```python
if np.any(pixel > 0):  # Valid pixel
    angle_map[i, j] = spectral_angle(pixel, endmember_spectrum)
else:
    angle_map[i, j] = np.pi  # Invalid
```

But if cube has VERY few valid pixels, this explains 0% classification.

#### 4. **Threshold May Be Inappropriate**
**Current:** 1.40 radians = 80.2 degrees

This is actually quite permissive. Typical SAM thresholds:
- **Strict:** 0.08-0.10 radians (4.6-5.7 degrees)
- **Moderate:** 0.10-0.15 radians (5.7-8.6 degrees)
- **Permissive:** 0.15-0.20 radians (8.6-11.5 degrees)

If 1.40 radians gives 0%, the problem is definitely in the data, not the threshold.

---

## Recommended Solutions

### Solution 1: Fix Scale Issues (PRIORITY)

**Add explicit scale checking and correction:**

```python
# After loading cube and endmembers, add this diagnostic section:

print("\n" + "="*70)
print("SCALE DIAGNOSTIC")
print("="*70)

# Check cube scale
cube_valid = cube[cube > 0]
if len(cube_valid) > 0:
    cube_min = np.min(cube_valid)
    cube_max = np.max(cube_valid)
    cube_mean = np.mean(cube_valid)
    
    print(f"\nCube statistics (valid pixels only):")
    print(f"  Min: {cube_min:.6f}")
    print(f"  Max: {cube_max:.6f}")
    print(f"  Mean: {cube_mean:.6f}")
    
    # Determine scale
    if cube_max > 10000:
        print("  → Detected 16-bit scale (0-65535)")
        print("  → APPLYING CORRECTION: dividing by 65535")
        cube = cube / 65535.0
    elif cube_max > 2.0:
        print("  → Detected integer reflectance scale (0-10000)")
        print("  → APPLYING CORRECTION: dividing by 10000")
        cube = cube / 10000.0
    elif cube_max <= 1.0:
        print("  → Correct scale detected (0-1)")
    else:
        print(f"  → WARNING: Unusual scale detected (max={cube_max})")

# Check endmember scale
endmember_spectra = np.array(list(endmembers.values()))
em_min = np.min(endmember_spectra)
em_max = np.max(endmember_spectra)
em_mean = np.mean(endmember_spectra)

print(f"\nEndmember statistics:")
print(f"  Min: {em_min:.6f}")
print(f"  Max: {em_max:.6f}")
print(f"  Mean: {em_mean:.6f}")

if em_max > 2.0:
    print("  → Detected incorrect scale")
    print("  → APPLYING CORRECTION: dividing by 10000")
    endmembers = {name: spec / 10000.0 for name, spec in endmembers.items()}
elif em_max <= 1.0:
    print("  → Correct scale detected (0-1)")

print("="*70 + "\n")
```

### Solution 2: Fix Band Matching

**Add explicit band alignment:**

```python
print("\n" + "="*70)
print("BAND MATCHING DIAGNOSTIC")
print("="*70)

cube_bands = cube.shape[2]
endmember_bands = len(list(endmembers.values())[0])

print(f"\nCube bands: {cube_bands}")
print(f"Endmember bands: {endmember_bands}")

if cube_bands != endmember_bands:
    print("\n*** CRITICAL ERROR: Band count mismatch!")
    
    if cube_bands > endmember_bands:
        print(f"→ Trimming cube from {cube_bands} to {endmember_bands} bands")
        cube = cube[:, :, :endmember_bands]
        print(f"→ New cube shape: {cube.shape}")
    else:
        print(f"→ ERROR: Cube has fewer bands ({cube_bands}) than endmembers ({endmember_bands})")
        print("→ Check your endmember library file - it may not be properly resampled")
        print("→ CANNOT PROCEED")
        # You need to fix the endmember library
else:
    print("→ [OK] Band counts match!")

print("="*70 + "\n")
```

### Solution 3: Validate Data Before SAM

**Add pre-SAM validation:**

```python
def validate_sam_inputs(cube, endmembers):
    """
    Validate inputs before running SAM
    Returns: (is_valid, error_message)
    """
    
    # Check 1: Band counts
    cube_bands = cube.shape[2]
    endmember_bands = len(list(endmembers.values())[0])
    
    if cube_bands != endmember_bands:
        return False, f"Band mismatch: cube={cube_bands}, endmembers={endmember_bands}"
    
    # Check 2: Valid pixels exist
    valid_pixels = np.sum(np.any(cube > 0, axis=2))
    total_pixels = cube.shape[0] * cube.shape[1]
    valid_pct = (valid_pixels / total_pixels) * 100
    
    if valid_pixels == 0:
        return False, "No valid pixels in cube (all zeros)"
    
    if valid_pct < 1:
        print(f"WARNING: Only {valid_pct:.2f}% of pixels are valid")
    
    # Check 3: Scales look reasonable
    cube_valid = cube[cube > 0]
    cube_max = np.max(cube_valid)
    em_max = max(np.max(spec) for spec in endmembers.values())
    
    if cube_max > 2.0 or em_max > 2.0:
        return False, f"Scale issue detected: cube_max={cube_max:.2f}, em_max={em_max:.2f}"
    
    # Check 4: Test spectral angle calculation
    test_pixel = cube[cube.shape[0]//2, cube.shape[1]//2, :]
    if np.any(test_pixel > 0):
        test_em = list(endmembers.values())[0]
        
        dot = np.dot(test_pixel, test_em)
        norm_p = np.linalg.norm(test_pixel)
        norm_e = np.linalg.norm(test_em)
        
        if norm_p > 0 and norm_e > 0:
            cos_angle = dot / (norm_p * norm_e)
            
            if abs(cos_angle) > 1.0:
                return False, f"Invalid cosine: {cos_angle:.6f} (scale mismatch!)"
    
    return True, "All validation checks passed"


# Use before SAM:
is_valid, message = validate_sam_inputs(cube_for_sam, endmembers)
print(f"\nValidation: {message}")

if not is_valid:
    print("STOPPING - Fix data issues before running SAM")
    sys.exit(1)
```

### Solution 4: Use PySptools as Alternative

**If debugging continues to be difficult, switch to PySptools:**

```python
# Instead of manual SAM implementation:
from pysptools.classification import SAM

# Convert endmembers dict to 2D array
endmember_names = list(endmembers.keys())
endmember_array = np.array([endmembers[name] for name in endmember_names])

print(f"Endmember array shape: {endmember_array.shape}")  # Should be (n_minerals, n_bands)
print(f"Cube shape: {cube.shape}")  # Should be (rows, cols, n_bands)

# Create SAM classifier
sam = SAM()

# Classify with threshold in radians
# Start with 0.15 (moderate threshold)
print("\nRunning PySptools SAM classification...")
class_map = sam.classify(cube_for_sam, endmember_array, threshold=0.15)

# Print results
print(f"\nClassification complete!")
print(f"Class map shape: {class_map.shape}")

# Print classification statistics
for idx, mineral_name in enumerate(endmember_names, 1):
    count = np.sum(class_map == idx)
    pct = (count / class_map.size) * 100
    print(f"{mineral_name}: {count} pixels ({pct:.2f}%)")

# Access individual angle maps
angle_maps_dict = {}
for idx, mineral_name in enumerate(endmember_names):
    angle_maps_dict[mineral_name] = sam.angles[:, :, idx]
```

**Advantages of PySptools:**
- Automatic normalization of spectra
- Battle-tested on many datasets
- Handles edge cases better
- Less debugging needed

---

## Debugging Checklist

Run through these checks in order:

### ✅ Step 1: Check Scales
```python
print("Cube max:", np.max(cube[cube > 0]))
print("Endmember max:", max(np.max(spec) for spec in endmembers.values()))
# Both should be ≤ 1.0
```

### ✅ Step 2: Check Band Counts
```python
print("Cube bands:", cube.shape[2])
print("Endmember bands:", len(list(endmembers.values())[0]))
# Should be identical
```

### ✅ Step 3: Check Valid Pixels
```python
valid = np.sum(np.any(cube > 0, axis=2))
print(f"Valid pixels: {valid} / {cube.shape[0] * cube.shape[1]}")
# Should be > 0, ideally > 50%
```

### ✅ Step 4: Manual Angle Test
```python
test_pixel = cube[100, 100, :]
test_em = list(endmembers.values())[0]

if np.any(test_pixel > 0):
    cos_angle = np.dot(test_pixel, test_em) / (np.linalg.norm(test_pixel) * np.linalg.norm(test_em))
    print(f"Cosine angle: {cos_angle:.6f}")
    
    if abs(cos_angle) <= 1.0:
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        print(f"Angle: {angle_rad:.4f} rad ({angle_deg:.2f} deg)")
    else:
        print("ERROR: Invalid cosine (scale mismatch!)")
```

### ✅ Step 5: Try Wider Threshold
```python
# If getting 0%, try very wide threshold first
class_map, angle_maps = run_sam_all_minerals(cube, endmembers, threshold=0.50)
# Even 0.50 radians (28 degrees) should classify something if data is correct
```

---

## Expected Behavior After Fixes

Once fixed, you should see:

```
SCALE DIAGNOSTIC
================
Cube statistics (valid pixels only):
  Min: 0.001234
  Max: 0.876543
  Mean: 0.234567
  → Correct scale detected (0-1)

Endmember statistics:
  Min: 0.002345
  Max: 0.945678
  Mean: 0.345678
  → Correct scale detected (0-1)

BAND MATCHING DIAGNOSTIC
=========================
Cube bands: 196
Endmember bands: 196
→ [OK] Band counts match!

Processing 1000 x 800 pixels...
  Row 0/1000
  Row 50/1000
  ...
Classified pixels: 342567 (42.82%)
Mean angle: 0.123 rad
Min angle: 0.045 rad
```

---

## Summary of Recommended Changes

1. **Add scale checking and auto-correction** (see Solution 1)
2. **Add band matching and alignment** (see Solution 2)
3. **Add pre-SAM validation function** (see Solution 3)
4. **Consider PySptools if manual implementation continues to fail** (see Solution 4)

The most likely culprit is **scale mismatch**. Fix this first, and your 0% problem should resolve.

---

## Quick Fix Code Block

**Add this immediately after loading data, before SAM:**

```python
# ========================================================================
# CRITICAL FIX: Scale and Band Alignment
# ========================================================================

print("\n" + "="*70)
print("APPLYING DATA CORRECTIONS")
print("="*70)

# Fix 1: Scale correction
cube_valid = cube[cube > 0]
if len(cube_valid) > 0:
    cube_max = np.max(cube_valid)
    if cube_max > 10000:
        cube = cube / 65535.0
        print("✓ Corrected cube scale: /65535")
    elif cube_max > 2.0:
        cube = cube / 10000.0
        print("✓ Corrected cube scale: /10000")
    else:
        print("✓ Cube scale OK")

em_max = max(np.max(spec) for spec in endmembers.values())
if em_max > 2.0:
    endmembers = {name: spec / 10000.0 for name, spec in endmembers.items()}
    print("✓ Corrected endmember scale: /10000")
else:
    print("✓ Endmember scale OK")

# Fix 2: Band alignment
cube_bands = cube.shape[2]
endmember_bands = len(list(endmembers.values())[0])

if cube_bands != endmember_bands:
    if cube_bands > endmember_bands:
        cube = cube[:, :, :endmember_bands]
        print(f"✓ Trimmed cube bands: {cube_bands} → {endmember_bands}")
    else:
        print(f"✗ ERROR: Cannot proceed (cube bands < endmember bands)")
        sys.exit(1)
else:
    print("✓ Band counts match")

# Fix 3: Validation
valid_pixels = np.sum(np.any(cube > 0, axis=2))
total_pixels = cube.shape[0] * cube.shape[1]
print(f"✓ Valid pixels: {valid_pixels:,} ({valid_pixels/total_pixels*100:.1f}%)")

print("="*70 + "\n")
```

This code block will auto-detect and fix most common issues.
