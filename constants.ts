

export const FOCUS_THRESHOLDS = {
  // If gaze vector deviates more than this from center screen, considered "distracted"
  MAX_YAW_DEVIATION: 25, // degrees
  MAX_PITCH_DEVIATION: 20, // degrees
  BLINK_THRESHOLD: 0.4, // Eye aspect ratio for blink
};

// Physical Constants for Gaze Calculation
export const PHYSICAL_CONFIG = {
  IPD_MM: 64, // Average Inter-Pupillary Distance in mm
  // Approx 24-inch monitor dimensions (can be configurable later)
  SCREEN_WIDTH_MM: 530, 
  SCREEN_HEIGHT_MM: 300,
  CAMERA_FOV_X_DEG: 63, // Tuned for standard laptop webcam (usually 60-65)
};

export const EYE_CALIBRATION = {
  // How many degrees the eye can rotate relative to head
  MAX_YAW_DEG: 30,
  MAX_PITCH_DEG: 20,
  // Multiplier to exaggerate/dampen eye movement if needed
  SENSITIVITY: 3.2, // Tuned down slightly from 4.0 for better stability
};

// Key Landmark Indices for Face Mesh (Shared)
export const FACE_INDICES = {
  // Left Eye
  LEFT_EYE_TOP: 386,
  LEFT_EYE_BOTTOM: 374,
  LEFT_EYE_INNER: 362,
  LEFT_EYE_OUTER: 263,
  LEFT_IRIS_CENTER: 473,
  LEFT_EYE: 263, // Used for general eye position (outer corner)
  
  // Right Eye
  RIGHT_EYE_TOP: 159,
  RIGHT_EYE_BOTTOM: 145,
  RIGHT_EYE_INNER: 133,
  RIGHT_EYE_OUTER: 33,
  RIGHT_IRIS_CENTER: 468,
  RIGHT_EYE: 33, // Used for general eye position (outer corner)

  // Reference for Head Pose
  NOSE_TIP: 1,
  CHIN: 152,
  LEFT_EAR_TRAGION: 234,
  RIGHT_EAR_TRAGION: 454
};

// Using the Face Landmarker with Iris support
export const MODEL_ASSET_PATH = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
// Use a compatible WASM version that matches package version 0.10.18
export const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";

export const GEMINI_MODEL = "gemini-2.5-flash";