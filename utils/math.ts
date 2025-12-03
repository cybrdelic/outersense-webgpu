

import { NormalizedLandmark } from "@mediapipe/tasks-vision";
import { Vector3, Quaternion } from "../types";
import { PHYSICAL_CONFIG, FACE_INDICES, EYE_CALIBRATION } from "../constants";

// Convert MediaPipe landmark to Vector3
export const toVector = (l: NormalizedLandmark): Vector3 => ({ x: l.x, y: l.y, z: l.z });

// Basic Vector Math
export const sub = (v1: Vector3, v2: Vector3): Vector3 => ({ x: v1.x - v2.x, y: v1.y - v2.y, z: v1.z - v2.z });
export const add = (v1: Vector3, v2: Vector3): Vector3 => ({ x: v1.x + v2.x, y: v1.y + v2.y, z: v1.z + v2.z });
export const dot = (v1: Vector3, v2: Vector3): number => v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
export const cross = (v1: Vector3, v2: Vector3): Vector3 => ({
  x: v1.y * v2.z - v1.z * v2.y,
  y: v1.z * v2.x - v1.x * v2.z,
  z: v1.x * v2.y - v1.y * v2.x
});
export const mag = (v: Vector3): number => Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
export const normalize = (v: Vector3): Vector3 => {
  const m = mag(v);
  return m < 0.00001 ? { x: 0, y: 0, z: 0 } : { x: v.x / m, y: v.y / m, z: v.z / m };
};

export const dist = (v1: Vector3, v2: Vector3): number => {
    const dx = v1.x - v2.x;
    const dy = v1.y - v2.y;
    const dz = v1.z - v2.z;
    return Math.sqrt(dx*dx + dy*dy + dz*dz);
};

/**
 * Maps a MediaPipe normalized landmark (0-1) to WebGPU View Space.
 * View Space: Center (0,0), Y+ Up.
 * @param mirror If true, flips X axis for mirrored visualization (Webcam style)
 */
export const mapToViewSpace = (l: NormalizedLandmark, aspectRatio: number, mirror: boolean = false): Vector3 => {
    // MediaPipe: 0,0 is Top-Left.
    // View Space: 0,0 is Center. Y is Up.
    const x = l.x - 0.5;
    const y = 0.5 - l.y; // 0->0.5, 1->-0.5.
    
    // If Mirror: -x. If x was -0.5 (Left Image), becomes 0.5 (Right Screen).
    // If Normal: x. x was -0.5 (Left Image), stays -0.5 (Left Screen).
    return {
        x: (mirror ? -x : x) * 2.0 * aspectRatio,
        y: y * 2.0,
        z: l.z * 5.0 // Scale depth for better 3D effect
    };
};

/**
 * Multiplies two quaternions.
 * Order matters: q1 * q2 applies q2 rotation then q1 rotation (local to global often)
 */
export const multiplyQuaternions = (q1: Quaternion, q2: Quaternion): Quaternion => {
  return {
    w: q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
    x: q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
    y: q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
    z: q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
  };
};

/**
 * Create a quaternion from Euler angles (radians)
 * Order: ZYX (Yaw, Pitch, Roll)
 */
export const eulerToQuaternion = (pitch: number, yaw: number, roll: number): Quaternion => {
  const cy = Math.cos(yaw * 0.5);
  const sy = Math.sin(yaw * 0.5);
  const cp = Math.cos(pitch * 0.5);
  const sp = Math.sin(pitch * 0.5);
  const cr = Math.cos(roll * 0.5);
  const sr = Math.sin(roll * 0.5);

  return {
    w: cr * cp * cy + sr * sp * sy,
    x: sr * cp * cy - cr * sp * sy,
    y: cr * sp * cy + sr * cp * sy,
    z: cr * cp * sy - sr * sp * cy
  };
};

/**
 * Calculates a rotation quaternion based on face landmarks.
 * We use the eyes and nose to establish a local coordinate system.
 */
export const calculateFaceQuaternion = (
  leftEye: Vector3,
  rightEye: Vector3,
  noseTip: Vector3
): Quaternion => {
  // 1. X-Axis: Vector from left eye to right eye
  let xAxis = normalize(sub(rightEye, leftEye));
  if (isNaN(xAxis.x)) xAxis = { x: 1, y: 0, z: 0 };

  // 2. Y-Axis: Approximate up vector.
  const eyeMid = { x: (leftEye.x + rightEye.x) / 2, y: (leftEye.y + rightEye.y) / 2, z: (leftEye.z + rightEye.z) / 2 };
  const noseDir = sub(noseTip, eyeMid);
  
  // Z-Axis: Cross X and NoseDir to get Y (Up/Down relative to face)
  let yAxis = normalize(cross(xAxis, noseDir)); 
  if (isNaN(yAxis.x)) yAxis = { x: 0, y: 1, z: 0 };
  
  // Z-Axis: Cross X and Y to ensure orthogonality (Forward)
  let zAxis = normalize(cross(xAxis, yAxis));
  if (isNaN(zAxis.x)) zAxis = { x: 0, y: 0, z: 1 };

  // Create Rotation Matrix from axes [X, Y, Z]
  const m00 = xAxis.x, m01 = yAxis.x, m02 = zAxis.x;
  const m10 = xAxis.y, m11 = yAxis.y, m12 = zAxis.y;
  const m20 = xAxis.z, m21 = yAxis.z, m22 = zAxis.z;

  // Convert Matrix to Quaternion
  const tr = m00 + m11 + m22;
  let qw, qx, qy, qz;

  if (tr > 0) { 
    const S = Math.sqrt(tr+1.0) * 2; // S=4*qw 
    qw = 0.25 * S;
    qx = (m21 - m12) / S;
    qy = (m02 - m20) / S; 
    qz = (m10 - m01) / S; 
  } else if ((m00 > m11) && (m00 > m22)) { 
    const S = Math.sqrt(1.0 + m00 - m11 - m22) * 2; // S=4*qx 
    qw = (m21 - m12) / S;
    qx = 0.25 * S;
    qy = (m01 + m10) / S; 
    qz = (m02 + m20) / S; 
  } else if (m11 > m22) { 
    const S = Math.sqrt(1.0 + m11 - m00 - m22) * 2; // S=4*qy
    qw = (m02 - m20) / S;
    qx = (m01 + m10) / S; 
    qy = 0.25 * S;
    qz = (m12 + m21) / S; 
  } else { 
    const S = Math.sqrt(1.0 + m22 - m00 - m11) * 2; // S=4*qz
    qw = (m10 - m01) / S;
    qx = (m02 + m20) / S;
    qy = (m12 + m21) / S;
    qz = 0.25 * S;
  }
  
  // Safety check for NaN in Quaternion
  if (isNaN(qw) || isNaN(qx) || isNaN(qy) || isNaN(qz)) {
      return { w: 1, x: 0, y: 0, z: 0 };
  }

  return { w: qw, x: qx, y: qy, z: qz };
};

export const quaternionToEuler = (q: Quaternion) => {
  // Roll (x-axis rotation)
  const sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
  const cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
  const roll = Math.atan2(sinr_cosp, cosr_cosp);

  // Pitch (y-axis rotation)
  const sinp = 2 * (q.w * q.y - q.z * q.x);
  let pitch;
  if (Math.abs(sinp) >= 1)
      pitch = (Math.PI / 2) * Math.sign(sinp);
  else
      pitch = Math.asin(sinp);

  // Yaw (z-axis rotation)
  const siny_cosp = 2 * (q.w * q.z + q.x * q.y);
  const cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
  const yaw = Math.atan2(siny_cosp, cosy_cosp);

  return {
      roll: roll * (180 / Math.PI),
      pitch: pitch * (180 / Math.PI),
      yaw: yaw * (180 / Math.PI)
  };
};

export const calculateEyeOpenness = (
  upper: NormalizedLandmark, 
  lower: NormalizedLandmark, 
  inner: NormalizedLandmark, 
  outer: NormalizedLandmark
) => {
  const h = Math.abs(upper.y - lower.y);
  const w = Math.abs(inner.x - outer.x);
  return h / w;
};

/**
 * Calculates rotation quaternion for the eyes relative to the head.
 * Uses the position of the iris between eye corners.
 */
const calculateEyeRotation = (landmarks: NormalizedLandmark[]): Quaternion => {
  // Left Eye
  const lInner = landmarks[FACE_INDICES.LEFT_EYE_INNER];
  const lOuter = landmarks[FACE_INDICES.LEFT_EYE_OUTER];
  const lIris = landmarks[FACE_INDICES.LEFT_IRIS_CENTER];
  
  // Right Eye
  const rInner = landmarks[FACE_INDICES.RIGHT_EYE_INNER];
  const rOuter = landmarks[FACE_INDICES.RIGHT_EYE_OUTER];
  const rIris = landmarks[FACE_INDICES.RIGHT_IRIS_CENTER];

  // Helper to get normalized iris position (-1 to 1)
  const getIrisPos = (inner: NormalizedLandmark, outer: NormalizedLandmark, iris: NormalizedLandmark) => {
    const center = { x: (inner.x + outer.x) / 2, y: (inner.y + outer.y) / 2 };
    const width = Math.hypot(inner.x - outer.x, inner.y - outer.y);
    const height = width * 0.3; // Approximation

    const dx = iris.x - center.x;
    const dy = iris.y - center.y;
    
    // Normalize -1 to 1
    return {
      x: (dx / (width / 2)),
      y: (dy / (height / 2))
    };
  };

  const lOffset = getIrisPos(lInner, lOuter, lIris);
  const rOffset = getIrisPos(rInner, rOuter, rIris);

  // Average offsets
  const avgX = (lOffset.x + rOffset.x) / 2;
  const avgY = (lOffset.y + rOffset.y) / 2;

  // Convert to angles
  const yawRad = avgX * (EYE_CALIBRATION.MAX_YAW_DEG * Math.PI / 180) * EYE_CALIBRATION.SENSITIVITY;
  const pitchRad = avgY * (EYE_CALIBRATION.MAX_PITCH_DEG * Math.PI / 180) * EYE_CALIBRATION.SENSITIVITY;

  // Pitch flipped to correct Up/Down inversion
  // Yaw flipped to correct Left/Right inversion
  return eulerToQuaternion(-pitchRad, yawRad, 0); 
};

/**
 * Calculates where the user is looking on the screen based on head pose and IPD.
 * Assumes camera is mounted at top-center of screen.
 */
export const calculateGazeIntersection = (
  landmarks: NormalizedLandmark[], 
  headQuaternion: Quaternion
): { x: number, y: number, gazeVector: Vector3 } => {
  
  if (!landmarks || landmarks.length === 0) return { x: 0.5, y: 0.5, gazeVector: {x:0,y:0,z:-1} };

  const leftEyeInner = landmarks[FACE_INDICES.LEFT_EYE_INNER];
  const rightEyeInner = landmarks[FACE_INDICES.RIGHT_EYE_INNER];
  const nose = landmarks[FACE_INDICES.NOSE_TIP];

  // 1. Estimate Distance (Z) using IPD
  const dx = Math.abs(leftEyeInner.x - rightEyeInner.x);
  const dy = Math.abs(leftEyeInner.y - rightEyeInner.y);
  const distNorm = Math.sqrt(dx*dx + dy*dy);
  
  if (distNorm === 0) return { x: 0.5, y: 0.5, gazeVector: {x:0,y:0,z:-1} };

  // Real world Z (mm)
  const fovRad = (PHYSICAL_CONFIG.CAMERA_FOV_X_DEG * Math.PI) / 180;
  const zDistMM = PHYSICAL_CONFIG.IPD_MM / (2 * distNorm * Math.tan(fovRad / 2));

  // 2. Real World Head Position
  const visibleWidthAtZ = 2 * zDistMM * Math.tan(fovRad / 2);
  const visibleHeightAtZ = visibleWidthAtZ * (3/4); 
  
  const headX = (nose.x - 0.5) * visibleWidthAtZ;
  const headY = (nose.y - 0.5) * visibleHeightAtZ; 
  const headZ = zDistMM; 

  // 3. True Gaze Vector = Head Rotation * Eye Rotation
  const eyeQuaternion = calculateEyeRotation(landmarks);
  
  // Combine rotations: Head applied first, then Eyes relative to head.
  const q = multiplyQuaternions(headQuaternion, eyeQuaternion);
  
  // Rotate vector (0, 0, -1) by q (Assuming -Z is forward out of face)
  const vx = 2 * (q.x * q.z - q.w * q.y);
  const vy = 2 * (q.y * q.z + q.w * q.x);
  const vz = 1 - 2 * (q.x * q.x + q.y * q.y);
  const gazeDir = { x: -vx, y: -vy, z: -vz };

  // 4. Intersect Gaze Ray with Screen Plane (Z=0)
  if (Math.abs(gazeDir.z) < 0.01) return { x: 0.5, y: 0.5, gazeVector: gazeDir }; 
  
  // Ray: P = Head + t * Dir
  // Solve for Z=0: 0 = headZ + t * gazeDir.z
  const t = -headZ / gazeDir.z;
  
  const intersectX = headX + t * gazeDir.x;
  const intersectY = headY + t * gazeDir.y;

  // 5. Map to Normalized Screen Coordinates
  // Note: Y Axis in MP is Down. headY is distance from center.
  // 0 at Top. nose.y < 0.5 -> headY is negative.
  // gazeDir.y -> If looking Up, y is Negative? 
  // Let's assume standard coords for intersection logic: Center is (0,0).
  // Screen Top is -H/2. Screen Bottom is +H/2.
  const screenNormY = (intersectY + PHYSICAL_CONFIG.SCREEN_HEIGHT_MM / 2) / PHYSICAL_CONFIG.SCREEN_HEIGHT_MM;
  
  // For X: Left is -W/2, Right is +W/2.
  const rawNormX = (intersectX + PHYSICAL_CONFIG.SCREEN_WIDTH_MM / 2) / PHYSICAL_CONFIG.SCREEN_WIDTH_MM;
  
  // Direct mapping. 0 (Left) to 1 (Right).
  // Previous inversion "1.0 - rawNormX" was incorrect for the "Look Right -> Laser Right" behavior.
  const screenNormX = rawNormX;

  // Clamp safely for return, but don't prevent off-screen detection
  const safeX = isNaN(screenNormX) ? 0.5 : screenNormX;
  const safeY = isNaN(screenNormY) ? 0.5 : screenNormY;

  return { x: safeX, y: safeY, gazeVector: gazeDir };
};
