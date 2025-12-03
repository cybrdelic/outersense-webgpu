export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

export interface Quaternion {
  x: number;
  y: number;
  z: number;
  w: number;
}

export interface GazeMetrics {
  timestamp: number;
  pitch: number; // Head up/down
  yaw: number;   // Head left/right
  roll: number;  // Head tilt
  leftEyeOpen: number; // 0-1
  rightEyeOpen: number; // 0-1
  isDistracted: boolean; // Calculated based on gaze vector
  isBlinking: boolean;
  gazeX: number; // Normalized screen coordinate X (0-1)
  gazeY: number; // Normalized screen coordinate Y (0-1)
  gazeVector: Vector3; // The normalized 3D direction vector of the gaze
}

export interface FocusSession {
  startTime: number;
  totalSeconds: number;
  focusSeconds: number; // Time spent actually looking at screen
  distractionCount: number;
  averageAttentionSpan: number; // Average seconds before looking away
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'model';
  text: string;
  timestamp: Date;
}

export enum TrackingStatus {
  IDLE = 'IDLE',
  LOADING = 'LOADING',
  TRACKING = 'TRACKING',
  LOST = 'LOST',
  ERROR = 'ERROR'
}