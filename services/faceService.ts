import {
  FaceLandmarker,
  FilesetResolver
} from "@mediapipe/tasks-vision";
import { MODEL_ASSET_PATH, WASM_PATH } from "../constants";

let faceLandmarker: FaceLandmarker | null = null;

export const initializeFaceLandmarker = async (): Promise<FaceLandmarker> => {
  if (faceLandmarker) return faceLandmarker;

  try {
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    
    try {
      // Try WebGPU first
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_ASSET_PATH,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numFaces: 1,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        refineLandmarks: true
      });
      console.log("WebGPU FaceLandmarker initialized successfully");
    } catch (gpuError) {
      console.error("WebGPU initialization failed:", gpuError);
      console.warn("Falling back to CPU. Performance may be degraded.");
      
      // Fallback to CPU
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_ASSET_PATH,
          delegate: "CPU"
        },
        runningMode: "VIDEO",
        numFaces: 1,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        refineLandmarks: true
      });
    }

    return faceLandmarker!;
  } catch (e) {
    console.error("Failed to initialize FaceLandmarker:", e);
    throw e;
  }
};

export { FACE_INDICES } from "../constants";