
import React, { useEffect, useRef, useState, useCallback } from 'react';
import { FaceLandmarker, FaceLandmarkerOptions } from '@mediapipe/tasks-vision';
import { initializeFaceLandmarker } from '../services/faceService';
import { FACE_INDICES } from '../constants';
import { toVector, calculateFaceQuaternion, quaternionToEuler, calculateEyeOpenness, calculateGazeIntersection, mapToViewSpace, dist } from '../utils/math';
import { GazeMetrics, TrackingStatus } from '../types';
import { FOCUS_THRESHOLDS } from '../constants';

// --- WebGPU Type Definitions ---
interface CustomNavigator extends Navigator {
  gpu: any;
}

type GPUDevice = any;
type GPURenderPipeline = any;
type GPUBuffer = any;
type GPUBindGroup = any;
type GPUCanvasContext = any;
type GPURenderPassDescriptor = any;

const GPUBufferUsage = {
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
};

const GPUShaderStage = {
  VERTEX: 0x1,
  FRAGMENT: 0x2,
  COMPUTE: 0x4,
};

interface GazeTrackerProps {
  onMetricsUpdate: (metrics: GazeMetrics) => void;
  isActive: boolean;
  calibrationOffset: { x: number, y: number };
}

// ----------------------------------------------------------------------------
// WGSL SHADER: PHOTOREALISTIC PBR EYES & VOLUMETRIC LIGHT
// ----------------------------------------------------------------------------
const GAZE_SHADER = `
struct Uniforms {
  leftEye: vec4f,      // xyz mapped to view space
  rightEye: vec4f,     // xyz mapped to view space
  leftGazeEnd: vec4f,  // xyz mapped to view space
  rightGazeEnd: vec4f, // xyz mapped to view space
  leftShape: vec4f,    // width, height, reserved, reserved
  rightShape: vec4f,   // width, height, reserved, reserved
  resolution: vec2f,   // width, height
  time: f32,
  isDistracted: f32,   // 0.0 or 1.0
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var pos = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  var output: VertexOutput;
  output.position = vec4f(pos[vertexIndex], 0.0, 1.0);
  output.uv = pos[vertexIndex]; // -1 to 1
  return output;
}

// --- SDF Primitives ---

fn sdSegment(p: vec3f, a: vec3f, b: vec3f) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

// Noise function for Iris texture
fn hash(n: f32) -> f32 { return fract(sin(n)*43758.5453); }
fn noise(x: vec3f) -> f32 {
    let p = floor(x);
    let f = fract(x);
    let n = p.x + p.y*57.0 + 113.0*p.z;
    return mix(mix(mix(hash(n+0.0), hash(n+1.0),f.x),
                   mix(hash(n+57.0), hash(n+58.0),f.x),f.y),
               mix(mix(hash(n+113.0), hash(n+114.0),f.x),
                   mix(hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    let aspect = uniforms.resolution.x / uniforms.resolution.y;
    
    // Camera Setup: Z=2.0 looking at Z=0.
    let ro = vec3f(0.0, 0.0, 2.0);
    let rd = normalize(vec3f(uv.x * aspect, uv.y, -2.0)); 

    let lEye = uniforms.leftEye.xyz;
    let rEye = uniforms.rightEye.xyz;
    let lEnd = uniforms.leftGazeEnd.xyz;
    let rEnd = uniforms.rightGazeEnd.xyz;
    
    var col = vec3f(0.0);
    
    // Raymarching
    var t = 0.0;
    let steps = 60; 
    let stepSize = 0.1;
    let maxDist = 8.0;

    // Define Colors
    let colorFocus = vec3f(0.0, 0.9, 1.0); // Cyan
    let colorDistract = vec3f(1.0, 0.1, 0.05); // Orange-Red
    let beamColor = mix(colorFocus, colorDistract, uniforms.isDistracted);

    // Light Source (Virtual Screen Glare)
    let lightPos = vec3f(0.0, 1.0, 5.0);
    
    // Default scanning beam if no eyes detected
    let isTracking = length(lEye) > 0.01; // Eye center is never exactly 0,0,0 if tracked

    // "Scanning" idle animation if not tracking
    if (!isTracking) {
        let scanY = sin(uniforms.time * 3.0) * 0.8;
        let dScan = abs(uv.y - scanY);
        let scanBeam = smoothstep(0.05, 0.0, dScan) * 0.8; // Brighter
        col += vec3f(0.0, 0.5, 0.8) * scanBeam;
    }

    for(var i=0; i<steps; i++) {
        let p = ro + rd * t;
        if(t > maxDist) { break; }

        if (isTracking) {
            // --- Volumetric Gaze Beams ---
            let dL = sdSegment(p, lEye, lEnd);
            let dR = sdSegment(p, rEye, rEnd);
            
            // Inverse Square Falloff
            let beamWidth = 0.015;
            let scatterL = 1.0 / (1.0 + pow(dL / beamWidth, 2.0));
            let scatterR = 1.0 / (1.0 + pow(dR / beamWidth, 2.0));
            
            // Add Noise/Dust
            let dust = noise(p * 15.0 + vec3f(0.0, 0.0, uniforms.time * 2.0));
            let intensity = 0.003 * (1.0 + dust * 0.5);

            col += beamColor * (scatterL + scatterR) * intensity;
            
            // --- Photorealistic Eye Rendering ---
            let wL = max(uniforms.leftShape.x * 3.0, 0.01);
            let hL = max(uniforms.leftShape.y * 3.0, 0.01);
            let lDims = vec3f(wL, hL, 0.08); // Thickened Z-depth for reliability
            
            let wR = max(uniforms.rightShape.x * 3.0, 0.01);
            let hR = max(uniforms.rightShape.y * 3.0, 0.01);
            let rDims = vec3f(wR, hR, 0.08); // Thickened Z-depth for reliability
            
            let pL = p - lEye;
            let pR = p - rEye;
            
            // Check bounding volume (Relaxed radius for better stability)
            if (dot(pL, pL) < 0.04) { 
                let dEyeL = length(pL / lDims);
                if (dEyeL < 1.0) {
                     // Approximate Normal
                     let N = normalize(pL / (lDims * lDims));
                     let V = -rd;
                     let L = normalize(lightPos - p);
                     let H = normalize(L + V);

                     // Fresnel
                     let F0 = 0.04; 
                     let fresnel = F0 + (1.0 - F0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);
                     let spec = pow(max(dot(N, H), 0.0), 64.0);
                     
                     // Procedural Iris
                     let u = pL.x * 20.0;
                     let v = pL.y * 20.0;
                     let r = sqrt(u*u + v*v);
                     let theta = atan2(v, u);
                     
                     let f = noise(vec3f(r * 2.0, theta * 5.0, 0.0));
                     let f2 = noise(vec3f(r * 10.0, theta * 20.0, 0.0));
                     let irisColor = mix(vec3f(0.1, 0.3, 0.8), vec3f(0.6, 0.8, 1.0), f * 0.5 + f2 * 0.2);
                     let pupil = smoothstep(0.15, 0.2, r * 0.2);
                     
                     let surfColor = mix(vec3f(0.0), irisColor * beamColor, pupil);
                     let rim = smoothstep(0.8, 1.0, dEyeL);
                     let glow = beamColor * rim * 2.0;

                     let finalColor = surfColor + vec3f(spec) + glow + (fresnel * beamColor * 0.5);
                     col += finalColor * 0.3; // Accumulate transparency
                }
            }
            
            if (dot(pR, pR) < 0.04) { 
                let dEyeR = length(pR / rDims);
                if (dEyeR < 1.0) {
                     let N = normalize(pR / (rDims * rDims));
                     let V = -rd;
                     let L = normalize(lightPos - p);
                     let H = normalize(L + V);
                     
                     // Fresnel
                     let F0 = 0.04; 
                     let fresnel = F0 + (1.0 - F0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);
                     let spec = pow(max(dot(N, H), 0.0), 64.0);
                     
                     let u = pR.x * 20.0;
                     let v = pR.y * 20.0;
                     let r = sqrt(u*u + v*v);
                     let theta = atan2(v, u);
                     
                     let f = noise(vec3f(r * 2.0, theta * 5.0, 0.0));
                     let f2 = noise(vec3f(r * 10.0, theta * 20.0, 0.0));
                     let irisColor = mix(vec3f(0.1, 0.3, 0.8), vec3f(0.6, 0.8, 1.0), f * 0.5 + f2 * 0.2);
                     let pupil = smoothstep(0.15, 0.2, r * 0.2);
                     
                     let surfColor = mix(vec3f(0.0), irisColor * beamColor, pupil);
                     let rim = smoothstep(0.8, 1.0, dEyeR);
                     let glow = beamColor * rim * 2.0;

                     let finalColor = surfColor + vec3f(spec) + glow + (fresnel * beamColor * 0.5);
                     col += finalColor * 0.3; 
                }
            }
        }
        
        t += stepSize;
    }
    
    // --- Alpha Calculation for Transparency ---
    let maxComp = max(col.r, max(col.g, col.b));
    let alpha = clamp(maxComp, 0.0, 1.0);
    
    return vec4f(col, alpha);
}
`;

// ----------------------------------------------------------------------------
// WGSL SHADER: HOLOGRAPHIC FACE MESH
// ----------------------------------------------------------------------------
const MESH_SHADER = `
struct Uniforms {
  leftEye: vec4f,      
  rightEye: vec4f,     
  leftGazeEnd: vec4f,  
  rightGazeEnd: vec4f, 
  leftShape: vec4f,    
  rightShape: vec4f,   
  resolution: vec2f,   
  time: f32,
  isDistracted: f32,   
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3f
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) depth: f32,
    @location(1) uv: vec2f // Store for gradient
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    // MediaPipe (0..1) to Clip (-1..1)
    // Mirroring applied here: X -> 1.0 - X
    let x = (1.0 - input.position.x) * 2.0 - 1.0;
    let y = (1.0 - input.position.y) * 2.0 - 1.0; 
    
    output.position = vec4f(x, y, 0.0, 1.0);
    output.depth = input.position.z;
    output.uv = vec2f(input.position.x, input.position.y);
    return output;
}

@fragment
fn fs_main(@location(0) depth: f32, @location(1) uv: vec2f) -> @location(0) vec4f {
    let colorFocus = vec3f(0.0, 0.9, 1.0); // Cyan
    let colorDistract = vec3f(1.0, 0.1, 0.05); // Orange-Red
    let baseColor = mix(colorFocus, colorDistract, uniforms.isDistracted);
    
    // Holographic Scanline
    let scan = sin(uv.y * 40.0 - uniforms.time * 5.0);
    let pulse = (scan + 1.0) * 0.5;
    
    // Gradient fade from top to bottom
    let grad = 1.0 - uv.y;
    
    // Composite
    let alpha = 0.3 * grad + (pulse * 0.1);
    
    return vec4f(baseColor * (1.0 + pulse), alpha);
}
`;

export const GazeTracker: React.FC<GazeTrackerProps> = ({ onMetricsUpdate, isActive, calibrationOffset }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<TrackingStatus>(TrackingStatus.IDLE);
  const requestRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);
  const [videoAspect, setVideoAspect] = useState<number>(16/9); // Default

  // Smooth filter state
  const smoothedGaze = useRef({ x: 0.5, y: 0.5 });
  const smoothedVector = useRef({ x: 0, y: 0, z: -1 });
  const alpha = 0.35; 

  // WebGPU Refs
  const deviceRef = useRef<GPUDevice>(null);
  const pipelineRef = useRef<GPURenderPipeline>(null);
  const meshPipelineRef = useRef<GPURenderPipeline>(null); 
  const uniformBufferRef = useRef<GPUBuffer>(null);
  const vertexBufferRef = useRef<GPUBuffer>(null); 
  const indexBufferRef = useRef<GPUBuffer>(null); 
  const bindGroupRef = useRef<GPUBindGroup>(null);
  const contextRef = useRef<GPUCanvasContext>(null);
  const indexCountRef = useRef<number>(0);

  // Initialize WebGPU
  useEffect(() => {
    const initWebGPU = async () => {
      const navigator = window.navigator as CustomNavigator;
      if (!navigator.gpu) {
        console.error("WebGPU not supported");
        return;
      }

      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        console.error("No WebGPU adapter found");
        return;
      }

      const device = await adapter.requestDevice();
      deviceRef.current = device;

      const canvas = canvasRef.current;
      if (!canvas) return;

      const context = canvas.getContext('webgpu') as GPUCanvasContext;
      contextRef.current = context;
      
      const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
      context.configure({
        device,
        format: presentationFormat,
        alphaMode: 'premultiplied',
      });

      // --- Uniform Buffer ---
      const uniformBufferSize = 128; 
      const uniformBuffer = device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      uniformBufferRef.current = uniformBuffer;

      // --- Gaze Pipeline (Raymarching) ---
      const shaderModule = device.createShaderModule({
        label: 'Gaze Shader',
        code: GAZE_SHADER,
      });

      const bindGroupLayout = device.createBindGroupLayout({
        entries: [{
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, // Shared
          buffer: { type: 'uniform' },
        }],
      });

      const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      });

      const pipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
          module: shaderModule,
          entryPoint: 'vs_main',
        },
        fragment: {
          module: shaderModule,
          entryPoint: 'fs_main',
          targets: [{
            format: presentationFormat,
            blend: {
                color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          }],
        },
        primitive: { topology: 'triangle-list' },
      });
      pipelineRef.current = pipeline;

      // --- Mesh Pipeline (Wireframe) ---
      const meshShaderModule = device.createShaderModule({
          label: 'Mesh Shader',
          code: MESH_SHADER
      });

      const meshPipeline = device.createRenderPipeline({
          layout: pipelineLayout, 
          vertex: {
              module: meshShaderModule,
              entryPoint: 'vs_main',
              buffers: [{
                  arrayStride: 12, // vec3f 
                  attributes: [{
                      shaderLocation: 0,
                      offset: 0,
                      format: 'float32x3'
                  }]
              }]
          },
          fragment: {
              module: meshShaderModule,
              entryPoint: 'fs_main',
              targets: [{
                  format: presentationFormat,
                  blend: {
                    color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                  }
              }]
          },
          primitive: { topology: 'line-list' },
          depthStencil: undefined 
      });
      meshPipelineRef.current = meshPipeline;

      // Initialize Mesh Buffers
      const vertexBuffer = device.createBuffer({
          size: 478 * 3 * 4,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
      });
      vertexBufferRef.current = vertexBuffer;

      const connectors = FaceLandmarker.FACE_LANDMARKS_TESSELATION;
      const indexCount = connectors.length * 2;
      indexCountRef.current = indexCount;
      const indexData = new Uint16Array(indexCount);
      connectors.forEach((c, i) => {
          indexData[i * 2] = c.start;
          indexData[i * 2 + 1] = c.end;
      });

      const indexBuffer = device.createBuffer({
          size: indexData.byteLength,
          usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
          mappedAtCreation: true
      });
      new Uint16Array(indexBuffer.getMappedRange()).set(indexData);
      indexBuffer.unmap();
      indexBufferRef.current = indexBuffer;

      // Bind Group
      const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{
          binding: 0,
          resource: { buffer: uniformBuffer },
        }],
      });
      bindGroupRef.current = bindGroup;
    };

    initWebGPU();
  }, []);

  // Handle Video Metadata to lock aspect ratio
  const handleVideoLoad = () => {
      if (videoRef.current) {
          const w = videoRef.current.videoWidth;
          const h = videoRef.current.videoHeight;
          if (w && h) {
              setVideoAspect(w / h);
              // Resize canvas buffer to match video
              if (canvasRef.current) {
                  canvasRef.current.width = w;
                  canvasRef.current.height = h;
              }
          }
          predict();
      }
  };


  // --- Face Tracking Loop ---
  const predict = useCallback(async () => {
    if (!videoRef.current || !isActive) return;

    if (videoRef.current.readyState < 2 || videoRef.current.videoWidth === 0) {
        requestRef.current = requestAnimationFrame(predict);
        return;
    }

    const now = performance.now();
    const dt = (now - lastTimeRef.current) / 1000;
    lastTimeRef.current = now;

    let faceLandmarker = null;
    try {
      faceLandmarker = await initializeFaceLandmarker();
    } catch (e) {
      setStatus(TrackingStatus.ERROR);
      return;
    }

    if (status !== TrackingStatus.TRACKING) setStatus(TrackingStatus.TRACKING);

    const results = faceLandmarker.detectForVideo(videoRef.current, now);

    // --- WebGPU Rendering Prep ---
    let lEyeView = { x: 0, y: 0, z: 0 };
    let rEyeView = { x: 0, y: 0, z: 0 };
    let lGazeEnd = { x: 0, y: 0, z: 0 };
    let rGazeEnd = { x: 0, y: 0, z: 0 };
    let isDistractedVal = 0;
    let lShape = { w: 0.01, h: 0.005 };
    let rShape = { w: 0.01, h: 0.005 };

    if (results.faceLandmarks.length > 0) {
      const landmarks = results.faceLandmarks[0];
      
      // Update Mesh Vertices
      if (deviceRef.current && vertexBufferRef.current) {
          const vertexData = new Float32Array(landmarks.length * 3);
          for (let i = 0; i < landmarks.length; i++) {
              vertexData[i * 3] = landmarks[i].x;
              vertexData[i * 3 + 1] = landmarks[i].y;
              vertexData[i * 3 + 2] = landmarks[i].z;
          }
          deviceRef.current.queue.writeBuffer(vertexBufferRef.current, 0, vertexData);
      }
      
      const lEyeOuter = toVector(landmarks[FACE_INDICES.LEFT_EYE_OUTER]);
      const rEyeOuter = toVector(landmarks[FACE_INDICES.RIGHT_EYE_OUTER]);
      const noseTip = toVector(landmarks[FACE_INDICES.NOSE_TIP]);
      const headQuat = calculateFaceQuaternion(lEyeOuter, rEyeOuter, noseTip);
      const euler = quaternionToEuler(headQuat);

      const { x: gazeXRaw, y: gazeYRaw, gazeVector } = calculateGazeIntersection(landmarks, headQuat);

      smoothedGaze.current.x = smoothedGaze.current.x + (gazeXRaw - smoothedGaze.current.x) * alpha;
      smoothedGaze.current.y = smoothedGaze.current.y + (gazeYRaw - smoothedGaze.current.y) * alpha;
      smoothedVector.current.x = smoothedVector.current.x + (gazeVector.x - smoothedVector.current.x) * alpha;
      smoothedVector.current.y = smoothedVector.current.y + (gazeVector.y - smoothedVector.current.y) * alpha;
      smoothedVector.current.z = smoothedVector.current.z + (gazeVector.z - smoothedVector.current.z) * alpha;

      const calGazeX = smoothedGaze.current.x + calibrationOffset.x;
      const calGazeY = smoothedGaze.current.y + calibrationOffset.y;

      const isLookingAwayX = calGazeX < -0.1 || calGazeX > 1.1; 
      const isLookingAwayY = calGazeY > 1.2; 
      const isDistracted = isLookingAwayX || isLookingAwayY;
      isDistractedVal = isDistracted ? 1.0 : 0.0;

      const lOpen = calculateEyeOpenness(
        landmarks[FACE_INDICES.LEFT_EYE_TOP], landmarks[FACE_INDICES.LEFT_EYE_BOTTOM],
        landmarks[FACE_INDICES.LEFT_EYE_INNER], landmarks[FACE_INDICES.LEFT_EYE_OUTER]
      );
      const rOpen = calculateEyeOpenness(
        landmarks[FACE_INDICES.RIGHT_EYE_TOP], landmarks[FACE_INDICES.RIGHT_EYE_BOTTOM],
        landmarks[FACE_INDICES.RIGHT_EYE_INNER], landmarks[FACE_INDICES.RIGHT_EYE_OUTER]
      );
      const isBlinking = (lOpen + rOpen) / 2 < FOCUS_THRESHOLDS.BLINK_THRESHOLD;

      onMetricsUpdate({
        timestamp: now,
        pitch: euler.pitch,
        yaw: euler.yaw,
        roll: euler.roll,
        leftEyeOpen: lOpen,
        rightEyeOpen: rOpen,
        isDistracted,
        isBlinking,
        gazeX: calGazeX,
        gazeY: calGazeY,
        gazeVector: smoothedVector.current
      });

      const aspectRatio = videoRef.current.videoWidth / videoRef.current.videoHeight;
      
      lEyeView = mapToViewSpace(landmarks[FACE_INDICES.LEFT_IRIS_CENTER], aspectRatio, true);
      rEyeView = mapToViewSpace(landmarks[FACE_INDICES.RIGHT_IRIS_CENTER], aspectRatio, true);
      lEyeView.z = 0.0;
      rEyeView.z = 0.0;

      const beamLen = 10.0;
      lGazeEnd = { 
        x: lEyeView.x + smoothedVector.current.x * beamLen * aspectRatio, 
        y: lEyeView.y + smoothedVector.current.y * beamLen,
        z: lEyeView.z + smoothedVector.current.z * beamLen 
      };
      rGazeEnd = { 
        x: rEyeView.x + smoothedVector.current.x * beamLen * aspectRatio,
        y: rEyeView.y + smoothedVector.current.y * beamLen,
        z: rEyeView.z + smoothedVector.current.z * beamLen 
      };

      const lW = dist(toVector(landmarks[FACE_INDICES.LEFT_EYE_INNER]), toVector(landmarks[FACE_INDICES.LEFT_EYE_OUTER]));
      const lH = dist(toVector(landmarks[FACE_INDICES.LEFT_EYE_TOP]), toVector(landmarks[FACE_INDICES.LEFT_EYE_BOTTOM]));
      const rW = dist(toVector(landmarks[FACE_INDICES.RIGHT_EYE_INNER]), toVector(landmarks[FACE_INDICES.RIGHT_EYE_OUTER]));
      const rH = dist(toVector(landmarks[FACE_INDICES.RIGHT_EYE_TOP]), toVector(landmarks[FACE_INDICES.RIGHT_EYE_BOTTOM]));
      lShape = { w: lW, h: lH };
      rShape = { w: rW, h: rH };
    }

    if (deviceRef.current && uniformBufferRef.current && pipelineRef.current && bindGroupRef.current && contextRef.current) {
        const floatData = new Float32Array([
            lEyeView.x, lEyeView.y, lEyeView.z, 0,      
            rEyeView.x, rEyeView.y, rEyeView.z, 0,      
            lGazeEnd.x, lGazeEnd.y, lGazeEnd.z, 0,      
            rGazeEnd.x, rGazeEnd.y, rGazeEnd.z, 0,      
            lShape.w, lShape.h, 0, 0,                   
            rShape.w, rShape.h, 0, 0,                   
            videoRef.current.videoWidth, videoRef.current.videoHeight, 
            now / 1000.0, isDistractedVal,              
        ]);

        deviceRef.current.queue.writeBuffer(uniformBufferRef.current, 0, floatData);

        const encoder = deviceRef.current.createCommandEncoder();
        const view = contextRef.current.getCurrentTexture().createView();
        
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: view,
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
                loadOp: 'clear',
                storeOp: 'store',
            }]
        });

        pass.setPipeline(pipelineRef.current);
        pass.setBindGroup(0, bindGroupRef.current);
        pass.draw(6);

        if (meshPipelineRef.current && vertexBufferRef.current && indexBufferRef.current && indexCountRef.current > 0) {
            pass.setPipeline(meshPipelineRef.current);
            pass.setBindGroup(0, bindGroupRef.current); 
            pass.setVertexBuffer(0, vertexBufferRef.current);
            pass.setIndexBuffer(indexBufferRef.current, 'uint16');
            pass.drawIndexed(indexCountRef.current);
        }

        pass.end();
        deviceRef.current.queue.submit([encoder.finish()]);
    }

    requestRef.current = requestAnimationFrame(predict);
  }, [isActive, calibrationOffset, status]);


  useEffect(() => {
    if (isActive) {
      const getMedia = async () => {
         try {
           const stream = await navigator.mediaDevices.getUserMedia({ 
             video: { 
                 width: { ideal: 1280 },
                 height: { ideal: 720 },
                 facingMode: 'user' 
             } 
           });
           if (videoRef.current) {
             videoRef.current.srcObject = stream;
           }
         } catch(e) {
           console.error("Camera denied", e);
           setStatus(TrackingStatus.ERROR);
         }
      };
      getMedia();
    } else {
       if (videoRef.current && videoRef.current.srcObject) {
         const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
         tracks.forEach(t => t.stop());
         videoRef.current.srcObject = null;
       }
       cancelAnimationFrame(requestRef.current);
       setStatus(TrackingStatus.IDLE);
    }
  }, [isActive]);


  return (
    <div className="relative w-full h-full flex justify-center items-center bg-black rounded-lg overflow-hidden border border-slate-800 shadow-2xl">
      {/* Aspect Ratio Locked Container */}
      <div className="relative max-w-full max-h-[600px]" style={{ aspectRatio: videoAspect }}>
          {/* Video Feed */}
          {/* object-cover ensures it fills the aspect-locked container perfectly */}
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            onLoadedMetadata={handleVideoLoad}
            className="w-full h-full object-cover scale-x-[-1]"
            style={{ opacity: isActive ? 0.6 : 0.1 }}
          />

          {/* WebGPU Overlay */}
          {/* Matches container size exactly */}
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full pointer-events-none mix-blend-screen"
          />
      </div>

      {/* HUD Overlay */}
      <div className="absolute top-4 left-4 font-mono text-xs text-indigo-400 space-y-1">
        <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${status === 'TRACKING' ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`}></div>
            <span>{status}</span>
        </div>
        <div>GPU: {deviceRef.current ? 'ACTIVE' : 'INITIALIZING...'}</div>
        <div>CALIBRATION: {calibrationOffset.x.toFixed(2)}, {calibrationOffset.y.toFixed(2)}</div>
      </div>
      
      {!isActive && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="text-slate-500 font-mono tracking-[0.2em] animate-pulse">SYSTEM STANDBY</div>
          </div>
      )}
    </div>
  );
};
