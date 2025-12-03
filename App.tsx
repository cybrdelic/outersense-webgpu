

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { GazeTracker } from './components/CameraFeed'; // Renamed import
import { Dashboard } from './components/Dashboard';
import { AssistantChat } from './components/AssistantChat';
import { GazeMetrics, FocusSession } from './types';

const App: React.FC = () => {
  const [isActive, setIsActive] = useState(false);
  const [currentMetrics, setCurrentMetrics] = useState<GazeMetrics | null>(null);
  const [history, setHistory] = useState<GazeMetrics[]>([]);
  const [calibrationOffset, setCalibrationOffset] = useState({ x: 0, y: 0 });
  
  // Track continuous focus time for averages
  const lastDistractionTime = useRef<number>(Date.now());
  
  const [sessionStats, setSessionStats] = useState<FocusSession>({
    startTime: Date.now(),
    totalSeconds: 0,
    focusSeconds: 0,
    distractionCount: 0,
    averageAttentionSpan: 0,
  });

  const handleMetricsUpdate = useCallback((metrics: GazeMetrics) => {
    setCurrentMetrics(metrics);
    setHistory(prev => [...prev.slice(-100), metrics]);

    setSessionStats(prev => {
      const now = Date.now();
      const dt = 1/60; // Approx 60fps
      
      let newDistractionCount = prev.distractionCount;
      let newAvgSpan = prev.averageAttentionSpan;
      
      // Rising edge detection for distraction
      if (metrics.isDistracted && (Date.now() - lastDistractionTime.current > 2000)) {
         newDistractionCount += 1;
         const span = (now - lastDistractionTime.current) / 1000;
         newAvgSpan = (prev.averageAttentionSpan * prev.distractionCount + span) / (prev.distractionCount + 1);
         lastDistractionTime.current = now;
      } else if (metrics.isDistracted) {
        lastDistractionTime.current = now; // Reset timer while distracted
      }

      return {
        ...prev,
        totalSeconds: prev.totalSeconds + dt,
        focusSeconds: prev.focusSeconds + (metrics.isDistracted ? 0 : dt),
        distractionCount: newDistractionCount,
        averageAttentionSpan: newAvgSpan || 0
      };
    });
  }, []);

  const toggleSession = () => {
    if (!isActive) {
      setSessionStats({
        startTime: Date.now(),
        totalSeconds: 0,
        focusSeconds: 0,
        distractionCount: 0,
        averageAttentionSpan: 0
      });
      setHistory([]);
      lastDistractionTime.current = Date.now();
      setCalibrationOffset({ x: 0, y: 0 }); // Optional: Reset calibration on new session? Maybe keep it.
    }
    setIsActive(!isActive);
  };

  // Continuous Click Calibration Handler
  const handleCalibrationClick = (e: React.MouseEvent) => {
    if (!isActive || !currentMetrics) return;

    // 1. Where the user clicked (Target)
    const clientX = e.clientX;
    const clientY = e.clientY;
    const targetX = clientX / window.innerWidth;
    const targetY = clientY / window.innerHeight;

    // 2. Where the tracker thinks they are looking (Current Corrected)
    const currentX = currentMetrics.gazeX;
    const currentY = currentMetrics.gazeY;

    // 3. Calculate Correction needed
    // New Corrected = Old Corrected + Delta
    // Target = Old Corrected + Delta
    // Delta = Target - Old Corrected
    const deltaX = targetX - currentX;
    const deltaY = targetY - currentY;

    // 4. Update Calibration Offset
    setCalibrationOffset(prev => ({
      x: prev.x + deltaX,
      y: prev.y + deltaY
    }));

    // Optional: Visual Ripple or Feedback could happen here
    console.log(`Calibrated! Click: ${targetX.toFixed(2)},${targetY.toFixed(2)} | Gaze: ${currentX.toFixed(2)},${currentY.toFixed(2)} | Offset: ${deltaX.toFixed(2)},${deltaY.toFixed(2)}`);
  };

  return (
    <div 
      className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-indigo-500 selection:text-white cursor-crosshair"
      onClick={handleCalibrationClick}
    >
      {/* Navbar */}
      <nav className="border-b border-slate-800 bg-slate-950/50 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-14 items-center">
            <div className="flex items-center gap-2">
              <div className="text-indigo-500 font-bold text-xl tracking-tighter">
                OCULUS<span className="text-white">FOCUS</span>
              </div>
            </div>
            <div className="flex items-center gap-4">
               <span className="text-[10px] font-mono text-slate-500 uppercase border border-slate-800 px-2 py-1 rounded hidden sm:inline-block">
                 Click anywhere to calibrate
               </span>
               <button 
                  onClick={(e) => { e.stopPropagation(); toggleSession(); }} // Prevent calibration click on button
                  className={`px-4 py-1.5 rounded text-sm font-mono font-bold transition-all ${
                    isActive 
                    ? 'bg-red-500/10 text-red-500 border border-red-500/50 hover:bg-red-500/20' 
                    : 'bg-indigo-600 text-white hover:bg-indigo-500 hover:shadow-[0_0_15px_rgba(99,102,241,0.5)]'
                  }`}
                >
                  {isActive ? '[ TERMINATE SESSION ]' : '[ INITIALIZE TRACKING ]'}
                </button>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          
          {/* Main Visualizer */}
          <div className="xl:col-span-2 space-y-6">
            <div onClick={(e) => e.stopPropagation() /* prevent calibration on video click? or allow it? allow it. */}>
                 <GazeTracker 
                    isActive={isActive} 
                    onMetricsUpdate={handleMetricsUpdate} 
                    calibrationOffset={calibrationOffset}
                 />
            </div>
            <div onClick={(e) => e.stopPropagation()}>
                <Dashboard 
                currentMetrics={currentMetrics}
                history={history}
                sessionStats={sessionStats}
                />
            </div>
          </div>

          {/* Side Panel */}
          <div className="xl:col-span-1 space-y-6" onClick={(e) => e.stopPropagation()}>
            <AssistantChat sessionStats={sessionStats} />
            
            {/* Stats Panel */}
            <div className="bg-slate-900 border border-slate-800 rounded-lg p-5">
              <h4 className="text-xs font-mono text-slate-500 uppercase mb-4">Session Analytics</h4>
              <div className="space-y-4">
                <div className="flex justify-between items-end border-b border-slate-800 pb-2">
                  <span className="text-sm text-slate-400">Avg Attention Span</span>
                  <span className="font-mono text-indigo-400 font-bold">{sessionStats.averageAttentionSpan.toFixed(1)}s</span>
                </div>
                <div className="flex justify-between items-end border-b border-slate-800 pb-2">
                  <span className="text-sm text-slate-400">Distractions / Min</span>
                  <span className="font-mono text-indigo-400 font-bold">
                    {sessionStats.totalSeconds > 0 ? ((sessionStats.distractionCount / sessionStats.totalSeconds) * 60).toFixed(1) : '0'}
                  </span>
                </div>
              </div>
            </div>
            
            <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
                <h4 className="text-xs font-mono text-slate-500 uppercase mb-2">Calibration Data</h4>
                <div className="text-xs font-mono text-slate-400">
                    X Offset: <span className="text-white">{calibrationOffset.x.toFixed(3)}</span><br/>
                    Y Offset: <span className="text-white">{calibrationOffset.y.toFixed(3)}</span>
                </div>
                <button 
                    onClick={() => setCalibrationOffset({x:0, y:0})}
                    className="mt-2 text-[10px] bg-slate-800 hover:bg-slate-700 text-white px-2 py-1 rounded"
                >
                    RESET CALIBRATION
                </button>
            </div>
          </div>

        </div>
      </main>
    </div>
  );
};

export default App;