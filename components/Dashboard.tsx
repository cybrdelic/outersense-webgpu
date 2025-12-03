
import React from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { GazeMetrics, FocusSession } from '../types';

interface DashboardProps {
  currentMetrics: GazeMetrics | null;
  history: GazeMetrics[];
  sessionStats: FocusSession;
}

export const Dashboard: React.FC<DashboardProps> = ({ currentMetrics, history, sessionStats }) => {
  
  // High-performance mapping
  const chartData = history.slice(-60).map((h, i) => ({
    time: i,
    attention: h.isDistracted ? 0 : 100,
    yaw: h.yaw
  }));

  const focusScore = sessionStats.totalSeconds > 0 
    ? Math.round((sessionStats.focusSeconds / sessionStats.totalSeconds) * 100) 
    : 100;

  return (
    <div className="space-y-6">
      {/* Real-time Cyberpunk Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Status Card */}
        <div className={`p-4 rounded border-l-4 transition-colors duration-100 bg-slate-900 ${currentMetrics?.isDistracted ? 'border-red-500' : 'border-emerald-500'}`}>
           <h3 className="text-xs font-mono text-slate-400 uppercase">Current State</h3>
           <div className={`mt-1 text-xl font-bold font-mono ${currentMetrics?.isDistracted ? 'text-red-400' : 'text-emerald-400'}`}>
              {currentMetrics ? (currentMetrics.isDistracted ? 'DISTRACTED' : 'FOCUSED') : 'OFFLINE'}
           </div>
        </div>

        {/* Gaze Vector Numeric */}
        <div className="p-4 rounded border border-slate-800 bg-slate-900">
           <h3 className="text-xs font-mono text-slate-400 uppercase">Gaze Vector (X/Y)</h3>
           <div className="mt-1 text-xl font-bold font-mono text-indigo-400">
             {currentMetrics ? `${currentMetrics.gazeVector.x.toFixed(2)} / ${currentMetrics.gazeVector.y.toFixed(2)}` : '--'}
           </div>
        </div>

        {/* Attention Span */}
        <div className="p-4 rounded border border-slate-800 bg-slate-900">
           <h3 className="text-xs font-mono text-slate-400 uppercase">Focus Consistency</h3>
           <div className="mt-1 text-xl font-bold font-mono text-white">
             {focusScore}%
           </div>
        </div>

         {/* Eye Openness */}
         <div className="p-4 rounded border border-slate-800 bg-slate-900">
           <h3 className="text-xs font-mono text-slate-400 uppercase">Eye Status</h3>
           <div className="mt-1 text-xl font-bold font-mono text-white">
             {currentMetrics?.isBlinking ? 'BLINK' : 'OPEN'}
           </div>
        </div>
      </div>

      {/* Focus Timeline Chart */}
      <div className="bg-slate-900 rounded border border-slate-800 h-64 relative overflow-hidden">
        <div className="absolute top-4 left-4 z-20">
             <h3 className="text-xs font-mono text-slate-400 uppercase">Attention Waveform</h3>
        </div>
        
        {/* Grid Background */}
        <div className="absolute inset-0 opacity-10 pointer-events-none" 
             style={{backgroundImage: 'linear-gradient(#4f46e5 1px, transparent 1px), linear-gradient(90deg, #4f46e5 1px, transparent 1px)', backgroundSize: '20px 20px'}}>
        </div>

        {/* Responsive Container Wrapper with absolute positioning to guarantee dimensions */}
        <div className="absolute inset-0 top-10 bottom-2 left-2 right-2 z-10">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="colorFocus" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <YAxis hide domain={[0, 100]} />
              <Area 
                type="step" 
                dataKey="attention" 
                stroke="#10b981" 
                strokeWidth={2}
                fillOpacity={1} 
                fill="url(#colorFocus)" 
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
