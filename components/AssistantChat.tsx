import React, { useState, useEffect, useRef } from 'react';
import { generateFocusCoaching } from '../services/geminiService';
import { ChatMessage, FocusSession } from '../types';

interface AssistantChatProps {
  sessionStats: FocusSession;
}

export const AssistantChat: React.FC<AssistantChatProps> = ({ sessionStats }) => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      role: 'model',
      text: "Oculus System Online. Monitoring visual attention patterns.",
      timestamp: new Date()
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      text: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    const responseText = await generateFocusCoaching(sessionStats, input);

    const aiMsg: ChatMessage = {
      id: (Date.now() + 1).toString(),
      role: 'model',
      text: responseText,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, aiMsg]);
    setIsLoading(false);
  };

  return (
    <div className="flex flex-col h-[500px] bg-slate-900 rounded-lg border border-slate-800 shadow-2xl overflow-hidden font-mono">
      {/* Header */}
      <div className="bg-slate-950 p-4 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
            <h3 className="font-bold text-slate-200 text-sm tracking-wider">AI NEURAL LINK</h3>
        </div>
        <span className="text-[10px] text-slate-500">GEMINI 2.5 FLASH</span>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-hide bg-slate-900">
        {messages.map((msg) => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] p-3 text-xs leading-relaxed border ${
              msg.role === 'user' 
                ? 'bg-indigo-900/30 text-indigo-200 border-indigo-500/50 rounded-tl-lg rounded-bl-lg rounded-tr-lg' 
                : 'bg-slate-800/50 text-slate-300 border-slate-700 rounded-tr-lg rounded-br-lg rounded-tl-lg'
            }`}>
              <p>{msg.text}</p>
            </div>
          </div>
        ))}
        {isLoading && (
           <div className="text-xs text-indigo-400 animate-pulse ml-2"> Processing...</div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-3 bg-slate-950 border-t border-slate-800">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Command..."
            className="flex-1 bg-slate-900 border border-slate-800 rounded px-3 py-2 focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 text-xs text-slate-200 placeholder-slate-600 outline-none"
          />
        </div>
      </div>
    </div>
  );
};