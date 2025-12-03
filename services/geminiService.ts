import { GoogleGenAI, Type } from "@google/genai";
import { GEMINI_MODEL } from "../constants";
import { FocusSession } from "../types";

const apiKey = process.env.API_KEY || '';
let ai: GoogleGenAI | null = null;

if (apiKey) {
  ai = new GoogleGenAI({ apiKey });
}

export const generateFocusCoaching = async (session: FocusSession, currentContext: string): Promise<string> => {
  if (!ai) return "API Key not configured.";

  const prompt = `
    You are an AI Attention Coach named "Oculus". 
    User Data:
    - Session Length: ${Math.round(session.totalSeconds)}s
    - Time Focused: ${Math.round((session.focusSeconds / session.totalSeconds) * 100)}%
    - Distraction Events: ${session.distractionCount}
    - Avg Attention Span: ${session.averageAttentionSpan.toFixed(1)}s
    
    User Context: "${currentContext}"
    
    If distraction count is high, suggest specific focus techniques (Pomodoro, Box Breathing).
    Be concise, slightly robotic/cyberpunk in tone but helpful. Max 60 words.
  `;

  try {
    const response = await ai.models.generateContent({
      model: GEMINI_MODEL,
      contents: prompt,
      config: {
        systemInstruction: "You are a high-tech attention optimization interface.",
        temperature: 0.7,
      }
    });

    return response.text || "Systems offline.";
  } catch (error) {
    console.error("Gemini API Error:", error);
    return "Connection interrupted.";
  }
};

export const generateSessionSummary = async (session: FocusSession): Promise<{summary: string, score: number}> => {
    if (!ai) return { summary: "Data corrupted.", score: 0 };
    
    const prompt = `
      Analyze this focus session.
      Stats: Duration ${session.totalSeconds}s, Distractions ${session.distractionCount}, Focus % ${Math.round((session.focusSeconds/session.totalSeconds)*100)}.
      Return JSON: { "summary": "string", "score": number (0-100) }
    `;

    try {
        const response = await ai.models.generateContent({
            model: GEMINI_MODEL,
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        summary: { type: Type.STRING },
                        score: { type: Type.NUMBER }
                    }
                }
            }
        });
        
        const text = response.text;
        if (!text) throw new Error("No response text");
        return JSON.parse(text);
    } catch (e) {
        return { summary: "Analysis failed.", score: 0 };
    }
}