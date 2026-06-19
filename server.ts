import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";

const app = express();
const PORT = 3000;

app.use(express.json({ limit: "15mb" }));

// Lazy initializer for GoogleGenAI to protect dev server start if API keys are not supplied.
function getGenAI() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error(
      "GEMINI_API_KEY environment variable is missing. Please define it in the Secrets panel in AI Studio."
    );
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

// Wrapper function to invoke Gemini with temporary error retry and model fallback
async function generateContentWithFallback(params: {
  contents: string;
  config: any;
}) {
  const models = ["gemini-3.5-flash", "gemini-flash-latest", "gemini-3.1-flash-lite"];
  let lastError: any = null;

  for (const model of models) {
    const maxAttempts = 3;
    let delay = 1000; // start with 1 second

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        console.log(`[Gemini] Requesting ${model} (Attempt ${attempt}/${maxAttempts})...`);
        const ai = getGenAI();
        const response = await ai.models.generateContent({
          model,
          contents: params.contents,
          config: params.config,
        });
        return response;
      } catch (error: any) {
        lastError = error;
        const msg = (error?.message || "").toLowerCase();
        const status = error?.status || error?.code || 0;

        // Check if retryable (e.g. 503, 429, high demand, rate limits, resource exhausted, overloaded)
        const isRetryable =
          status === 503 ||
          status === 429 ||
          msg.includes("503") ||
          msg.includes("429") ||
          msg.includes("demand") ||
          msg.includes("overloaded") ||
          msg.includes("unavailable") ||
          msg.includes("exhausted") ||
          msg.includes("limit");

        if (isRetryable && attempt < maxAttempts) {
          console.warn(`[Gemini Retry] ${model} temporary failure. Retrying in ${delay}ms... Error: ${error.message}`);
          await new Promise((resolve) => setTimeout(resolve, delay));
          delay *= 2; // exponential backoff
        } else {
          console.warn(`[Gemini Fail] ${model} attempt failed or non-retryable. Moving forward.`);
          break; // break retry loop, move to next model or fail
        }
      }
    }
  }

  throw lastError || new Error("All generative AI models exhausted.");
}

// REST route to leverage Gemini for parsing text transcripts or custom topics into structured syllabus objects.
app.post("/api/generate-syllabus", async (req, res) => {
  try {
    const { prompt, totalChapters } = req.body;
    if (!prompt) {
      return res
        .status(400)
        .json({ error: "No prompt text provided for syllabus generation." });
    }

    const systemInstruction = `You are an expert academic planning coordinator and syllabus parser.
Your task is to take a course syllabus, list of topics, textbook outline, or study request, and return a robust JSON format.
Strictly structure it as a sequence of logically sound, chronological chapters and lectures.
If the user specifies a preference (e.g., target ${
      totalChapters || "some"
    } chapters), conform to it.
For each lecture, estimate the reasonable study duration required for a detailed understanding (e.g., "30 min", "1 hour", "1.5 hours", "2 hours", etc.).`;

    const userPromptText = `Parse or create a study path for:
"${prompt}"

Structure the chapters sequentially. For each chapter, compile an array of specific, distinct lectures. Ensure each lecture has a title and a time estimate. Make the titles human-readable and detailed.`;

    const response = await generateContentWithFallback({
      contents: userPromptText,
      config: {
        systemInstruction,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            chapters: {
              type: Type.ARRAY,
              description: "Array of structured chapter roadmaps for the planner.",
              items: {
                type: Type.OBJECT,
                properties: {
                  chapter_name: {
                    type: Type.STRING,
                    description: "High-level name of the chapter or study unit.",
                  },
                  lectures: {
                    type: Type.ARRAY,
                    description: "Chronological lectures under this chapter.",
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        lecture_name: {
                          type: Type.STRING,
                          description: "Specific title or topic of this lecture.",
                        },
                        time_estimate: {
                          type: Type.STRING,
                          description: "Estimated study duration, e.g. '1 hour' or '45 min'.",
                        },
                      },
                      required: ["lecture_name", "time_estimate"],
                    },
                  },
                },
                required: ["chapter_name", "lectures"],
              },
            },
          },
          required: ["chapters"],
        },
      },
    });

    const outputText = response.text;
    if (!outputText) {
      throw new Error("Gemini API did not yield text output.");
    }

    const payload = JSON.parse(outputText);
    return res.json(payload);
  } catch (error: any) {
    console.error("Error generated during server-side Gemini invocation:", error);
    
    // Check if it's the high-demand error so we can format it nicely and clearly for the user
    const errorMsg = error?.message || "";
    if (errorMsg.includes("high demand") || errorMsg.includes("503") || errorMsg.includes("UNAVAILABLE")) {
      return res.status(503).json({
        error: "Study Planner is currently under high load. Spikes in demand are usually temporary. Please click 'Generate' again to retry.",
      });
    }

    return res.status(500).json({
      error:
        errorMsg ||
        "An unexpected error occurred while compiling your study plan with AI.",
    });
  }
});

// Configure Vite integration for dev server or fallback static bundle serving for production builds.
async function initServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[StudyPlanner] Server booted successfully and running at http://0.0.0.0:${PORT}`);
  });
}

initServer();
