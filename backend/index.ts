import "dotenv/config";
import { GoogleGenAI } from "@google/genai";
import { Agent, handoff, run, tool } from "@openai/agents";
import { z } from "zod";
import express from "express";
import cors from "cors";
import fs from "fs/promises";
import path from "path";
import { createWriteStream } from "fs";

const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

type VideoStyle = "hype" | "ad" | "cinematic";

// Create output directory if it doesn't exist
const OUTPUT_DIR = path.join(process.cwd(), "output");

async function ensureOutputDir() {
  try {
    await fs.access(OUTPUT_DIR);
  } catch {
    await fs.mkdir(OUTPUT_DIR, { recursive: true });
    console.log(`Created output directory: ${OUTPUT_DIR}`);
  }
}

// Download video from URI and save to file
async function downloadVideo(uri: string, filename: string): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  const urlWithKey = `${uri}&key=${apiKey}`;

  console.log(`Downloading video from: ${uri}`);

  const response = await fetch(urlWithKey);
  if (!response.ok) {
    throw new Error(`Failed to download video: ${response.statusText}`);
  }

  const filePath = path.join(OUTPUT_DIR, filename);
  const fileStream = createWriteStream(filePath);

  if (!response.body) {
    throw new Error("No response body");
  }

  const reader = response.body.getReader();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      fileStream.write(value);
    }
  } finally {
    fileStream.end();
  }

  console.log(`Video saved to: ${filePath}`);
  return filePath;
}

async function generateVideo({
  prompt,
  style,
  narrationPrompt,
  visualPrompt,
}: {
  prompt: string;
  style: VideoStyle;
  narrationPrompt: string;
  visualPrompt: string;
}): Promise<{ filePaths: string[] }> {
  console.log(
    `[${style.toUpperCase()}] Starting video generation for prompt: ${prompt}`
  );

  // Ensure output directory exists
  await ensureOutputDir();

  console.log(`[${style.toUpperCase()}] Generating narration...`);
  const narrationRes = await genAI.models.generateContent({
    model: "gemini-2.5-flash",
    contents: [narrationPrompt],
  });

  const narration = narrationRes.text?.replace(/\*/g, "").trim();
  if (!narration) {
    console.error(`[${style.toUpperCase()}] Failed to generate narration text`);
    throw new Error("Failed to generate narration text");
  }
  console.log(
    `[${style.toUpperCase()}] Narration generated: ${narration.substring(
      0,
      100
    )}...`
  );

  console.log(`[${style.toUpperCase()}] Generating image...`);
  const imageResp = await genAI.models.generateImages({
    model: "imagen-3.0-generate-002",
    prompt: visualPrompt,
    config: { numberOfImages: 1 },
  });

  const image = imageResp.generatedImages?.[0]?.image;
  if (!image?.imageBytes) {
    console.error(`[${style.toUpperCase()}] Failed to generate image`);
    throw new Error("Failed to generate image");
  }
  console.log(`[${style.toUpperCase()}] Image generated successfully`);

  console.log(`[${style.toUpperCase()}] Starting video generation with Veo...`);
  let op = await genAI.models.generateVideos({
    model: "veo-2.0-generate-001",
    prompt: narration,
    image: {
      imageBytes: image.imageBytes,
      mimeType: "image/png",
    },
    config: {
      aspectRatio: "9:16",
      numberOfVideos: 2,
    },
  });

  let pollCount = 0;
  while (!op.done) {
    pollCount++;
    console.log(
      `[${style.toUpperCase()}] Waiting for video generation... (poll #${pollCount})`
    );
    await new Promise((r) => setTimeout(r, 10000));
    op = await genAI.operations.getVideosOperation({ operation: op });
  }

  const videos = op.response?.generatedVideos;
  if (!videos || videos.length === 0) {
    console.error(
      `[${style.toUpperCase()}] Video generation failed - no videos returned`
    );
    throw new Error("Video generation failed");
  }

  // Download and save videos
  const filePaths: string[] = [];
  for (let i = 0; i < videos.length; i++) {
    const uri = videos[i]?.video?.uri;
    if (uri) {
      console.log(
        `[${style.toUpperCase()}] Video ${i + 1} URI generated: ${uri}`
      );

      // Generate filename with timestamp and style
      const timestamp = Date.now();
      const filename = `${style}_video_${timestamp}_${i + 1}.mp4`;

      try {
        const filePath = await downloadVideo(uri, filename);
        filePaths.push(filePath);
        console.log(
          `[${style.toUpperCase()}] Video ${i + 1} saved to: ${filePath}`
        );
      } catch (error) {
        console.error(
          `[${style.toUpperCase()}] Failed to download video ${i + 1}:`,
          error
        );
        // Continue with other videos even if one fails
      }
    }
  }

  if (filePaths.length === 0) {
    throw new Error("Failed to download any videos");
  }

  console.log(
    `[${style.toUpperCase()}] Video generation complete - ${
      filePaths.length
    } videos saved`
  );
  return { filePaths };
}

const PromptSchema = z.object({ prompt: z.string() });

const createHypeAgent = () =>
  new Agent<{ prompt: string }>({
    name: "Hype Video Agent",
    instructions:
      "You create fast-paced, high-energy hype videos that get people pumped up. Use bold visuals and powerful, energetic narration.",
    tools: [
      tool({
        name: "generate_hype_video",
        description:
          "Generate a fast-paced, high-energy hype video from a user prompt.",
        parameters: PromptSchema,
        execute: async ({ prompt }) => {
          console.log(`[HYPE] Starting hype video generation for: ${prompt}`);
          return await generateVideo({
            prompt,
            style: "hype",
            narrationPrompt: `Create a high-octane, adrenaline-pumping voiceover script for: ${prompt}. Use short sentences, punchy verbs, and crowd-rallying phrases.`,
            visualPrompt: `Design energetic, flashy visuals with quick cuts, bold typography, vibrant motion graphics, and fast transitions — all themed around: ${prompt}`,
          });
        },
      }),
    ],
  });

const createAdAgent = () =>
  new Agent<{ prompt: string }>({
    name: "Ad Video Agent",
    instructions:
      "You create professional, sleek promotional videos that highlight products, services, or ideas in a clean and marketable way.",
    tools: [
      tool({
        name: "generate_ad_video",
        description: "Generate a clean and polished promotional ad video.",
        parameters: PromptSchema,
        execute: async ({ prompt }) => {
          console.log(`[AD] Starting ad video generation for: ${prompt}`);
          return await generateVideo({
            prompt,
            style: "ad",
            narrationPrompt: `Write a clear, persuasive product ad script for: ${prompt}. Focus on key features, benefits, and a strong call to action. Keep it brand-friendly and concise.`,
            visualPrompt: `Sleek, minimal commercial visuals for: ${prompt}. Use clean lighting, product showcases, soft motion effects, and whitespace.`,
          });
        },
      }),
    ],
  });

const createCinematicAgent = () =>
  new Agent<{ prompt: string }>({
    name: "Cinematic Video Agent",
    instructions:
      "You create emotional, story-driven cinematic videos with a poetic tone and visually rich atmosphere.",
    tools: [
      tool({
        name: "generate_cinematic_video",
        description: "Generate a visually rich, emotional cinematic video.",
        parameters: PromptSchema,
        execute: async ({ prompt }) => {
          console.log(
            `[CINEMATIC] Starting cinematic video generation for: ${prompt}`
          );
          return await generateVideo({
            prompt,
            style: "cinematic",
            narrationPrompt: `Craft a poetic, emotionally resonant script (3–5 lines) that captures the essence of: ${prompt}. Use metaphors, vivid imagery, and a soft, reflective tone.`,
            visualPrompt: `Create visually cinematic, atmospheric visuals for: ${prompt}. Use slow motion, natural lighting, deep contrast, and wide shots to evoke emotion.`,
          });
        },
      }),
    ],
  });

const createTriageAgent = () => {
  const HypeAgent = createHypeAgent();
  const AdAgent = createAdAgent();
  const CinematicAgent = createCinematicAgent();

  return new Agent({
    name: "Triage Agent",
    instructions: `
You are a video director assistant.
Based on the user's input prompt, decide the best style and hand off:
- Hype for excitement
- Ad for product promotions
- Cinematic for emotional stories`,
    handoffs: [
      handoff(HypeAgent, {
        toolNameOverride: "use_hype_tool",
        toolDescriptionOverride: "Send to hype video agent.",
      }),
      handoff(AdAgent, {
        toolNameOverride: "use_ad_tool",
        toolDescriptionOverride: "Send to ad video agent.",
      }),
      handoff(CinematicAgent, {
        toolNameOverride: "use_cinematic_tool",
        toolDescriptionOverride: "Send to cinematic video agent.",
      }),
    ],
  });
};

const app = express();
app.use(cors());
app.use(express.json());

app.post("/api/generate-video", async (req: any, res: any) => {
  const { prompt } = req.body;
  console.log(`[API] New video generation request: ${prompt}`);

  if (!prompt) {
    console.error("[API] No prompt provided");
    return res.status(400).json({ error: "Prompt is required" });
  }

  try {
    console.log("[API] Initializing triage agent...");
    const TriageAgent = createTriageAgent();

    console.log("[API] Running agent with prompt...");
    const result = await run(TriageAgent, prompt);
    const output = await result.finalOutput;

    if (!output) {
      console.error("[API] No output received from agent");
      throw new Error("No output received from agent");
    }

    console.log(
      "[API] Agent completed successfully, returning file paths:",
      output
    );
    res.json(output);
  } catch (error) {
    console.error("[API] Error in video generation:", error);
    res.status(500).json({
      error:
        error instanceof Error ? error.message : "Failed to generate video",
    });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
