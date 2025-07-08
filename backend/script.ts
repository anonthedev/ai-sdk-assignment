import "dotenv/config";
import { GoogleGenAI } from "@google/genai";
import { Agent, handoff, run, tool } from "@openai/agents";
import { z } from "zod";
import fs from "fs/promises";
import path from "path";
import { createWriteStream } from "fs";
import { exec } from "child_process";
import { promisify } from "util";

const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const execAsync = promisify(exec);

type VideoStyle = "hype" | "ad" | "cinematic";

// Create temp directory if it doesn't exist
const TEMP_DIR = "/tmp";

async function ensureTempDir() {
  try {
    await fs.access(TEMP_DIR);
  } catch {
    await fs.mkdir(TEMP_DIR, { recursive: true });
    console.log(`Created temp directory: ${TEMP_DIR}`);
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

  const filePath = path.join(TEMP_DIR, filename);
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
}): Promise<{ videoPath: string; imagePath: string }> {
  console.log(
    `[${style.toUpperCase()}] Starting video generation for prompt: ${prompt}`
  );

  // Ensure temp directory exists
  await ensureTempDir();

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

  // Save image to file
  const imagePath = path.join(TEMP_DIR, `${style}_image_${Date.now()}.png`);
  const imageBuffer = Buffer.from(image.imageBytes, "base64");
  await fs.writeFile(imagePath, imageBuffer);
  console.log(`[${style.toUpperCase()}] Image saved to: ${imagePath}`);

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

  let finalVideoPath: string;

  if (filePaths.length === 1) {
    console.log(
      `[${style.toUpperCase()}] Only one video downloaded, using as final output`
    );
    finalVideoPath = filePaths[0];
  } else {
    console.log(
      `[${style.toUpperCase()}] Concatenating ${filePaths.length} videos...`
    );

    // Create concat list file
    const listPath = path.join(TEMP_DIR, `${style}_concat_list.txt`);
    const listContent = filePaths.map((p) => `file '${p}'`).join("\n");
    await fs.writeFile(listPath, listContent);

    // Concatenate videos using ffmpeg
    finalVideoPath = path.join(TEMP_DIR, `${style}_final_${Date.now()}.mp4`);
    const { stdout, stderr } = await execAsync(
      `ffmpeg -f concat -safe 0 -i ${listPath} -c copy ${finalVideoPath}`
    );

    console.log(
      `[${style.toUpperCase()}] Videos concatenated successfully: ${finalVideoPath}`
    );
  }

  console.log(
    `[${style.toUpperCase()}] Video generation complete - saved to ${finalVideoPath}`
  );
  return { videoPath: finalVideoPath, imagePath };
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

async function main() {
  if (!process.argv[2]) {
    console.error("Please provide a prompt as a command line argument");
    console.error("Example: npm run script 'my video prompt'");
    process.exit(1);
  }

  const prompt = process.argv[2];
  console.log(`[SCRIPT] Starting video generation for prompt: ${prompt}`);

  try {
    console.log("[SCRIPT] Initializing triage agent...");
    const TriageAgent = createTriageAgent();

    console.log("[SCRIPT] Running agent with prompt...");
    const result = await run(TriageAgent, prompt);
    const output = await result.finalOutput;

    if (!output) {
      console.error("[SCRIPT] No output received from agent");
      throw new Error("No output received from agent");
    }

    if (typeof output === "string") {
      console.log("\n✅ FINAL OUTPUT:", output);
      console.log("No video was generated.");
    } else {
      const videoOutput = output as { videoPath: string; imagePath: string };

      if (videoOutput.videoPath && videoOutput.imagePath) {
        console.log("\n✅ VIDEO GENERATED:", videoOutput.videoPath);
        console.log("✅ IMAGE GENERATED:", videoOutput.imagePath);
      } else if (videoOutput.imagePath) {
        console.log("\n✅ IMAGE GENERATED:", videoOutput.imagePath);
        console.log("❌ No video was generated.");
      } else {
        console.log("\n❌ No content was generated.");
      }
    }
  } catch (error) {
    console.error("[SCRIPT] Error in video generation:", error);
    process.exit(1);
  }
}

if (require.main === module) {
  main().catch((err) => {
    console.error("Fatal error:", err);
    process.exit(1);
  });
}