import "dotenv/config";
import { GoogleGenAI } from "@google/genai";
import { Agent, handoff, run, tool } from "@openai/agents";
import { z } from "zod";
import { createWriteStream } from "fs";
import { Readable } from "stream";
import { exec } from "child_process";
import { promisify } from "util";
import fs from "fs/promises";

const log = {
  info: (msg: string, data?: any) => {
    console.log(`[INFO] ${msg}`, data ? data : "");
    return { type: "info", message: msg, data };
  },
  error: (msg: string, error?: any) => {
    console.error(`[ERROR] ${msg}`, error ? error : "");
    return { type: "error", message: msg, data: error };
  },
  warn: (msg: string, data?: any) => {
    console.warn(`[WARN] ${msg}`, data ? data : "");
    return { type: "warn", message: msg, data };
  },
  debug: (msg: string, data?: any) => {
    console.debug(`[DEBUG] ${msg}`, data ? data : "");
    return { type: "debug", message: msg, data };
  },
};

const execAsync = promisify(exec);
const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

type VideoStyle = "hype" | "ad" | "cinematic";

const styleConfigs = {
  hype: {
    narrationPrompt: (prompt: string) => `Write a high-energy hype script with action verbs and excitement for: ${prompt}`,
    visualPrompt: (prompt: string) => `Energetic visuals for: ${prompt}, with bold colors and motion.`
  },
  ad: {
    narrationPrompt: (prompt: string) => `Write a professional, product-focused ad script for: ${prompt}`,
    visualPrompt: (prompt: string) => `Clean, commercial visuals for: ${prompt}`
  },
  cinematic: {
    narrationPrompt: (prompt: string) => `Write a poetic cinematic script (3-5 short lines) about: ${prompt}`,
    visualPrompt: (prompt: string) => `Moody, dramatic film-style visuals for: ${prompt}`
  }
};

async function generateVideo(prompt: string, style: VideoStyle): Promise<{videoPath: string, imagePath: string}> {
  const config = styleConfigs[style];
  log.info(`Starting video generation for style: ${style}`);

  try {
    log.info("Generating narration with Gemini");
    const narrationRes = await genAI.models.generateContent({
      model: "gemini-2.5-flash",
      contents: [config.narrationPrompt(prompt)],
    });

    const narration = narrationRes.text?.replace(/\*/g, "").trim();
    log.debug("Generated narration", { narration });

    if (!narration) {
      log.error("Failed to generate narration text");
      throw new Error("Failed to generate narration text");
    }

    log.info("Generating image with Imagen");
    const imageResp = await genAI.models.generateImages({
      model: "imagen-3.0-generate-002",
      prompt: config.visualPrompt(prompt),
      config: { numberOfImages: 1 },
    });
    log.debug("Image generation response received");

    const image = imageResp.generatedImages?.[0]?.image;
    if (!image?.imageBytes) {
      log.error("Failed to generate image - no imageBytes in response");
      throw new Error("❌ Failed to generate image.");
    }
    log.info("Image generated successfully");

    const imagePath = `/tmp/${style}_image_${Date.now()}.png`;
    log.info("Saving image to temporary file", { path: imagePath });
    const imageBuffer = Buffer.from(image.imageBytes, 'base64');
    await fs.writeFile(imagePath, imageBuffer);
    log.debug("Image saved successfully", { path: imagePath });

    log.info("Starting video generation with Veo");
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
    log.debug("Initial video operation response", {
      operationName: op.name,
      done: op.done,
    });

    let pollCount = 0;
    while (!op.done) {
      pollCount++;
      log.info(`⏳ Generating Veo video... (poll #${pollCount})`);
      await new Promise((r) => setTimeout(r, 10000));
      op = await genAI.operations.getVideosOperation({ operation: op });
      log.debug(`Poll #${pollCount} result`, { done: op.done });
    }

    const videos = op.response?.generatedVideos;
    log.debug("Video generation complete", {
      videoCount: videos?.length,
      operationName: op.name,
      hasResponse: !!op.response,
      responseKeys: op.response ? Object.keys(op.response) : [],
      error: op.error,
      metadata: op.metadata
    });

    if (!videos || videos.length === 0) {
      const errorMsg = "Video generation failed: The API did not return any video data";
      log.error(errorMsg, { 
        apiError: op.error,
        response: op.response
      });
      throw new Error(errorMsg);
    }

    const paths: string[] = [];
    log.info("Starting video downloads", { requested: 2, found: videos.length });
    
    const videoCount = Math.min(videos.length, 2);
    for (let i = 0; i < videoCount; i++) {
      try {
        log.debug(`Downloading video ${i + 1}/${videoCount}`);
        const uri = videos[i]?.video?.uri;

        if (!uri) {
          log.error(`No URI found for video ${i + 1}`);
          continue;
        }

        log.debug(`Fetching video from URI: ${uri.substring(0, 30)}...`);
        const res = await fetch(`${uri}&key=${process.env.GEMINI_API_KEY}`);

        if (!res.ok) {
          log.error(`Failed to fetch video ${i + 1}`, { status: res.status });
          continue;
        }

        const outPath = `/tmp/${style}_${Date.now()}_${i}.mp4`;
        log.debug(`Writing video ${i + 1} to: ${outPath}`);

        const writer = createWriteStream(outPath);
        Readable.fromWeb(res.body as any).pipe(writer);

        await new Promise<void>((res, rej) => {
          writer.on("finish", () => {
            log.debug(`Video ${i + 1} written successfully`);
            res();
          });
          writer.on("error", (err) => {
            log.error(`Error writing video ${i + 1}`, err);
            rej(err);
          });
        });

        paths.push(outPath);
      } catch (err) {
        log.error(`Error processing video ${i + 1}`, err);
      }
    }

    if (paths.length === 0) {
      log.warn("Video download failed for all clips.");
      throw new Error("Video download failed for all clips.");
    }

    if (paths.length === 1) {
      log.info("Only one video was downloaded successfully, skipping concatenation");
      return { videoPath: paths[0], imagePath };
    }

    log.info("Creating concat list for", { videoCount: paths.length });
    const listPath = `/tmp/${style}_concat_list.txt`;
    const listContent = paths.map((p) => `file '${p}'`).join("\n");
    await fs.writeFile(listPath, listContent);
    log.debug("Created concat list", { path: listPath, content: listContent });

    const finalPath = `/tmp/${style}_final_${Date.now()}.mp4`;
    log.info("Running ffmpeg to concatenate videos", {
      command: `ffmpeg -f concat -safe 0 -i ${listPath} -c copy ${finalPath}`,
    });

    const { stdout, stderr } = await execAsync(
      `ffmpeg -f concat -safe 0 -i ${listPath} -c copy ${finalPath}`
    );
    log.debug("ffmpeg output", { stdout, stderr });

    log.info("Video generation complete", { finalPath, imagePath });
    return { videoPath: finalPath, imagePath };
  } catch (error) {
    log.error(`Error in generateVideo for style ${style}`, error);
    throw error;
  }
}

const PromptSchema = z.object({ prompt: z.string() });

const createVideoTool = (style: VideoStyle) => tool({
  name: `generate_${style}_video`,
  description: `Generate a ${style} video from a user prompt.`,
  parameters: PromptSchema,
  execute: async ({ prompt }) => {
    log.info(`${style} video tool called with prompt`, { prompt });
    try {
      const result = await generateVideo(prompt, style);
      log.info(`${style} video generated successfully`, result);
      return result;
    } catch (error) {
      log.error(`Error generating ${style} video`, error);
      throw error;
    }
  },
});

const HypeAgent = new Agent({
  name: "Hype Video Agent",
  instructions: "You create fast-paced, high-energy hype videos using your tools.",
  tools: [createVideoTool("hype")],
});

const AdAgent = new Agent({
  name: "Ad Video Agent", 
  instructions: "You create clean and polished promotional ad videos using your tools.",
  tools: [createVideoTool("ad")],
});

const CinematicAgent = new Agent({
  name: "Cinematic Video Agent",
  instructions: "You create emotional, cinematic-style storytelling videos using your tools.",
  tools: [createVideoTool("cinematic")],
});

const TriageAgent = new Agent({
  name: "Triage Agent",
  instructions: `You are a video director assistant.
Based on the user's input prompt, decide the best style and hand off:
- Hype for excitement, energy, action
- Ad for product promotions, marketing
- Cinematic for emotional stories, dramatic content`,
  handoffs: [
    handoff(HypeAgent, {
      toolNameOverride: "use_hype_tool",
      toolDescriptionOverride: "Send to hype video agent for high-energy content.",
    }),
    handoff(AdAgent, {
      toolNameOverride: "use_ad_tool", 
      toolDescriptionOverride: "Send to ad video agent for promotional content.",
    }),
    handoff(CinematicAgent, {
      toolNameOverride: "use_cinematic_tool",
      toolDescriptionOverride: "Send to cinematic video agent for emotional storytelling.",
    }),
  ],
});

async function main() {
  if (!process.argv[2]) {
    console.error("Please provide a prompt as a command line argument");
    console.error("Example: node script.js 'my video prompt'");
    process.exit(1);
  }

  const prompt = process.argv[2];
  log.info("Starting video generation process", { prompt });
  
  try {
    log.debug("Environment check", {
      geminiKeyExists: !!process.env.GEMINI_API_KEY,
    });

    log.debug("Running triage agent with prompt");
    const result = await run(TriageAgent, prompt);
    log.debug("Triage agent run complete, awaiting final output");
    
    const output = await result.finalOutput;
    if (!output) {
      log.error("No output received from agent");
      throw new Error("No output received from agent");
    }
    
    if (typeof output === "string") {
      console.log("\n✅ FINAL OUTPUT:", output);
      console.log("No video was generated.");
    } else {
      const videoOutput = output as { videoPath: string, imagePath: string };
      
      if (videoOutput.videoPath) {
        console.log("\n✅ VIDEO GENERATED: " + videoOutput.videoPath);
        console.log("✅ IMAGE GENERATED: " + videoOutput.imagePath);
      } else {
        console.log("\n✅ IMAGE GENERATED: " + videoOutput.imagePath);
        console.log("❌ No video was generated.");
      }
    }
  } catch (error) {
    log.error("Error in video creation process", error);
    process.exit(1);
  }
}

export { generateVideo, TriageAgent, HypeAgent, AdAgent, CinematicAgent };

if (require.main === module) {
  main().catch(err => {
    console.error("Fatal error:", err);
    process.exit(1);
  });
}