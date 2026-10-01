import express from "express";
import multer from "multer";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import crypto from "crypto";
import { execFile } from "child_process";
import { promisify } from "util";
import Groq from "groq-sdk";
import { GoogleGenAI } from "@google/genai";

dotenv.config();

const execFileAsync = promisify(execFile);

const app = express();

/* =========================================================
   PORT
========================================================= */

const PORT = Number(process.env.PORT) || 3000;

/* =========================================================
   DIRECTORIES
========================================================= */

const ROOT = process.cwd();

const PUBLIC_DIR = path.join(ROOT, "public");
const UPLOAD_DIR = path.join(ROOT, "uploads");
const JOB_DIR = path.join(ROOT, "jobs");

for (const dir of [
  PUBLIC_DIR,
  UPLOAD_DIR,
  JOB_DIR
]) {
  fs.mkdirSync(dir, {
    recursive: true
  });
}

/* =========================================================
   MODELS
========================================================= */

const GROQ_MODEL = "whisper-large-v3-turbo";
const GROQ_RECAP_MODEL = "openai/gpt-oss-120b";
const GEMINI_TTS_MODEL = "gemini-3.8-flash-lite-tts";

/* =========================================================
   SETTINGS
========================================================= */

const AUDIO_CHUNK_SECONDS = 90;
const SCENE_FPS = 2;
const MAX_SCENES = 18;
const MIN_SCENE_SECONDS = 3;
const MAX_VIDEO_SIZE = 500 * 1024 * 1024;

const GEMINI_MAX_RETRIES = 0;
const GEMINI_INITIAL_RETRY_DELAY = 2000;
const SCENE_THRESHOLD = 0.35;

/* =========================================================
   EXPRESS
========================================================= */

app.use(
  express.json({
    limit: "10mb"
  })
);

app.use(
  express.urlencoded({
    extended: true
  })
);

/* =========================================================
   MULTER
========================================================= */

const upload = multer({
  dest: UPLOAD_DIR,

  limits: {
    fileSize: MAX_VIDEO_SIZE
  },

  fileFilter: (req, file, cb) => {
    const allowed = [
      "video/mp4",
      "video/webm",
      "video/quicktime",
      "video/x-matroska",
      "video/x-msvideo"
    ];

    if (allowed.includes(file.mimetype)) {
      cb(null, true);
      return;
    }

    cb(
      new Error(
        "Only MP4, MKV, MOV, WEBM or AVI video files are allowed."
      )
    );
  }
});

/* =========================================================
   JOB STORAGE
========================================================= */

const jobs = new Map();

const JOB_DATA_DIR = path.join(
  JOB_DIR,
  "_data"
);

fs.mkdirSync(JOB_DATA_DIR, { recursive: true });

function getJobFile(id) {
  return path.join(JOB_DATA_DIR, `${id}.json`);
}

function saveJob(job) {
  try {
    fs.writeFileSync(
      getJobFile(job.id),
      JSON.stringify(job, null, 2),
      "utf8"
    );
  } catch (error) {
    console.error(`[JOB STORAGE] Failed to save ${job.id}:`, error);
  }
}

function loadJob(id) {
  const file = getJobFile(id);
  if (!fs.existsSync(file)) return null;

  try {
    const job = JSON.parse(fs.readFileSync(file, "utf8"));
    jobs.set(id, job);
    return job;
  } catch (error) {
    console.error(`[JOB STORAGE] Failed to load ${id}:`, error);
    return null;
  }
}

function loadAllJobs() {
  try {
    const files = fs.readdirSync(JOB_DATA_DIR);
    let loaded = 0;
    for (const file of files) {
      if (!file.endsWith(".json")) continue;
      if (loadJob(file.slice(0, -5))) loaded++;
    }
    console.log(`[JOB STORAGE] Loaded ${loaded} saved job(s).`);
  } catch (error) {
    console.error("[JOB STORAGE] Failed to load saved jobs:", error);
  }
}

function createJob() {
  const id = crypto.randomUUID();
  const job = {
    id,
    status: "created",
    stage: "Waiting",
    progress: 0,
    message: "Job created.",
    createdAt: new Date().toISOString(),
    duration: null,
    totalChunks: null,
    transcript: null,
    scenePlan: null,
    output: null,
    error: null
  };
  jobs.set(id, job);
  saveJob(job);
  return job;
}

function updateJob(id, data) {
  let job = jobs.get(id) || loadJob(id);
  if (!job) return;
  Object.assign(job, data);
  jobs.set(id, job);
  saveJob(job);
}

function getJob(id) {
  return jobs.get(id) || loadJob(id);
}

loadAllJobs();

/* =========================================================
   ENV
========================================================= */

function requireEnv(name) {
  const value = process.env[name];

  if (!value) {
    throw new Error(`${name} is not configured.`);
  }

  return value;
}

/* =========================================================
   COMMAND
========================================================= */

async function runCommand(command, args) {
  console.log(
    `[CMD] ${command} ${args.join(" ")}`
  );

  try {
    const result = await execFileAsync(
      command,
      args,
      {
        maxBuffer: 100 * 1024 * 1024
      }
    );

    return result;
  } catch (error) {
    console.error(
      `[CMD ERROR] ${command}`,
      error
    );

    throw new Error(
      `${command} failed: ${
        error.stderr ||
        error.message ||
        "Unknown error"
      }`
    );
  }
}

/* =========================================================
   SLEEP
========================================================= */

function sleep(ms) {
  return new Promise(resolve => {
    setTimeout(resolve, ms);
  });
}

/* =========================================================
   GEMINI RETRY
========================================================= */

function getGeminiErrorStatus(error) {
  const values = [
    error?.status,
    error?.code,
    error?.response?.status,
    error?.error?.status,
    error?.error?.code,
    error?.cause?.status,
    error?.cause?.code
  ];

  for (const value of values) {
    const number = Number(value);

    if (Number.isFinite(number)) {
      return number;
    }
  }

  const message = String(
    error?.message ||
    error ||
    ""
  );

  const match = message.match(
    /\b(429|500|502|503|504)\b/
  );

  return match ? Number(match[1]) : null;
}

function isRetryableGeminiError(error) {
  const status = getGeminiErrorStatus(error);

  return [
    429,
    500,
    502,
    503,
    504
  ].includes(status);
}

async function callGeminiWithRetry(
  operationName,
  operation
) {
  let lastError = null;

  for (
    let attempt = 0;
    attempt <= GEMINI_MAX_RETRIES;
    attempt++
  ) {
    try {
      if (attempt > 0) {
        const baseDelay =
          GEMINI_INITIAL_RETRY_DELAY *
          Math.pow(2, attempt - 1);

        const jitter =
          Math.floor(Math.random() * 1000);

        const wait = Math.min(
          baseDelay + jitter,
          30000
        );

        console.log(
          `[GEMINI RETRY] ${operationName} ${attempt}/${GEMINI_MAX_RETRIES} waiting ${wait}ms`
        );

        await sleep(wait);
      }

      console.log(
        `[GEMINI] ${operationName} attempt ${
          attempt + 1
        }/${GEMINI_MAX_RETRIES + 1}`
      );

      return await operation();

    } catch (error) {
      lastError = error;

      const status =
        getGeminiErrorStatus(error);

      console.error(
        `[GEMINI ERROR] ${operationName}`,
        status || "",
        error?.message || error
      );

      if (!isRetryableGeminiError(error)) {
        throw error;
      }

      if (attempt >= GEMINI_MAX_RETRIES) {
        break;
      }
    }
  }

  const status =
    getGeminiErrorStatus(lastError);

  throw new Error(
    `Gemini ${operationName} failed after ${
      GEMINI_MAX_RETRIES + 1
    } attempts${
      status ? ` (HTTP ${status})` : ""
    }. ${
      lastError?.message ||
      "Temporary Gemini API error."
    }`
  );
}

/* =========================================================
   VIDEO DURATION
========================================================= */

async function getVideoDuration(videoPath) {
  const result = await runCommand(
    "ffprobe",
    [
      "-v",
      "error",
      "-show_entries",
      "format=duration",
      "-of",
      "default=noprint_wrappers=1:nokey=1",
      videoPath
    ]
  );

  const duration = Number(
    result.stdout.trim()
  );

  if (!Number.isFinite(duration)) {
    throw new Error(
      "Unable to read video duration."
    );
  }

  return duration;
}

/* =========================================================
   EXTRACT AUDIO
========================================================= */

async function extractAudio(
  videoPath,
  outputPath
) {
  await runCommand(
    "ffmpeg",
    [
      "-y",
      "-i",
      videoPath,
      "-vn",
      "-map",
      "0:a:0",
      "-ac",
      "1",
      "-ar",
      "16000",
      "-b:a",
      "64k",
      "-c:a",
      "libmp3lame",
      outputPath
    ]
  );
}

/* =========================================================
   SPLIT AUDIO
========================================================= */

async function splitAudio(
  audioPath,
  outputDir
) {
  fs.mkdirSync(outputDir, {
    recursive: true
  });

  await runCommand(
    "ffmpeg",
    [
      "-y",
      "-i",
      audioPath,
      "-f",
      "segment",
      "-segment_time",
      String(AUDIO_CHUNK_SECONDS),
      "-reset_timestamps",
      "1",
      "-c",
      "copy",
      path.join(
        outputDir,
        "chunk-%04d.mp3"
      )
    ]
  );

  const files = fs
    .readdirSync(outputDir)
    .filter(
      file =>
        file.startsWith("chunk-") &&
        file.endsWith(".mp3")
    )
    .sort();

  if (!files.length) {
    throw new Error(
      "FFmpeg could not create audio chunks."
    );
  }

  return files.map(file =>
    path.join(outputDir, file)
  );
}

/* =========================================================
   GROQ WHISPER
========================================================= */

async function transcribeChunk(
  groq,
  audioPath,
  chunkIndex
) {
  const file =
    fs.createReadStream(audioPath);

  const response =
    await groq.audio.transcriptions.create({
      file,
      model: GROQ_MODEL,
      response_format: "verbose_json",
      timestamp_granularities: [
        "segment"
      ],
      temperature: 0
    });

  const offset =
    chunkIndex *
    AUDIO_CHUNK_SECONDS;

  const segments =
    Array.isArray(response.segments)
      ? response.segments
      : [];

  return {
    text: response.text || "",

    segments: segments.map(segment => ({
      start:
        Number(segment.start || 0) +
        offset,

      end:
        Number(segment.end || 0) +
        offset,

      text: String(
        segment.text || ""
      ).trim()
    }))
  };
}

async function transcribeMovie(
  jobId,
  audioChunks
) {
  const groq = new Groq({
    apiKey: requireEnv(
      "GROQ_API_KEY"
    )
  });

  const allSegments = [];
  const allTexts = [];

  for (
    let i = 0;
    i < audioChunks.length;
    i++
  ) {
    const progress =
      15 +
      Math.round(
        (i / audioChunks.length) *
        35
      );

    updateJob(jobId, {
      stage: "Whisper",
      progress,
      message:
        `Groq Whisper: ${
          i + 1
        } / ${
          audioChunks.length
        }`
    });

    const result =
      await transcribeChunk(
        groq,
        audioChunks[i],
        i
      );

    if (result.text) {
      allTexts.push(result.text);
    }

    allSegments.push(
      ...result.segments
    );
  }

  allSegments.sort(
    (a, b) =>
      a.start - b.start
  );

  return {
    text: allTexts
      .join(" ")
      .trim(),

    segments: allSegments
  };
}

/* =========================================================
   TRANSCRIPT TIMELINE
========================================================= */

function buildTranscriptTimeline(
  segments
) {
  if (!Array.isArray(segments)) {
    return "";
  }

  return segments
    .map(
      (segment, index) =>
        `SEGMENT ${index}: [${
          segment.start.toFixed(2)
        }s - ${
          segment.end.toFixed(2)
        }s] ${segment.text}`
    )
    .join("\n");
}

/* =========================================================
   FFMPEG SCENE DETECTION + GROQ RECAP PLAN
========================================================= */

function normalizeDetectedBoundaries(boundaries, duration) {
  const values = [0, ...boundaries, duration]
    .map(Number)
    .filter(Number.isFinite)
    .map(v => Math.max(0, Math.min(duration, v)))
    .sort((a, b) => a - b);

  const unique = [];
  for (const value of values) {
    if (!unique.length || value - unique[unique.length - 1] >= MIN_SCENE_SECONDS) {
      unique.push(value);
    }
  }

  if (unique[unique.length - 1] !== duration) unique.push(duration);
  if (unique[0] !== 0) unique.unshift(0);

  let scenes = [];
  for (let i = 0; i < unique.length - 1; i++) {
    const start = unique[i];
    const end = unique[i + 1];
    if (end - start >= MIN_SCENE_SECONDS) {
      scenes.push({ start, end });
    }
  }

  while (scenes.length > MAX_SCENES) {
    let index = 0;
    let smallest = Infinity;
    for (let i = 0; i < scenes.length - 1; i++) {
      const combined = scenes[i + 1].end - scenes[i].start;
      if (combined < smallest) { smallest = combined; index = i; }
    }
    scenes[index].end = scenes[index + 1].end;
    scenes.splice(index + 1, 1);
  }

  if (!scenes.length) {
    scenes = [{ start: 0, end: duration }];
  }

  scenes[0].start = 0;
  scenes[scenes.length - 1].end = duration;

  return scenes.map((scene, index) => ({
    index: index + 1,
    start: Number(scene.start.toFixed(3)),
    end: Number(scene.end.toFixed(3)),
    duration: Number((scene.end - scene.start).toFixed(3)),
    visual: `Movie scene ${index + 1}`,
    event: "",
    narration: ""
  }));
}

async function detectMovieScenes(moviePath, duration) {
  const result = await runCommand(
    "ffmpeg",
    [
      "-hide_banner",
      "-i", moviePath,
      "-vf", `select='gt(scene,${SCENE_THRESHOLD})',showinfo`,
      "-an",
      "-f", "null",
      "-"
    ]
  );

  const log = `${result.stderr || ""}\n${result.stdout || ""}`;
  const boundaries = [];
  const regex = /pts_time:([0-9.]+)/g;
  let match;
  while ((match = regex.exec(log)) !== null) {
    const time = Number(match[1]);
    if (Number.isFinite(time) && time > 0.5 && time < duration - 0.5) {
      boundaries.push(time);
    }
  }

  const scenes = normalizeDetectedBoundaries(boundaries, duration);
  console.log(`[SCENE DETECTION] ${scenes.length} scenes created from FFmpeg.`);
  return scenes;
}

async function generateGroqRecapPlan(
  jobId,
  scenes,
  transcript,
  language = "my",
  style = "cinematic"
) {
  const groq = new Groq({ apiKey: requireEnv("GROQ_API_KEY") });
  const timeline = buildTranscriptTimeline(transcript.segments);

  const languageInstruction = language === "en"
    ? "Write natural spoken English."
    : "Write natural spoken Myanmar (Burmese).";

  const styleInstruction = style === "short"
    ? "Use concise, fast-paced recap narration."
    : style === "storytelling"
      ? "Use smooth storytelling with suspense and emotional flow."
      : "Use cinematic movie recap narration.";

  const sceneList = scenes.map(scene =>
    `SCENE ${scene.index}: ${scene.start.toFixed(2)}s - ${scene.end.toFixed(2)}s (${scene.duration.toFixed(2)}s)`
  ).join("\n");

  updateJob(jobId, {
    stage: "Gemini",
    progress: 58,
    message: "Groq is creating the synchronized Myanmar recap script..."
  });

  const prompt = `
Create a professional movie recap script using ONLY the supplied timestamped transcript and scene boundaries.

IMPORTANT:
- Do not invent events, characters, dialogue, locations, or outcomes.
- Each scene timestamp is fixed. Do not change start_sec or end_sec.
- Write narration that matches the transcript content occurring inside that scene.
- If a scene has little dialogue, summarize only information supported by nearby transcript context.
- Keep the narration short enough to fit naturally inside the scene duration.
- Cover all scenes.
- Keep the ending.
- Return ONLY valid JSON.

LANGUAGE: ${languageInstruction}
STYLE: ${styleInstruction}

SCENE BOUNDARIES:
${sceneList}

WHISPER TIMELINE:
${timeline}

JSON FORMAT:
{
  "scenes": [
    {
      "index": 1,
      "visual_summary": "Brief scene description based only on available context.",
      "event_summary": "Supported story event.",
      "narration": "Spoken recap narration."
    }
  ]
}`;

  const response = await groq.chat.completions.create({
    model: GROQ_RECAP_MODEL,
    temperature: 0.2,
    messages: [
      { role: "system", content: "You are a precise movie recap writer. Never invent facts." },
      { role: "user", content: prompt }
    ],
    response_format: { type: "json_object" }
  });

  const text = response?.choices?.[0]?.message?.content || "";
  if (!text) throw new Error("Groq returned an empty recap plan.");

  let parsed;
  try { parsed = JSON.parse(text); }
  catch { throw new Error("Groq returned invalid recap JSON."); }

  const generated = Array.isArray(parsed.scenes) ? parsed.scenes : [];
  const byIndex = new Map(generated.map(scene => [Number(scene.index), scene]));

  const finalScenes = scenes.map(scene => {
    const item = byIndex.get(scene.index) || {};
    return {
      ...scene,
      visual: String(item.visual_summary || `Movie scene ${scene.index}`).trim(),
      event: String(item.event_summary || "").trim(),
      narration: String(item.narration || "").trim() || "ဆက်လက်ဖြစ်ပျက်နေသော အကြောင်းအရာကို ဇာတ်လမ်းအတိုင်း ဆက်လက်ဖော်ပြထားသည်။"
    };
  });

  return { scenes: finalScenes };
}

/* =========================================================
   TTS VOICE
========================================================= */

function resolveVoice(voice) {
  if (voice === "male") {
    return "Puck";
  }

  if (voice === "female") {
    return "Kore";
  }

  const allowed = [
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat"
  ];

  if (allowed.includes(voice)) {
    return voice;
  }

  return "Kore";
}

/* =========================================================
   TTS
========================================================= */

async function generateSceneTTS(
  ai,
  scene,
  outputPath,
  voice
) {
  const actualVoice =
    resolveVoice(voice);

  const text =
    String(
      scene.narration || ""
    ).trim();

  if (!text) {
    throw new Error(
      `Scene ${scene.index} has no narration.`
    );
  }

  const response =
    await callGeminiWithRetry(
      `Scene ${scene.index} TTS`,
      () =>
        ai.models.generateContent({
          model: GEMINI_TTS_MODEL,

          contents: [
            {
              role: "user",

              parts: [
                {
                  text,

                  speech_metadata: {
                    style:
                      "Natural cinematic movie recap narration. Clear pronunciation, smooth pacing, emotional but controlled storyteller voice."
                  }
                }
              ]
            }
          ],

          config: {
            responseModalities: [
              "AUDIO"
            ],

            speechConfig: {
              voiceConfig: {
                voice: actualVoice
              }
            }
          }
        })
    );

  const base64 =
    response
      ?.candidates?.[0]
      ?.content?.parts
      ?.find(
        part =>
          part?.inlineData?.data
      )
      ?.inlineData?.data;

  if (!base64) {
    throw new Error(
      `Gemini returned no audio for scene ${scene.index}.`
    );
  }

  fs.writeFileSync(
    outputPath,
    Buffer.from(
      base64,
      "base64"
    )
  );

  const stats =
    fs.statSync(outputPath);

  if (stats.size < 100) {
    throw new Error(
      `TTS scene ${scene.index} audio is invalid.`
    );
  }

  return outputPath;
}

/* =========================================================
   AUDIO DURATION
========================================================= */

async function getAudioDuration(
  audioPath
) {
  const result =
    await runCommand(
      "ffprobe",
      [
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        audioPath
      ]
    );

  const duration =
    Number(
      result.stdout.trim()
    );

  if (!Number.isFinite(duration)) {
    throw new Error(
      "Unable to read generated TTS duration."
    );
  }

  return duration;
}

/* =========================================================
   ATEMPO
========================================================= */

function buildAtempoFilters(
  factor
) {
  let value = factor;
  const filters = [];

  while (value > 2) {
    filters.push(
      "atempo=2.0"
    );

    value /= 2;
  }

  while (value < 0.5) {
    filters.push(
      "atempo=0.5"
    );

    value /= 0.5;
  }

  if (
    Math.abs(value - 1) >
    0.001
  ) {
    filters.push(
      `atempo=${value.toFixed(6)}`
    );
  }

  return filters;
}

/* =========================================================
   FIT TTS TO SCENE
========================================================= */

async function fitAudioToScene(
  inputAudio,
  outputAudio,
  targetDuration
) {
  const sourceDuration =
    await getAudioDuration(
      inputAudio
    );

  if (sourceDuration <= 0) {
    throw new Error(
      "Generated TTS has invalid duration."
    );
  }

  const factor =
    sourceDuration /
    targetDuration;

  const filters =
    buildAtempoFilters(
      factor
    );

  filters.push("apad");

  filters.push(
    `atrim=duration=${targetDuration.toFixed(
      3
    )}`
  );

  await runCommand(
    "ffmpeg",
    [
      "-y",
      "-i",
      inputAudio,
      "-af",
      filters.join(","),
      "-ar",
      "24000",
      "-ac",
      "1",
      "-c:a",
      "pcm_s16le",
      outputAudio
    ]
  );

  return {
    sourceDuration,
    targetDuration,
    factor
  };
}

/* =========================================================
   CUT VIDEO SCENE
========================================================= */

async function renderSceneVideo(
  moviePath,
  scene,
  outputPath
) {
  const duration =
    scene.end - scene.start;

  await runCommand(
    "ffmpeg",
    [
      "-y",

      "-ss",
      scene.start.toFixed(3),

      "-i",
      moviePath,

      "-t",
      duration.toFixed(3),

      "-an",

      "-c:v",
      "libx264",

      "-preset",
      "veryfast",

      "-crf",
      "23",

      "-pix_fmt",
      "yuv420p",

      "-movflags",
      "+faststart",

      outputPath
    ]
  );

  return outputPath;
}

/* =========================================================
   MUX SCENE
========================================================= */

async function muxScene(
  sceneVideo,
  sceneAudio,
  outputPath
) {
  await runCommand(
    "ffmpeg",
    [
      "-y",

      "-i",
      sceneVideo,

      "-i",
      sceneAudio,

      "-map",
      "0:v:0",

      "-map",
      "1:a:0",

      "-c:v",
      "copy",

      "-c:a",
      "aac",

      "-b:a",
      "128k",

      "-shortest",

      "-movflags",
      "+faststart",

      outputPath
    ]
  );

  return outputPath;
}

/* =========================================================
   CONCAT SCENES
========================================================= */

async function concatScenes(
  sceneFiles,
  outputPath,
  listPath
) {
  if (!sceneFiles.length) {
    throw new Error(
      "No rendered scenes were found."
    );
  }

  const list =
    sceneFiles
      .map(
        file =>
          `file '${file.replace(
            /'/g,
            "'\\''"
          )}'`
      )
      .join("\n");

  fs.writeFileSync(
    listPath,
    list,
    "utf8"
  );

  await runCommand(
    "ffmpeg",
    [
      "-y",

      "-f",
      "concat",

      "-safe",
      "0",

      "-i",
      listPath,

      "-c",
      "copy",

      "-movflags",
      "+faststart",

      outputPath
    ]
  );

  return outputPath;
}

/* =========================================================
   RENDER FINAL VIDEO
========================================================= */

async function renderSceneSyncVideo(
  jobId,
  moviePath,
  scenePlan,
  voice,
  jobFolder
) {
  const ai =
    new GoogleGenAI({
      apiKey:
        requireEnv(
          "GEMINI_API_KEY"
        )
    });

  const sceneDir =
    path.join(
      jobFolder,
      "scenes"
    );

  fs.mkdirSync(
    sceneDir,
    {
      recursive: true
    }
  );

  const renderedScenes = [];

  for (
    let i = 0;
    i < scenePlan.scenes.length;
    i++
  ) {
    const scene =
      scenePlan.scenes[i];

    const progress =
      65 +
      Math.round(
        (i /
          scenePlan.scenes.length) *
          25
      );

    updateJob(jobId, {
      stage: "Voice",
      progress,
      message:
        `Scene ${scene.index} / ${scenePlan.scenes.length} — generating voice...`
    });

    const number =
      String(
        scene.index
      ).padStart(3, "0");

    const rawTTS =
      path.join(
        sceneDir,
        `scene-${number}-raw.wav`
      );

    const fittedTTS =
      path.join(
        sceneDir,
        `scene-${number}-audio.wav`
      );

    const sceneVideo =
      path.join(
        sceneDir,
        `scene-${number}-video.mp4`
      );

    const sceneFinal =
      path.join(
        sceneDir,
        `scene-${number}-final.mp4`
      );

    await generateSceneTTS(
      ai,
      scene,
      rawTTS,
      voice
    );

    await fitAudioToScene(
      rawTTS,
      fittedTTS,
      scene.duration
    );

    updateJob(jobId, {
      stage: "FFmpeg",
      progress: progress + 2,
      message:
        `Scene ${scene.index} / ${scenePlan.scenes.length} — syncing video and voice...`
    });

    await renderSceneVideo(
      moviePath,
      scene,
      sceneVideo
    );

    await muxScene(
      sceneVideo,
      fittedTTS,
      sceneFinal
    );

    renderedScenes.push(
      sceneFinal
    );
  }

  updateJob(jobId, {
    stage: "FFmpeg",
    progress: 94,
    message:
      "Joining synchronized scenes..."
  });

  const listPath =
    path.join(
      jobFolder,
      "scene-list.txt"
    );

  const outputPath =
    path.join(
      jobFolder,
      "YNT-One-Clips-Recap.mp4"
    );

  await concatScenes(
    renderedScenes,
    outputPath,
    listPath
  );

  return outputPath;
}

/* =========================================================
   PROCESS JOB
========================================================= */

async function processOneClip(
  jobId,
  moviePath,
  language,
  style,
  voice
) {
  try {
    const jobFolder =
      path.dirname(moviePath);

    updateJob(jobId, {
      stage: "Upload",
      progress: 8,
      message:
        "Reading movie information..."
    });

    const duration =
      await getVideoDuration(
        moviePath
      );

    updateJob(jobId, {
      duration
    });

    console.log(
      `[JOB ${jobId}] Duration: ${duration.toFixed(2)}s`
    );

    /* -----------------------------------------
       AUDIO
    ----------------------------------------- */

    const audioPath =
      path.join(
        jobFolder,
        "movie-audio.mp3"
      );

    updateJob(jobId, {
      stage: "FFmpeg",
      progress: 10,
      message:
        "Extracting movie audio..."
    });

    await extractAudio(
      moviePath,
      audioPath
    );

    /* -----------------------------------------
       CHUNKS
    ----------------------------------------- */

    const chunksDir =
      path.join(
        jobFolder,
        "audio-chunks"
      );

    updateJob(jobId, {
      stage: "FFmpeg",
      progress: 13,
      message:
        "Preparing audio for Whisper..."
    });

    const audioChunks =
      await splitAudio(
        audioPath,
        chunksDir
      );

    updateJob(jobId, {
      totalChunks:
        audioChunks.length
    });

    /* -----------------------------------------
       WHISPER
    ----------------------------------------- */

    const transcript =
      await transcribeMovie(
        jobId,
        audioChunks
      );

    if (!transcript.text) {
      throw new Error(
        "Whisper returned an empty transcript."
      );
    }

    fs.writeFileSync(
      path.join(
        jobFolder,
        "transcript.json"
      ),
      JSON.stringify(
        transcript,
        null,
        2
      ),
      "utf8"
    );

    updateJob(jobId, {
      stage: "Transcript",
      progress: 52,
      message:
        "Timestamp transcript completed.",

      transcript: {
        characters:
          transcript.text.length,

        segments:
          transcript.segments.length
      }
    });

    /* -----------------------------------------
       SCENE DETECTION + GROQ RECAP
    ----------------------------------------- */

    updateJob(jobId, {
      stage: "Gemini",
      progress: 54,
      message: "FFmpeg is detecting movie scene boundaries..."
    });

    const detectedScenes =
      await detectMovieScenes(
        moviePath,
        duration
      );

    const scenePlan =
      await generateGroqRecapPlan(
        jobId,
        detectedScenes,
        transcript,
        language,
        style
      );

    const scenePlanPath =
      path.join(
        jobFolder,
        "scene-plan.json"
      );

    fs.writeFileSync(
      scenePlanPath,
      JSON.stringify(
        scenePlan,
        null,
        2
      ),
      "utf8"
    );

    updateJob(jobId, {
      stage: "Gemini",
      progress: 62,
      message:
        `${scenePlan.scenes.length} synchronized scenes created.`,

      scenePlan: {
        scenes:
          scenePlan.scenes.length,

        path:
          scenePlanPath
      }
    });

    /* -----------------------------------------
       RENDER
    ----------------------------------------- */

    const outputPath =
      await renderSceneSyncVideo(
        jobId,
        moviePath,
        scenePlan,
        voice,
        jobFolder
      );

    /* -----------------------------------------
       VERIFY
    ----------------------------------------- */

    if (
      !fs.existsSync(
        outputPath
      )
    ) {
      throw new Error(
        "FFmpeg did not create the final MP4."
      );
    }

    const stats =
      fs.statSync(
        outputPath
      );

    if (stats.size <= 0) {
      throw new Error(
        "Final MP4 file is empty."
      );
    }

    const filename =
      `YNT-One-Clips-${jobId}.mp4`;

    updateJob(jobId, {
      status: "completed",
      stage: "Ready",
      progress: 100,
      message:
        "Your synchronized recap video is ready.",

      output: {
        path: outputPath,
        filename,
        size: stats.size,
        url:
          `/api/download/${jobId}`
      }
    });

    console.log(
      `[JOB ${jobId}] COMPLETED`
    );

  } catch (error) {
    console.error(
      `[JOB ${jobId}] ERROR:`,
      error
    );

    updateJob(jobId, {
      status: "error",
      stage: "Error",
      progress: 0,
      message:
        error.message ||
        "Movie processing failed.",
      error:
        error.message ||
        "Movie processing failed."
    });
  }
}

/* =========================================================
   HEALTH
========================================================= */

app.get(
  "/health",
  (req, res) => {
    res.json({
      ok: true,
      name: "YNT One Clips",
      status: "online",

      models: {
        whisper: GROQ_MODEL,
        recap: GROQ_RECAP_MODEL,
        tts: GEMINI_TTS_MODEL
      },

      sceneSync: true,
      sceneFPS: SCENE_FPS
    });
  }
);

/* =========================================================
   STATUS
========================================================= */

app.get(
  "/api/status/:id",
  (req, res) => {
    const job =
      getJob(
        req.params.id
      );

    if (!job) {
      return res
        .status(404)
        .json({
          error:
            "Job not found."
        });
    }

    return res.json(job);
  }
);

/* =========================================================
   DOWNLOAD
========================================================= */

app.get(
  "/api/download/:id",
  (req, res) => {
    const job =
      getJob(
        req.params.id
      );

    if (!job) {
      return res
        .status(404)
        .json({
          error:
            "Job not found."
        });
    }

    if (
      job.status !==
        "completed" ||
      !job.output
    ) {
      return res
        .status(404)
        .json({
          error:
            "Final video is not ready yet."
        });
    }

    if (
      !fs.existsSync(
        job.output.path
      )
    ) {
      return res
        .status(404)
        .json({
          error:
            "Output video file no longer exists."
        });
    }

    return res.download(
      job.output.path,
      job.output.filename
    );
  }
);

/* =========================================================
   ONE CLIP
========================================================= */

app.post(
  "/api/one-clip",
  upload.single("movie"),

  async (req, res) => {
    try {
      if (!req.file) {
        return res
          .status(400)
          .json({
            error:
              "Movie file is required."
          });
      }

      const job =
        createJob();

      const jobFolder =
        path.join(
          JOB_DIR,
          job.id
        );

      fs.mkdirSync(
        jobFolder,
        {
          recursive: true
        }
      );

      const originalName =
        req.file.originalname ||
        "movie.mp4";

      const extension =
        path.extname(
          originalName
        ) || ".mp4";

      const moviePath =
        path.join(
          jobFolder,
          `movie${extension}`
        );

      fs.renameSync(
        req.file.path,
        moviePath
      );

      const language =
        req.body?.language ||
        "my";

      const style =
        req.body?.style ||
        "cinematic";

      const voice =
        req.body?.voice ||
        "female";

      updateJob(job.id, {
        status: "processing",
        stage: "Upload",
        progress: 5,
        message:
          "Movie uploaded. Processing started."
      });

      console.log(
        "========================================"
      );

      console.log(
        `[JOB ${job.id}] STARTED`
      );

      console.log(
        `[JOB ${job.id}] Language: ${language}`
      );

      console.log(
        `[JOB ${job.id}] Style: ${style}`
      );

      console.log(
        `[JOB ${job.id}] Voice: ${voice}`
      );

      console.log(
        `[JOB ${job.id}] Scene Sync: ENABLED`
      );

      console.log(
        "========================================"
      );

      res
        .status(202)
        .json({
          success: true,
          jobId: job.id,
          status: "processing",
          message:
            "Movie processing started.",
          statusUrl:
            `/api/status/${job.id}`,
          download:
            `/api/download/${job.id}`
        });

      setImmediate(() => {
        processOneClip(
          job.id,
          moviePath,
          language,
          style,
          voice
        ).catch(error => {
          console.error(
            `[JOB ${job.id}] UNHANDLED ERROR:`,
            error
          );

          updateJob(job.id, {
            status: "error",
            stage: "Error",
            progress: 0,
            message:
              error.message ||
              "Movie processing failed.",
            error:
              error.message ||
              "Movie processing failed."
          });
        });
      });

    } catch (error) {
      console.error(
        "ONE CLIP START ERROR:",
        error
      );

      if (
        req.file?.path &&
        fs.existsSync(
          req.file.path
        )
      ) {
        try {
          fs.unlinkSync(
            req.file.path
          );
        } catch {}
      }

      return res
        .status(500)
        .json({
          error:
            error.message ||
            "Unable to start movie processing."
        });
    }
  }
);

/* =========================================================
   FRONTEND
========================================================= */

app.use(
  express.static(
    PUBLIC_DIR
  )
);

/* =========================================================
   ERROR HANDLER
========================================================= */

app.use(
  (err, req, res, next) => {
    console.error(
      "SERVER ERROR:",
      err
    );

    if (
      err instanceof
      multer.MulterError
    ) {
      return res
        .status(400)
        .json({
          error:
            err.message
        });
    }

    return res
      .status(500)
      .json({
        error:
          err.message ||
          "Server error."
      });
  }
);

/* =========================================================
   START SERVER
========================================================= */

app.listen(
  PORT,
  "0.0.0.0",
  () => {
    console.log(
      "========================================"
    );

    console.log(
      "YNT One Clips"
    );

    console.log(
      `Server running on 0.0.0.0:${PORT}`
    );

    console.log(
      `Groq Whisper: ${GROQ_MODEL}`
    );

    console.log(
      `Groq Recap: ${GROQ_RECAP_MODEL}`
    );

    console.log(
      `Gemini TTS: ${GEMINI_TTS_MODEL}`
    );

    console.log(
      `FFmpeg Scene Detection Threshold: ${SCENE_THRESHOLD}`
    );

    console.log(
      "Scene Synchronization: READY"
    );

    console.log(
      "Gemini Retry System: READY"
    );

    console.log(
      "Background Jobs: READY"
    );

    console.log(
      "FFmpeg Scene Renderer: READY"
    );

    console.log(
      "========================================"
    );
  }
);
