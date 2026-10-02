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
import pg from "pg";

const { Pool } = pg;

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

// DATA_DIR can point to a persistent disk on the hosting platform.
// If it is not set, the project directory is used.
const DATA_DIR = process.env.DATA_DIR
  ? path.resolve(process.env.DATA_DIR)
  : ROOT;

const PUBLIC_DIR = path.join(ROOT, "public");
const UPLOAD_DIR = path.join(DATA_DIR, "uploads");
const JOB_DIR = path.join(DATA_DIR, "jobs");

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
const GEMINI_MODEL = "gemini-3.8-flash";
const GEMINI_TTS_MODEL = "gemini-3.8-flash-tts";

/* =========================================================
   SETTINGS
========================================================= */

const AUDIO_CHUNK_SECONDS = 90;
const MAX_EVENTS = 16;
const MIN_EVENT_SECONDS = 2.5;
const MAX_VISUAL_KEYFRAMES = 24;
const VISION_IMAGES_PER_REQUEST = 3;
const GROQ_VISION_MODEL = "qwen/qwen3.8-27b";
const MAX_TTS_CHARS_PER_REQUEST = 6000;
const GROQ_RECAP_MODEL = "openai/gpt-oss-120b";
const MAX_VIDEO_SIZE = 500 * 1024 * 1024;

const GEMINI_MAX_RETRIES = 0;
const GEMINI_INITIAL_RETRY_DELAY = 2000;

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
   JOB STORAGE — POSTGRESQL
========================================================= */

const DATABASE_URL = process.env.DATABASE_URL;

if (!DATABASE_URL) {
  throw new Error(
    "DATABASE_URL is required. Add a PostgreSQL connection string to the environment."
  );
}

const db = new Pool({
  connectionString: DATABASE_URL,
  ssl:
    process.env.DATABASE_SSL === "false"
      ? false
      : { rejectUnauthorized: false },
  max: 5,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 10000
});

const jobs = new Map();
const jobWriteQueues = new Map();

async function initDatabase() {
  await db.query(`
    CREATE TABLE IF NOT EXISTS jobs (
      id TEXT PRIMARY KEY,
      status TEXT NOT NULL,
      stage TEXT,
      progress INTEGER NOT NULL DEFAULT 0,
      message TEXT,
      created_at TIMESTAMPTZ NOT NULL,
      updated_at TIMESTAMPTZ NOT NULL,
      original_filename TEXT,
      duration DOUBLE PRECISION,
      total_chunks INTEGER,
      transcript JSONB,
      scene_plan JSONB,
      output JSONB,
      error TEXT
    )
  `);

  await db.query(`
    CREATE INDEX IF NOT EXISTS idx_jobs_updated_at
    ON jobs (updated_at DESC)
  `);

  console.log(
    "[JOB STORAGE] PostgreSQL database ready."
  );
}

function jobToDbRow(job) {
  return {
    id: job.id,
    status: job.status || "created",
    stage: job.stage || "Waiting",
    progress: Number.isFinite(Number(job.progress))
      ? Number(job.progress)
      : 0,
    message: job.message || null,
    created_at: job.createdAt || new Date().toISOString(),
    updated_at: job.updatedAt || new Date().toISOString(),
    original_filename: job.originalFilename || null,
    duration:
      job.duration === null || job.duration === undefined
        ? null
        : Number(job.duration),
    total_chunks:
      job.totalChunks === null || job.totalChunks === undefined
        ? null
        : Number(job.totalChunks),
    transcript: job.transcript ?? null,
    scene_plan: job.scenePlan ?? null,
    output: job.output ?? null,
    error: job.error || null
  };
}

function dbRowToJob(row) {
  return {
    id: row.id,
    status: row.status,
    stage: row.stage,
    progress: row.progress,
    message: row.message,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    originalFilename: row.original_filename,
    duration: row.duration,
    totalChunks: row.total_chunks,
    transcript: row.transcript,
    scenePlan: row.scene_plan,
    output: row.output,
    error: row.error
  };
}

async function persistJob(job) {
  const row = jobToDbRow(job);

  await db.query(
    `
      INSERT INTO jobs (
        id, status, stage, progress, message,
        created_at, updated_at, original_filename,
        duration, total_chunks, transcript, scene_plan,
        output, error
      )
      VALUES (
        $1, $2, $3, $4, $5,
        $6, $7, $8,
        $9, $10, $11::jsonb, $12::jsonb,
        $13::jsonb, $14
      )
      ON CONFLICT (id) DO UPDATE SET
        status = EXCLUDED.status,
        stage = EXCLUDED.stage,
        progress = EXCLUDED.progress,
        message = EXCLUDED.message,
        updated_at = EXCLUDED.updated_at,
        original_filename = EXCLUDED.original_filename,
        duration = EXCLUDED.duration,
        total_chunks = EXCLUDED.total_chunks,
        transcript = EXCLUDED.transcript,
        scene_plan = EXCLUDED.scene_plan,
        output = EXCLUDED.output,
        error = EXCLUDED.error
    `,
    [
      row.id,
      row.status,
      row.stage,
      row.progress,
      row.message,
      row.created_at,
      row.updated_at,
      row.original_filename,
      row.duration,
      row.total_chunks,
      JSON.stringify(row.transcript),
      JSON.stringify(row.scene_plan),
      JSON.stringify(row.output),
      row.error
    ]
  );
}

function queueJobWrite(job) {
  const previous =
    jobWriteQueues.get(job.id) ||
    Promise.resolve();

  const next = previous
    .catch(() => {})
    .then(() => persistJob(job))
    .catch(error => {
      console.error(
        `[JOB STORAGE] Failed to persist ${job.id}:`,
        error?.message || error
      );
    });

  jobWriteQueues.set(job.id, next);

  next.finally(() => {
    if (jobWriteQueues.get(job.id) === next) {
      jobWriteQueues.delete(job.id);
    }
  }).catch(() => {});

  return next;
}

async function createJob() {
  const id = crypto.randomUUID();
  const now = new Date().toISOString();

  const job = {
    id,
    status: "created",
    stage: "Waiting",
    progress: 0,
    message: "Job created.",
    createdAt: now,
    updatedAt: now,
    originalFilename: null,
    duration: null,
    totalChunks: null,
    transcript: null,
    scenePlan: null,
    output: null,
    error: null
  };

  jobs.set(id, job);
  await persistJob(job);

  return job;
}

function updateJob(id, data) {
  const job = jobs.get(id);

  if (!job) {
    return null;
  }

  Object.assign(job, data, {
    updatedAt: new Date().toISOString()
  });

  queueJobWrite(job);

  return job;
}

async function getJob(id) {
  const cached = jobs.get(id);

  if (cached) {
    return cached;
  }

  const result = await db.query(
    `SELECT * FROM jobs WHERE id = $1 LIMIT 1`,
    [id]
  );

  if (!result.rows.length) {
    return null;
  }

  const job = dbRowToJob(result.rows[0]);
  jobs.set(job.id, job);

  return job;
}

async function waitForJobWrites(jobId) {
  const pending = jobWriteQueues.get(jobId);

  if (pending) {
    await pending;
  }
}

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
   GEMINI VIDEO UPLOAD
========================================================= */

async function uploadVideoToGemini(
  ai,
  videoPath
) {
  const extension =
    path.extname(videoPath)
      .toLowerCase();

  let mimeType = "video/mp4";

  if (extension === ".webm") {
    mimeType = "video/webm";
  } else if (extension === ".mov") {
    mimeType = "video/mov";
  } else if (extension === ".avi") {
    mimeType = "video/avi";
  } else if (extension === ".mkv") {
    mimeType = "video/x-matroska";
  }

  console.log(
    `[GEMINI VIDEO] Uploading ${videoPath}`
  );

  const file =
    await ai.files.upload({
      file: videoPath,
      config: {
        mimeType
      }
    });

  let current = file;

  while (
    current?.state === "PROCESSING"
  ) {
    console.log(
      "[GEMINI VIDEO] Processing..."
    );

    await sleep(3000);

    current =
      await ai.files.get({
        name: file.name
      });
  }

  if (
    current?.state === "FAILED"
  ) {
    throw new Error(
      "Gemini failed to process the movie video."
    );
  }

  if (
    current?.state !== "ACTIVE"
  ) {
    throw new Error(
      `Gemini video is not ready. State: ${
        current?.state || "unknown"
      }`
    );
  }

  console.log(
    "[GEMINI VIDEO] ACTIVE"
  );

  return current;
}

/* =========================================================
   STORY EVENT MAP
   Event-first sync: the AI identifies meaningful story events and
   anchors each event to unique source footage. We do NOT slice the
   movie into arbitrary fixed-length timeline scenes.
========================================================= */

function parseJsonObject(text, fallback = null) {
  const raw = String(text || "").trim();
  try { return JSON.parse(raw); } catch {}
  const first = raw.indexOf("{");
  const last = raw.lastIndexOf("}");
  if (first >= 0 && last > first) {
    try { return JSON.parse(raw.slice(first, last + 1)); } catch {}
  }
  return fallback;
}

async function detectVisualKeyframes(videoPath, duration, jobFolder) {
  const frameDir = path.join(jobFolder, "story-frames");
  fs.mkdirSync(frameDir, { recursive: true });

  // Local FFmpeg shot-change detection. This is NOT a fixed timeline split.
  // It only finds candidate moments where the visual content changes.
  const result = await runCommand("ffmpeg", [
    "-hide_banner",
    "-i", videoPath,
    "-vf", "select='gt(scene,0.30)',showinfo",
    "-an",
    "-f", "null",
    "-"
  ]);

  const rawTimes = [];
  const combined = `${result.stdout || ""}\n${result.stderr || ""}`;
  for (const match of combined.matchAll(/pts_time:([0-9.]+)/g)) {
    const t = Number(match[1]);
    if (Number.isFinite(t) && t >= 0 && t <= duration) rawTimes.push(t);
  }

  // Always include the opening and ending context, then fill gaps only when
  // the movie has too few detected visual changes.
  const unique = [];
  for (const t of [0, ...rawTimes, Math.max(0, duration - 0.25), duration]) {
    if (!unique.some(x => Math.abs(x - t) < 1.0)) unique.push(t);
  }
  unique.sort((a,b) => a-b);

  if (unique.length < 8 && duration > 20) {
    const target = Math.min(MAX_VISUAL_KEYFRAMES, Math.max(8, Math.ceil(duration / 10)));
    for (let i = 0; i < target; i++) {
      const t = duration * i / Math.max(1, target - 1);
      if (!unique.some(x => Math.abs(x - t) < 2.0)) unique.push(t);
    }
    unique.sort((a,b) => a-b);
  }

  let selected = unique;
  if (selected.length > MAX_VISUAL_KEYFRAMES) {
    const picked = [];
    for (let i = 0; i < MAX_VISUAL_KEYFRAMES; i++) {
      const idx = Math.round(i * (selected.length - 1) / (MAX_VISUAL_KEYFRAMES - 1));
      picked.push(selected[idx]);
    }
    selected = [...new Set(picked)].sort((a,b) => a-b);
  }

  const frames = [];
  for (let i = 0; i < selected.length; i++) {
    const time = selected[i];
    const file = path.join(frameDir, `frame-${String(i+1).padStart(3,"0")}.jpg`);
    await runCommand("ffmpeg", [
      "-y", "-ss", time.toFixed(3), "-i", videoPath,
      "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "7", file
    ]);
    if (fs.existsSync(file)) frames.push({ index:i+1, time, path:file });
  }
  return frames;
}

function imageDataUrl(filePath) {
  const base64 = fs.readFileSync(filePath).toString("base64");
  return `data:image/jpeg;base64,${base64}`;
}

async function analyzeStoryKeyframes(jobId, videoPath, duration, jobFolder) {
  const groq = new Groq({ apiKey: requireEnv("GROQ_API_KEY") });
  const frames = await detectVisualKeyframes(videoPath, duration, jobFolder);
  const observations = [];

  for (let start = 0; start < frames.length; start += VISION_IMAGES_PER_REQUEST) {
    const group = frames.slice(start, start + VISION_IMAGES_PER_REQUEST);
    const content = [{
      type: "text",
      text: `You are making a movie recap event map. Inspect these chronological keyframes. For each frame, describe only what is visibly happening: people, actions, objects, location, important visual change, and any readable text. Do not guess hidden motives or dialogue. Return JSON only as {"frames":[{"index":1,"time_sec":0,"description":"..."}]}. Keep each description concise.`
    }];
    for (const frame of group) {
      content.push({ type:"text", text:`FRAME ${frame.index} — ${frame.time.toFixed(2)} seconds` });
      content.push({ type:"image_url", image_url:{ url:imageDataUrl(frame.path) } });
    }

    updateJob(jobId, {
      stage: "Story Map",
      progress: 55 + Math.round(((start + group.length) / Math.max(1, frames.length)) * 8),
      message: `Reading important visual moments ${Math.min(start + group.length, frames.length)}/${frames.length}`
    });

    const response = await groq.chat.completions.create({
      model: GROQ_VISION_MODEL,
      messages: [{ role:"user", content }],
      temperature: 0.1,
      max_completion_tokens: 500,
      reasoning_effort: "none",
      response_format: { type:"json_object" }
    });
    const parsed = parseJsonObject(response?.choices?.[0]?.message?.content, {frames:[]});
    if (Array.isArray(parsed?.frames)) observations.push(...parsed.frames);
  }

  observations.sort((a,b) => Number(a.time_sec||0)-Number(b.time_sec||0));
  return { frames, observations };
}

function buildVisualEvidence(observations) {
  return (observations || []).map((item, i) =>
    `VISUAL ${i+1}: [${Number(item.time_sec||0).toFixed(2)}s] ${String(item.description||"").trim()}`
  ).join("\n");
}

function intervalOverlap(a, b) {
  const start = Math.max(a.start, b.start);
  const end = Math.min(a.end, b.end);
  if (end <= start) return 0;
  return (end-start) / Math.max(0.001, Math.min(a.end-a.start, b.end-b.start));
}

function normalizeEventPlan(rawEvents, duration) {
  if (!Array.isArray(rawEvents)) throw new Error("Story Map did not return an event list.");
  const candidates = rawEvents.map(e => ({
    start: Math.max(0, Math.min(duration, Number(e.source_start_sec))),
    end: Math.max(0, Math.min(duration, Number(e.source_end_sec))),
    visual: String(e.visual_match || "").trim(),
    event: String(e.story_event || "").trim(),
    narration: String(e.narration || "").trim()
  })).filter(e => Number.isFinite(e.start) && Number.isFinite(e.end) && e.end > e.start && e.narration);

  candidates.sort((a,b) => a.start-b.start);
  const kept = [];
  for (const e of candidates) {
    if (e.end-e.start < MIN_EVENT_SECONDS) e.end = Math.min(duration, e.start + MIN_EVENT_SECONDS);
    if (kept.some(k => intervalOverlap(k,e) > 0.55)) continue;
    kept.push(e);
    if (kept.length >= MAX_EVENTS) break;
  }
  if (!kept.length) throw new Error("No usable story events were returned.");

  // Make source footage strictly non-overlapping. Gaps are allowed.
  const normalized = [];
  for (const e of kept) {
    if (normalized.length) {
      const prev = normalized[normalized.length-1];
      if (e.start < prev.end) e.start = prev.end;
    }
    if (e.end > e.start) normalized.push(e);
  }
  return normalized.map((e,i) => ({
    index:i+1,
    start:Number(e.start.toFixed(3)),
    end:Number(e.end.toFixed(3)),
    duration:Number((e.end-e.start).toFixed(3)),
    visual:e.visual,
    event:e.event,
    narration:e.narration
  }));
}

async function generateStoryEventMap(jobId, moviePath, duration, transcript, language="my", style="natural", jobFolder) {
  const groq = new Groq({ apiKey: requireEnv("GROQ_API_KEY") });
  const transcriptText = buildTranscriptTimeline(transcript.segments);
  const visual = await analyzeStoryKeyframes(jobId, moviePath, duration, jobFolder);
  const visualEvidence = buildVisualEvidence(visual.observations);

  let styleInstruction = "Natural spoken Myanmar movie recap: conversational, smooth, like a skilled Myanmar person telling a friend the movie story.";
  if (style === "suspense") styleInstruction = "Natural spoken Myanmar with controlled suspense, but never invent facts.";
  if (style === "emotional") styleInstruction = "Natural spoken Myanmar with gentle emotional emphasis only where supported by the story.";
  if (style === "fast") styleInstruction = "Concise, energetic natural spoken Myanmar while preserving important story beats.";
  if (style === "detailed") styleInstruction = "Detailed but conversational Myanmar, explaining important cause-and-effect without sounding like an essay.";

  updateJob(jobId, { stage:"Story Map", progress:65, message:"Connecting story events to their actual movie footage..." });

  const prompt = `
Create an EVENT-FIRST movie recap plan. Do NOT divide the movie into arbitrary fixed-length timeline scenes.

MOVIE DURATION: ${duration.toFixed(2)} seconds
STYLE: ${styleInstruction}
LANGUAGE: ${language === "en" ? "natural spoken English" : "natural spoken Myanmar Burmese"}

GOAL:
The final video must feel like one continuous movie recap. Every narration event must point to the actual source footage where that event happens. Events must stay in chronological story order and must not reuse the same footage.

RULES:
- Think in meaningful STORY EVENTS, not 5/10/15-second timeline blocks.
- Use both Whisper speech evidence and visual evidence.
- A source range must cover the actual visual event being narrated.
- Never assign a random nearby clip just because its timestamp is convenient.
- Never repeat or reuse the same source footage for two different events.
- Keep source ranges non-overlapping. Small gaps are fine.
- Do not narrate something that is unsupported by transcript or visible evidence.
- Do not translate dialogue line-by-line. Retell the plot naturally.
- Sound like a real Myanmar movie-recap narrator, not a subtitle translator.
- Avoid mechanical repetition of “ပြီးတော့”, “အဲဒီနောက်”, “သူက”.
- Use natural conversational Burmese particles and transitions where they fit.
- Keep the actual chronological order of the movie unless the movie itself clearly contains a flashback.
- Include setup, important discoveries, conflicts, turning points, climax and ending when present.
- Skip filler and repeated shots.
- Each narration should normally be 1–3 connected spoken sentences.
- Return 8–${MAX_EVENTS} meaningful events when the movie contains enough story material. Fewer is better than fake events.
- source_start_sec and source_end_sec MUST be actual movie timestamps.
- Return JSON only.

JSON shape:
{"events":[{"source_start_sec":12.4,"source_end_sec":19.8,"visual_match":"what is visibly happening","story_event":"why this moment matters in the story","narration":"natural spoken recap narration"}]}

WHISPER EVIDENCE:
${transcriptText}

VISUAL KEYFRAME EVIDENCE:
${visualEvidence}
`.trim();

  const response = await groq.chat.completions.create({
    model: GROQ_RECAP_MODEL,
    messages: [{role:"user", content:prompt}],
    temperature:0.28,
    max_completion_tokens:6000,
    response_format:{type:"json_object"}
  });

  const parsed = parseJsonObject(response?.choices?.[0]?.message?.content, null);
  const events = normalizeEventPlan(parsed?.events, duration);
  return { events, visualFrames: visual.frames.length, visualObservations: visual.observations.length };
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

async function generateGeminiTTSBatch(ai, text, outputPath, voice, requestLabel) {
  const actualVoice = resolveVoice(voice);
  const response = await callGeminiWithRetry(
    requestLabel,
    () => ai.models.generateContent({
      model: GEMINI_TTS_MODEL,
      contents: [{
        role: "user",
        parts: [{
          text,
          speech_metadata: {
            style: "Natural Myanmar spoken storytelling for a movie recap. Sound like a real Myanmar narrator talking naturally to viewers. Use smooth connected phrasing, normal pauses, clear pronunciation, and conversational rhythm. Do not speak word-by-word, do not rush, do not sound like a text-to-speech reading, and do not read labels, headings, JSON, or scene numbers."
          }
        }]
      }],
      config: {
        responseModalities: ["AUDIO"],
        responseFormat: {
          audio: {
            mimeType: "AUDIO_WAV",
            sampleRate: 24000
          }
        },
        speechConfig: {
          voiceConfig: { voice: actualVoice }
        }
      }
    })
  );

  const part = response?.candidates?.[0]?.content?.parts?.find(p => p?.inlineData?.data);
  if (!part?.inlineData?.data) throw new Error(`Gemini TTS returned no audio for ${requestLabel}.`);

  fs.writeFileSync(outputPath, Buffer.from(part.inlineData.data, "base64"));
  if (fs.statSync(outputPath).size < 100) throw new Error(`Gemini TTS audio is invalid for ${requestLabel}.`);
  return outputPath;
}

function normalizeSpeechText(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[\p{P}\p{S}\s]+/gu, "")
    .trim();
}

async function getAudioDuration(audioPath) {
  const result = await runCommand("ffprobe", [
    "-v", "error",
    "-show_entries", "format=duration",
    "-of", "default=noprint_wrappers=1:nokey=1",
    audioPath
  ]);
  const duration = Number(result.stdout.trim());
  if (!Number.isFinite(duration) || duration <= 0) {
    throw new Error("Generated TTS audio duration မဖတ်နိုင်ပါ။");
  }
  return duration;
}

async function transcribeGeneratedAudio(groq, audioPath) {
  const response = await groq.audio.transcriptions.create({
    file: fs.createReadStream(audioPath),
    model: GROQ_MODEL,
    response_format: "verbose_json",
    timestamp_granularities: ["segment"],
    temperature: 0
  });

  return Array.isArray(response?.segments)
    ? response.segments.map(s => ({
        start: Number(s.start || 0),
        end: Number(s.end || 0),
        text: String(s.text || "").trim()
      })).filter(s => s.text && s.end > s.start)
    : [];
}

function alignNarrationToAudio(scenes, whisperSegments, audioDuration) {
  if (!scenes.length) return [];
  if (!whisperSegments.length) {
    const totalChars = scenes.reduce((n, s) => n + normalizeSpeechText(s.narration).length, 0) || 1;
    let cursor = 0;
    return scenes.map(scene => {
      const len = normalizeSpeechText(scene.narration).length;
      const start = cursor;
      cursor += audioDuration * (len / totalChars);
      return { ...scene, audioStart: start, audioEnd: cursor };
    });
  }

  const result = [];
  let segIndex = 0;
  let cursor = 0;

  for (let i = 0; i < scenes.length; i++) {
    const target = normalizeSpeechText(scenes[i].narration);
    if (!target) {
      result.push({ ...scenes[i], audioStart: cursor, audioEnd: cursor });
      continue;
    }

    const startTime = i === 0 ? 0 : cursor;
    let accumulated = "";
    let endTime = startTime;
    let bestEndTime = startTime;
    let bestScore = 0;

    for (let j = segIndex; j < whisperSegments.length; j++) {
      accumulated += normalizeSpeechText(whisperSegments[j].text);
      endTime = Math.max(endTime, whisperSegments[j].end);
      const prefix = accumulated.slice(0, target.length);
      const common = target.length ? Math.min(prefix.length, target.length) / target.length : 1;
      if (common > bestScore) {
        bestScore = common;
        bestEndTime = endTime;
      }
      if (accumulated.length >= target.length * 0.92) {
        segIndex = j + 1;
        bestEndTime = endTime;
        break;
      }
      if (j >= segIndex + 7) break;
    }

    if (bestEndTime <= startTime) {
      const remainingChars = scenes.slice(i).reduce((n, s) => n + normalizeSpeechText(s.narration).length, 0) || 1;
      bestEndTime = Math.min(audioDuration, startTime + audioDuration * (target.length / remainingChars));
    }

    cursor = Math.min(audioDuration, Math.max(startTime + 0.05, bestEndTime));
    result.push({ ...scenes[i], audioStart: startTime, audioEnd: cursor });
  }

  if (result.length) result[result.length - 1].audioEnd = audioDuration;
  return result;
}

async function renderEventFromAudio(moviePath, event, audioPath, outputPath) {
  const narrationDuration = Math.max(0.5, event.audioEnd - event.audioStart);
  const sourceDuration = Math.max(0.5, event.end - event.start);
  const takeDuration = Math.min(sourceDuration, narrationDuration);

  // If narration is longer than the source event, freeze the last source frame
  // instead of cutting the narration. This prevents the old audio cutoff bug.
  const extra = Math.max(0, narrationDuration - takeDuration);
  const vf = extra > 0
    ? `fps=24,format=yuv420p,tpad=stop_mode=clone:stop_duration=${extra.toFixed(3)}`
    : "fps=24,format=yuv420p";

  await runCommand("ffmpeg", [
    "-y", "-ss", event.start.toFixed(3), "-i", moviePath,
    "-t", takeDuration.toFixed(3), "-an",
    "-vf", vf,
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart", outputPath
  ]);

  const sceneAudio = path.join(path.dirname(outputPath), `${path.basename(outputPath,".mp4")}-audio.wav`);
  await runCommand("ffmpeg", [
    "-y", "-ss", event.audioStart.toFixed(3), "-i", audioPath,
    "-t", narrationDuration.toFixed(3), "-ar", "24000", "-ac", "1",
    "-c:a", "pcm_s16le", sceneAudio
  ]);

  await runCommand("ffmpeg", [
    "-y", "-i", outputPath, "-i", sceneAudio,
    "-map", "0:v:0", "-map", "1:a:0",
    "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
    "-t", narrationDuration.toFixed(3),
    "-movflags", "+faststart", `${outputPath}.mux.mp4`
  ]);

  fs.renameSync(`${outputPath}.mux.mp4`, outputPath);
  try { fs.unlinkSync(sceneAudio); } catch {}
  return outputPath;
}

async function concatScenes(sceneFiles, outputPath, listPath) {
  const list = sceneFiles.map(file => `file '${file.replace(/'/g, "'\\''")}'`).join("\n");
  fs.writeFileSync(listPath, `${list}\n`, "utf8");
  await runCommand("ffmpeg", ["-y", "-f", "concat", "-safe", "0", "-i", listPath, "-c", "copy", "-movflags", "+faststart", outputPath]);
  return outputPath;
}

function groupEventsForTTS(events) {
  const groups = [];
  let current = [];
  let chars = 0;
  for (const event of events) {
    const n = normalizeSpeechText(event.narration).length;
    if (current.length && chars + n > MAX_TTS_CHARS_PER_REQUEST) {
      groups.push(current); current=[]; chars=0;
    }
    current.push(event); chars += n;
  }
  if (current.length) groups.push(current);
  return groups;
}

async function renderEventSyncVideo(jobId, moviePath, eventPlan, voice, jobFolder) {
  const ai = new GoogleGenAI({ apiKey: requireEnv("GEMINI_API_KEY") });
  const groq = new Groq({ apiKey: requireEnv("GROQ_API_KEY") });
  const eventDir = path.join(jobFolder, "events");
  fs.mkdirSync(eventDir, {recursive:true});
  const groups = groupEventsForTTS(eventPlan.events);
  const rendered = [];
  const alignedEvents = [];

  for (let g=0; g<groups.length; g++) {
    const group=groups[g];
    const text=group.map(e=>e.narration.trim()).join("\n\n");
    const rawAudio=path.join(eventDir,`tts-group-${String(g+1).padStart(2,"0")}.wav`);
    updateJob(jobId,{stage:"Voice",progress:70+Math.round((g/Math.max(1,groups.length))*15),message:`Gemini TTS — narration ${g+1}/${groups.length}`});
    await generateGeminiTTSBatch(ai,text,rawAudio,voice,`TTS group ${g+1}`);
    const audioDuration=await getAudioDuration(rawAudio);
    const whisperSegments=await transcribeGeneratedAudio(groq,rawAudio);
    const aligned=alignNarrationToAudio(group,whisperSegments,audioDuration);
    alignedEvents.push(...aligned);
    for (const event of aligned) {
      const file=path.join(eventDir,`event-${String(event.index).padStart(3,"0")}.mp4`);
      await renderEventFromAudio(moviePath,event,rawAudio,file);
      rendered.push(file);
    }
  }

  const plan={
    events: alignedEvents.map(e=>({...e,audioStart:Number(e.audioStart.toFixed(3)),audioEnd:Number(e.audioEnd.toFixed(3))}))
  };
  fs.writeFileSync(path.join(jobFolder,"story-event-plan-aligned.json"),JSON.stringify(plan,null,2),"utf8");

  updateJob(jobId,{stage:"FFmpeg",progress:94,message:"Joining unique story-event footage with narration..."});
  const outputPath=path.join(jobFolder,"YNT-One-Clips-Recap.mp4");
  await concatScenes(rendered,outputPath,path.join(jobFolder,"event-list.txt"));
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
      stage: "Upload",
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
      stage: "Whisper",
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
       STORY EVENT MAP
    ----------------------------------------- */

    const scenePlan =
      await generateStoryEventMap(
        jobId,
        moviePath,
        duration,
        transcript,
        language,
        style,
        jobFolder
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
      stage: "Story Map",
      progress: 68,
      message:
        `${scenePlan.events.length} story events matched to unique footage.`,

      scenePlan: {
        events:
          scenePlan.events.length,

        path:
          scenePlanPath
      }
    });

    /* -----------------------------------------
       RENDER
    ----------------------------------------- */

    const outputPath =
      await renderEventSyncVideo(
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
        tts: GEMINI_TTS_MODEL,
        vision: GROQ_VISION_MODEL
      },

      sceneSync: "Event-first visual + Whisper sync",
      sceneFPS: 0,
      visualVision: true,
      jobStorage: "PostgreSQL",
      dataDir: DATA_DIR
    });
  }
);

/* =========================================================
   STATUS
========================================================= */

app.get(
  "/api/status/:id",
  async (req, res) => {
    try {
      const job = await getJob(req.params.id);

      if (!job) {
        return res
          .status(404)
          .json({
            error: "Job not found."
          });
      }

      return res.json(job);
    } catch (error) {
      console.error(
        "STATUS ERROR:",
        error
      );

      return res
        .status(500)
        .json({
          error: "Unable to read job status."
        });
    }
  }
);

/* =========================================================
   DOWNLOAD
========================================================= */

app.get(
  "/api/download/:id",
  async (req, res) => {
    try {
      const job = await getJob(req.params.id);
      await waitForJobWrites(req.params.id);

      if (!job) {
        return res
          .status(404)
          .json({
            error: "Job not found."
          });
      }

      if (
        job.status !== "completed" ||
        !job.output
      ) {
        return res
          .status(404)
          .json({
            error: "Final video is not ready yet."
          });
      }

      if (
        !fs.existsSync(job.output.path)
      ) {
        return res
          .status(404)
          .json({
            error: "Output video file no longer exists."
          });
      }

      return res.download(
        job.output.path,
        job.output.filename
      );
    } catch (error) {
      console.error(
        "DOWNLOAD ERROR:",
        error
      );

      return res
        .status(500)
        .json({
          error: "Unable to download the video."
        });
    }
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
        await createJob();

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
        originalFilename: originalName,
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
        `[JOB ${job.id}] Event-first Visual Sync: ENABLED`
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

async function startServer() {
  try {
    await initDatabase();

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
          `Visual Sync: Whisper timestamp timeline`
        );

        console.log(
          `Maximum Scenes: ${MAX_SCENES}`
        );

        console.log(
          `Gemini TTS batching: ${MAX_TTS_CHARS_PER_REQUEST} chars/group`
        );

        console.log(
          `Vision Frame Analysis: DISABLED (Whisper timeline sync)`
        );

        console.log(
          `Gemini Retries: ${GEMINI_MAX_RETRIES}`
        );

        console.log(
          "Scene Synchronization: READY"
        );

        console.log(
          "Background Jobs: READY"
        );

        console.log(
          "FFmpeg Scene Renderer: READY"
        );

        console.log(
          "PostgreSQL Job Storage: READY"
        );

        console.log(
          `Persistent Data Directory: ${DATA_DIR}`
        );

        console.log(
          "========================================"
        );
      }
    );
  } catch (error) {
    console.error(
      "[STARTUP ERROR]",
      error?.message || error
    );

    await db.end().catch(() => {});
    process.exit(1);
  }
}

startServer();
