import express from "express";
import multer from "multer";
import dotenv from "dotenv";
import fs from "fs";
import os from "os";
import path from "path";
import crypto from "crypto";
import { execFile } from "child_process";
import { promisify } from "util";
import Groq from "groq-sdk";
import { GoogleGenAI } from "@google/genai";
import pg from "pg";
import {
  normalizeBurn,
  burnActive,
  probeVideoSize,
  buildBurnFilter,
  checkSubtitleFont,
  splitNarration
} from "./burn.js";
import {
  planSceneTiming,
  appendVideoTiming,
  TRIM_SILENCE_FILTER,
  makeSceneCues,
  buildSrt
} from "./sync.js";
const { Pool } = pg;

dotenv.config();

const execFileAsync = promisify(execFile);

const app = express();

// CORS: allow the separately hosted frontend to call this API.
// Set CORS_ORIGIN to a comma-separated list of trusted frontend origins.
// Defaults to "*" for compatibility with existing public frontend deployments.
const CORS_ORIGINS = String(process.env.CORS_ORIGIN || "*")
  .split(",")
  .map(origin => origin.trim())
  .filter(Boolean);

app.use((req, res, next) => {
  const requestOrigin = req.headers.origin;
  const allowOrigin = CORS_ORIGINS.includes("*")
    ? "*"
    : (requestOrigin && CORS_ORIGINS.includes(requestOrigin) ? requestOrigin : "");

  if (allowOrigin) {
    res.setHeader("Access-Control-Allow-Origin", allowOrigin);
    if (allowOrigin !== "*") res.setHeader("Vary", "Origin");
  }
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With");
  res.setHeader("Access-Control-Max-Age", "86400");

  if (req.method === "OPTIONS") {
    return res.status(204).end();
  }
  next();
});

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
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-3.8-flash";
const GEMINI_FALLBACK_MODELS = (process.env.GEMINI_FALLBACK_MODELS || "gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite")
  .split(",").map(name => name.trim()).filter(Boolean);
const GEMINI_TTS_MODEL = "gemini-3.8-flash-lite-tts";

/* =========================================================
   SETTINGS
========================================================= */

const AUDIO_CHUNK_SECONDS = 90;
// Video tokens scale linearly with fps. 1 fps is Gemini's default and halves scene-analysis token use.
const SCENE_FPS = Number(process.env.SCENE_FPS || 1);
const MAX_SCENES = 18;
// "sync" style (faithful dialogue + events) needs many more, shorter scenes.
const SYNC_MAX_SCENES = Number(process.env.SYNC_MAX_SCENES || 40);
const MIN_SCENE_SECONDS = 3;
const MAX_VIDEO_SIZE = 500 * 1024 * 1024;

// Only transient 5xx errors are retried. 429 (quota) is never retried on the same model.
const GEMINI_MAX_RETRIES = Number(process.env.GEMINI_MAX_RETRIES || 2);
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

// Lightweight API request logging helps identify hosting/network failures
// without logging uploaded content, API keys, or request bodies.
app.use((req, res, next) => {
  if (req.path.startsWith("/api/") || req.path === "/health") {
    const startedAt = Date.now();
    res.on("finish", () => {
      console.log(`[HTTP] ${req.method} ${req.originalUrl} -> ${res.statusCode} (${Date.now() - startedAt}ms)`);
    });
  }
  next();
});

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

  // 429 is intentionally NOT retried: repeating a quota-limited call only burns more quota.
  return [
    500,
    502,
    503,
    504
  ].includes(status);
}

const exhaustedModels = new Map();

function isQuotaMessage(message) {
  return /429|quota|rate limit|RESOURCE_EXHAUSTED/i.test(String(message || ""));
}

// Daily quota -> rest the model for 3h. Per-minute limit -> rest it for 70s.
function markModelCooldown(model, message) {
  const daily = /per ?day|daily/i.test(String(message || ""));
  exhaustedModels.set(model, Date.now() + (daily ? 3 * 60 * 60 * 1000 : 70 * 1000));
}

function isModelCoolingDown(model) {
  return (exhaustedModels.get(model) || 0) > Date.now();
}

async function callWithModelFallback(operationName, makeCall) {
  const models = [GEMINI_MODEL, ...GEMINI_FALLBACK_MODELS.filter(name => name !== GEMINI_MODEL)];
  let lastError = null;
  let tried = 0;

  for (const model of models) {
    if (isModelCoolingDown(model)) continue;
    tried++;

    try {
      const result = await callGeminiWithRetry(`${operationName} [${model}]`, () => makeCall(model));
      console.log(`[GEMINI] ${operationName}: ${model}`);
      return result;
    } catch (error) {
      lastError = error;
      const message = String(error?.message || error);
      if (isQuotaMessage(message)) markModelCooldown(model, message);
      if (/429|quota|rate limit|RESOURCE_EXHAUSTED|404|not found|not supported|NOT_FOUND|503|UNAVAILABLE|high demand|overloaded/i.test(message)) {
        console.warn(`[GEMINI] ${model} မရပါ၊ နောက် model ကို စမ်းပါမယ်။`);
        continue;
      }
      throw error;
    }
  }

  if (lastError) throw lastError;
  // Every model is cooling down: fail fast without sending any request.
  throw new Error("QUOTA: Gemini model အားလုံး quota ကုန်နေပါတယ်။ ခဏစောင့်ပြီး (သို့) နောက်ရက် ပြန်စမ်းပါ။");
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
  chunkIndex,
  language = ""
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
      ...(language ? { language } : {}),
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
  audioChunks,
  language = ""
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
        i,
        language
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
   SCENE SCHEMA
========================================================= */

const scenePlanSchema = {
  type: "object",

  properties: {
    scenes: {
      type: "array",

      items: {
        type: "object",

        properties: {
          start_sec: {
            type: "number"
          },

          end_sec: {
            type: "number"
          },

          visual_summary: {
            type: "string"
          },

          event_summary: {
            type: "string"
          },

          narration: {
            type: "string"
          }
        },

        required: [
          "start_sec",
          "end_sec",
          "visual_summary",
          "event_summary",
          "narration"
        ]
      }
    }
  },

  required: [
    "scenes"
  ]
};

/* =========================================================
   JSON PARSER
========================================================= */

function parseGeminiJSON(text) {
  let clean =
    String(text || "").trim();

  clean = clean.replace(
    /^```json\s*/i,
    ""
  );

  clean = clean.replace(
    /^```\s*/i,
    ""
  );

  clean = clean.replace(
    /\s*```$/i,
    ""
  );

  try {
    return JSON.parse(clean);
  } catch {
    const first =
      clean.indexOf("{");

    const last =
      clean.lastIndexOf("}");

    if (
      first >= 0 &&
      last > first
    ) {
      return JSON.parse(
        clean.slice(
          first,
          last + 1
        )
      );
    }

    throw new Error(
      "Gemini returned invalid scene JSON."
    );
  }
}

/* =========================================================
   NORMALIZE SCENES
========================================================= */

function normalizeScenes(
  rawScenes,
  duration,
  maxScenes = MAX_SCENES
) {
  if (!Array.isArray(rawScenes)) {
    throw new Error(
      "Gemini did not return a scene list."
    );
  }

  let scenes = rawScenes
    .map(scene => ({
      start: Number(
        scene.start_sec
      ),

      end: Number(
        scene.end_sec
      ),

      visual: String(
        scene.visual_summary || ""
      ).trim(),

      event: String(
        scene.event_summary || ""
      ).trim(),

      narration: String(
        scene.narration || ""
      ).trim()
    }))

    .filter(
      scene =>
        Number.isFinite(scene.start) &&
        Number.isFinite(scene.end) &&
        scene.end > scene.start &&
        scene.narration
    )

    .sort(
      (a, b) =>
        a.start - b.start
    );

  if (!scenes.length) {
    throw new Error(
      "No usable scenes were returned by Gemini."
    );
  }

  scenes = scenes.map(scene => ({
    ...scene,

    start: Math.max(
      0,
      Math.min(
        duration,
        scene.start
      )
    ),

    end: Math.max(
      0,
      Math.min(
        duration,
        scene.end
      )
    )
  }));

  scenes = scenes.filter(
    scene =>
      scene.end > scene.start
  );

  const normalized = [];

  for (const scene of scenes) {
    if (!normalized.length) {
      normalized.push({
        ...scene,
        start: 0
      });

      continue;
    }

    const previous =
      normalized[
        normalized.length - 1
      ];

    if (
      scene.start >
      previous.end
    ) {
      previous.end =
        scene.start;
    }

    if (
      scene.start <
      previous.end
    ) {
      scene.start =
        previous.end;
    }

    if (
      scene.end >
      scene.start
    ) {
      normalized.push(scene);
    }
  }

  if (normalized.length) {
    normalized[
      normalized.length - 1
    ].end = duration;
  }

  let merged = [];

  for (const scene of normalized) {
    const sceneDuration =
      scene.end - scene.start;

    if (
      sceneDuration <
        MIN_SCENE_SECONDS &&
      merged.length
    ) {
      const previous =
        merged[
          merged.length - 1
        ];

      previous.end =
        scene.end;

      previous.visual =
        `${previous.visual} ${scene.visual}`.trim();

      previous.event =
        `${previous.event} ${scene.event}`.trim();

      previous.narration =
        `${previous.narration} ${scene.narration}`.trim();

    } else {
      merged.push({
        ...scene
      });
    }
  }

  if (
    merged.length >= 2 &&
    merged[0].end -
      merged[0].start <
      MIN_SCENE_SECONDS
  ) {
    const first =
      merged.shift();

    const second =
      merged[0];

    second.start =
      first.start;

    second.visual =
      `${first.visual} ${second.visual}`.trim();

    second.event =
      `${first.event} ${second.event}`.trim();

    second.narration =
      `${first.narration} ${second.narration}`.trim();
  }

  while (
    merged.length >
    maxScenes
  ) {
    let smallestIndex = 0;
    let smallestDuration =
      Infinity;

    for (
      let i = 0;
      i < merged.length;
      i++
    ) {
      const d =
        merged[i].end -
        merged[i].start;

      if (
        d <
        smallestDuration
      ) {
        smallestDuration = d;
        smallestIndex = i;
      }
    }

    if (
      smallestIndex <
      merged.length - 1
    ) {
      const a =
        merged[smallestIndex];

      const b =
        merged[
          smallestIndex + 1
        ];

      b.start = a.start;

      b.visual =
        `${a.visual} ${b.visual}`.trim();

      b.event =
        `${a.event} ${b.event}`.trim();

      b.narration =
        `${a.narration} ${b.narration}`.trim();

      merged.splice(
        smallestIndex,
        1
      );

    } else {
      const a =
        merged[
          smallestIndex - 1
        ];

      const b =
        merged[
          smallestIndex
        ];

      a.end = b.end;

      a.visual =
        `${a.visual} ${b.visual}`.trim();

      a.event =
        `${a.event} ${b.event}`.trim();

      a.narration =
        `${a.narration} ${b.narration}`.trim();

      merged.pop();
    }
  }

  if (merged.length) {
    merged[0].start = 0;

    merged[
      merged.length - 1
    ].end = duration;
  }

  return merged.map(
    (scene, index) => ({
      index: index + 1,

      start: Number(
        scene.start.toFixed(3)
      ),

      end: Number(
        scene.end.toFixed(3)
      ),

      duration: Number(
        (
          scene.end -
          scene.start
        ).toFixed(3)
      ),

      visual: scene.visual,

      event: scene.event,

      narration: scene.narration
    })
  );
}

/* =========================================================
   GEMINI SCENE PLAN
========================================================= */

async function generateScenePlan(
  jobId,
  moviePath,
  duration,
  transcript,
  language = "my",
  style = "cinematic"
) {
  const ai =
    new GoogleGenAI({
      apiKey: requireEnv(
        "GEMINI_API_KEY"
      )
    });

  updateJob(jobId, {
    stage: "Gemini",
    progress: 55,
    message:
      "Gemini is analyzing the movie and matching scenes..."
  });

  const videoFile =
    await uploadVideoToGemini(
      ai,
      moviePath
    );

  const timeline =
    buildTranscriptTimeline(
      transcript.segments
    );

  let languageInstruction =
    "Write natural spoken Myanmar (Burmese).";

  if (language === "en") {
    languageInstruction =
      "Write natural spoken English.";
  }

  // Default narrator persona (suspense / crime recap voice).
  let styleInstruction = `
You are a professional cinematic recap narrator and scriptwriter. Write engaging,
suspenseful, fast-paced movie recap narration.
- Tone & voice: mysterious, thrilling and engaging, suitable for a suspense/crime recap.
- Hook: the very first sentence of the FIRST scene must be a strong attention-grabbing hook.
- Sentence structure: complete medium-length sentences in natural spoken Burmese, mainly past tense (e.g. "...ခဲ့တယ်").
- FLOW IS CRITICAL: write one connected story, not separate scene captions. Continue the same thought across scene boundaries when appropriate; avoid restarting sentences, repeating character names, or using a full-stop cadence after every tiny event.
- Use natural linking phrases only when they fit the story, and vary them. Each line should explain what happened, why it mattered, and what happened next.
- Language: fluent, idiomatic everyday Burmese that a skilled Myanmar YouTube narrator would actually say aloud; never literal translation, bookish wording, or clipped fragments.
- Keep the narration active through most of each scene. Avoid leaving long empty stretches after a short sentence, but do not invent events or add meaningless filler.
- Follow the narration length budget in rule 21.`;

  const syncMode = style === "sync";
  if (syncMode) {
    styleInstruction = `
FAITHFUL SYNC MODE. This is NOT a summary. The narration is a natural Burmese
translation of what is said AND what visibly happens, scene by scene.
- Every scene must start and end where a spoken line (WHISPER TIMELINE segment) or a visible event starts and ends. Use the segment timestamps as the scene boundaries.
- Translate EVERY spoken line inside the scene into natural spoken Burmese (meaning-for-meaning, not word-for-word, no bookish language). Do not skip, merge or reorder dialogue.
- Keep who-says-what clear (e.g. "သူက ... ပြောတယ်"), and add a short description only for important visible actions that happen in silent moments.
- Never describe events from another scene. The narration of a scene must only contain what happens between its start_sec and end_sec.
- Do NOT summarize and do NOT add events or dialogue that are not in the video.
- Even though every line is translated, it must still SOUND like one person telling the story, not like separate subtitles: use natural spoken Burmese, link scenes with spoken connectors (e.g. "အဲဒီအချိန်မှာ", "ဒါပေမဲ့", "နောက်တော့"), use "၊" for short pauses and end every scene with "။".`;
  }

  if (style === "natural") {
    styleInstruction = `
You are a Burmese storyteller telling this movie to a friend, as ONE continuous story from start to finish.
- Voice: warm, natural, spoken Burmese (ပြောစကား), like a person narrating a story out loud. Never bookish, never robotic, never a list of separate captions.
- Flow: make this sound like a real YouTube movie recap, not a synopsis, subtitle translation, or a list of scene descriptions. Tell the plot in a connected, engaging spoken-story flow. Link each scene naturally to the previous one with varied conversational transitions (e.g. "အဲဒီအချိန်မှာ", "ဒါပေမဲ့", "နောက်တော့", "ဒီလိုနဲ့") and make each scene lead into the next. Do not restart or re-introduce characters in every scene.
- Content: focus on what happens, why it matters, what the character wants, and what changes next. Keep important cause-and-effect and twists; avoid generic lines such as "ဒီအဖြစ်အပျက်က သူ့ဘဝကို ပြောင်းလဲစေခဲ့တယ်" unless the movie actually supports them. Do not merely describe visible actions.
- Sentences: use complete, idiomatic spoken Burmese with natural word order and conversational phrasing. Avoid literal English sentence structure, repetitive sentence openings, very short broken fragments, and overlong run-on sentences.
- Punctuation for speech rhythm: use "၊" for short breathing pauses and "။" at the end of every sentence, so the voice pauses naturally. Every scene narration must end with "။".
- Tense: tell it in past tense ("...ခဲ့တယ်"), with a gripping hook in the very first sentence of the first scene.
- Emotion: let tension, surprise and relief show through word choice, not through exclamation marks.
- The length budget in rule 21 still applies: never write more than can be spoken in the scene.`;
  }

  if (style === "short") {
    styleInstruction =
      "Use concise and fast-paced movie recap narration.";
  }

  if (style === "storytelling") {
    styleInstruction =
      "Use smooth storytelling with suspense and emotional flow.";
  }

  if (style === "detailed") {
    styleInstruction =
      "Use detailed but natural movie recap narration.";
  }

  const prompt = `
You are creating a PROFESSIONAL SYNCHRONIZED MOVIE RECAP.

You have access to the actual movie video.

You MUST analyze what is visibly happening.

You also have the original Whisper transcript with timestamps.

Your output will be used to create a final video where each narration
is placed directly over its selected video scene.

SYNCHRONIZATION IS EXTREMELY IMPORTANT.

RULES:

1. Analyze the actual movie video.
2. Identify meaningful visual scene changes.
3. Every scene needs start_sec and end_sec.
4. Use real timestamps from the movie.
5. Keep scenes chronological.
6. Do not overlap scenes.
7. Do not leave gaps.
8. Cover the complete movie from 0 to ${duration.toFixed(2)} seconds.
9. ${syncMode ? "Scene length follows the dialogue and events, not a fixed size." : "Prefer scenes around 6 to 14 seconds."}
10. ${syncMode ? `Create one scene per spoken line or visible event, up to ${SYNC_MAX_SCENES} scenes. Scenes may be as short as 1.5 seconds.` : `Create approximately 8 to ${MAX_SCENES} scenes.`}
11. Match narration to what is actually visible.
12. Use Whisper transcript as supporting information.
13. Do not invent events.
14. Do not invent characters.
15. Do not invent dialogue.
16. Do not move events to incorrect timestamps.
17. Keep important story events.
18. Keep the ending.
19. Avoid meaningless filler.
20. Narration must sound natural when spoken.
21. NARRATION LENGTH AND FLOW (very important): aim for narration that naturally occupies about 75–90% of each scene, leaving only a brief breathing pause. Do not make scenes so short that the video becomes silent for several seconds.
    - Myanmar: target roughly 12–14 Burmese characters per second of scene duration (a 10 second scene usually needs around 120–140 characters); do not exceed 16 characters per second. Prefer natural complete phrasing over padding or repetitive wording.
    - English: target about 2.2–2.6 words per second of scene duration.
    - ${syncMode ? "In sync mode all dialogue in the scene MUST still be translated; if it is too long, choose a shorter natural Burmese wording of the same meaning instead of dropping lines." : "Do NOT describe every small action. Summarize key story events with enough connected detail to sustain narration through the scene; skip minor details."}
    - Keep pauses brief and natural. Never add filler just to hit the length target, and never rush speech unnaturally.
22. Do not use Markdown.
23. Do not use headings in narration.
24. Do not mention AI.
25. Return ONLY JSON.

LANGUAGE:
${languageInstruction}

STYLE:
${styleInstruction}

JSON FORMAT:

{
  "scenes": [
    {
      "start_sec": 0,
      "end_sec": 8,
      "visual_summary": "What is visibly happening.",
      "event_summary": "Important story event.",
      "narration": "Natural recap narration."
    }
  ]
}

WHISPER TIMELINE:

${timeline}
`;

  let response;
  try {
  response =
    await callWithModelFallback(
      "Scene-Synchronized Movie Analysis",
      (model) =>
        ai.interactions.create({
          model,

          input: [
            {
              type: "video",

              uri: videoFile.uri,

              mime_type:
                videoFile.mimeType,

              processing: {
                type: "static",
                fps: SCENE_FPS
              }
            },

            {
              type: "text",
              text: prompt
            }
          ],

          response_format: {
            type: "text",
            mime_type:
              "application/json",
            schema:
              scenePlanSchema
          }
        })
    );
  } catch (err) {
    try { await ai.files.delete({ name: videoFile.name }); } catch {}
    throw err;
  }

  // Free the uploaded video from Gemini file storage (it is no longer needed).
  try {
    await ai.files.delete({ name: videoFile.name });
  } catch {}

  const outputText =
    response?.output_text ||
    response?.outputText ||
    "";

  if (!outputText) {
    throw new Error(
      "Gemini returned an empty scene plan."
    );
  }

  const parsed =
    parseGeminiJSON(
      outputText
    );

  const scenes =
    normalizeScenes(
      parsed.scenes,
      duration,
      style === "sync" ? SYNC_MAX_SCENES : MAX_SCENES
    );

  if (!scenes.length) {
    throw new Error(
      "Gemini returned no usable scenes."
    );
  }

  console.log(
    `[SCENE SYNC] ${scenes.length} scenes created`
  );

  {
    const rates = scenes.map(scene => String(scene.narration || "").length / Math.max(1, scene.end - scene.start));
    const average = rates.reduce((a, b) => a + b, 0) / rates.length;
    const tooLong = rates.filter(rate => rate > 14).length;
    console.log(`[SCENE SYNC] narration density avg ${average.toFixed(1)} chars/sec, ${tooLong} scene(s) too long`);
  }

  return {
    scenes
  };
}

/* =========================================================
   TTS VOICE
========================================================= */

function resolveVoice(voice) {
  if (voice === "Thiha") return "Puck";
  if (voice === "Nilar") return "Kore";
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

async function generateGeminiTTS(
  ai,
  scenes,
  outputPath,
  voice,
  recapStyle
) {
  const actualVoice = resolveVoice(voice);
  const narrations = scenes.map(scene => String(scene.narration || "").trim());

  if (narrations.some(text => !text)) {
    const bad = scenes.find(scene => !String(scene.narration || "").trim());
    throw new Error(`Scene ${bad?.index ?? "?"} has no narration.`);
  }

  const fullText = narrations.join(" ").trim();
  if (!fullText) throw new Error("No narration text was available for TTS.");

  const response = await callGeminiWithRetry(
    "Full Recap TTS",
    () => ai.models.generateContent({
      model: GEMINI_TTS_MODEL,
      contents: [{
        role: "user",
        parts: [{
          text: fullText,
          speech_metadata: {
            style: recapStyleText(recapStyle)
          }
        }]
      }],
      config: {
        responseModalities: ["AUDIO"],
        speechConfig: { voiceConfig: { voice: actualVoice } }
      }
    })
  );

  const base64 = response?.candidates?.[0]?.content?.parts?.find(part => part?.inlineData?.data)?.inlineData?.data;
  if (!base64) throw new Error("Gemini returned no full narration audio.");

  fs.writeFileSync(outputPath, Buffer.from(base64, "base64"));
  if (fs.statSync(outputPath).size < 100) throw new Error("Generated full narration audio is invalid.");
  return outputPath;
}

/* =========================================================
   EXTRA FREE TTS ENGINES (Azure / Google Translate) + AUTO FALLBACK
========================================================= */

const ttsEngineByJob = new Map();
const MALE_VOICES = new Set(["male","Thiha","Puck","Charon","Fenrir","Orus","Enceladus","Iapetus","Umbriel","Algenib","Rasalgethi","Alnilam","Schedar","Achird","Zubenelgenubi","Sadachbia","Sadaltager"]);
function isMaleVoice(voice) { return MALE_VOICES.has(String(voice)); }
const TTS_ENGINES = ["auto", "gemini", "edge", "azure", "google", "elevenlabs"];

function splitTtsText(text, maxChars) {
  const pieces = String(text).split(/(?<=[။!?.\n])\s*/).filter(Boolean);
  const parts = [];
  let current = "";
  for (const piece of pieces) {
    if (piece.length > maxChars) {
      if (current) { parts.push(current); current = ""; }
      for (let i = 0; i < piece.length; i += maxChars) parts.push(piece.slice(i, i + maxChars));
      continue;
    }
    if ((current + " " + piece).length > maxChars && current) { parts.push(current); current = ""; }
    current = current ? `${current} ${piece}` : piece;
  }
  if (current) parts.push(current);
  return parts;
}

function xmlEscapeTts(text) {
  return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&apos;");
}

async function audioBuffersToWav(buffers, outputPath, ext) {
  const tempPath = `${outputPath}.${ext}`;
  fs.writeFileSync(tempPath, Buffer.concat(buffers));
  try {
    await runCommand("ffmpeg", ["-y", "-i", tempPath, "-vn", "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav", outputPath]);
  } finally {
    try { fs.unlinkSync(tempPath); } catch {}
  }
}

// Free Microsoft Edge neural voices (same Burmese voices as Azure), no key or
// card needed. Uses the `edge-tts` command (installed in the Dockerfile via pip).
// ElevenLabs (paid/credit based). Burmese is supported by eleven_v3 only
// (NOT eleven_multilingual_v2 / flash). Needs ELEVENLABS_API_KEY + ELEVENLABS_VOICE_ID
// (optional ELEVENLABS_VOICE_ID_MALE / ELEVENLABS_VOICE_ID_FEMALE).
async function elevenLabsTts(text, outputPath, voice) {
  const key = (process.env.ELEVENLABS_API_KEY || "").trim();
  const male = (process.env.ELEVENLABS_VOICE_ID_MALE || "").trim();
  const female = (process.env.ELEVENLABS_VOICE_ID_FEMALE || "").trim();
  const voiceId = ((isMaleVoice(voice) ? male : female) || process.env.ELEVENLABS_VOICE_ID || "").trim();
  if (!key || !voiceId) throw new Error("ELEVENLABS_API_KEY / ELEVENLABS_VOICE_ID မထည့်ရသေးပါ");
  const model = (process.env.ELEVENLABS_MODEL || "eleven_v3").trim();
  // eleven_v3 only accepts stability 0.0 (Creative) / 0.5 (Natural) / 1.0 (Robust).
  const isV3 = /v3/i.test(model);
  const voice_settings = {
    stability: Number(process.env.ELEVENLABS_STABILITY || (isV3 ? 0.5 : 0.35)),
    similarity_boost: 0.8,
    style: 0.2,
    use_speaker_boost: true
  };
  const buffers = [];
  for (const part of splitTtsText(text, 2500)) {
    let audio = null;
    let lastError = null;
    for (let attempt = 0; attempt < 3 && !audio; attempt++) {
      const res = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${encodeURIComponent(voiceId)}?output_format=mp3_44100_128`, {
        method: "POST",
        headers: { "xi-api-key": key, "Content-Type": "application/json", Accept: "audio/mpeg" },
        body: JSON.stringify({ text: part, model_id: model, voice_settings })
      });
      if (res.ok) {
        audio = Buffer.from(await res.arrayBuffer());
      } else {
        const detail = (await res.text().catch(() => "")).slice(0, 160);
        lastError = new Error(`ElevenLabs HTTP ${res.status} ${detail}`);
        if (res.status !== 429 && res.status < 500) break;
        await new Promise(r => setTimeout(r, 2000 * (attempt + 1)));
      }
    }
    if (!audio || audio.length < 100) throw lastError || new Error("ElevenLabs empty audio");
    buffers.push(audio);
  }
  await audioBuffersToWav(buffers, outputPath, "mp3");
}

let edgeInstallTried = false;
async function edgeTts(text, outputPath, voice) {
  const voiceName = isMaleVoice(voice) ? "my-MM-ThihaNeural" : "my-MM-NilarNeural";
  const base = `${outputPath}.edge`;
  const textFile = `${base}.txt`;
  const mediaFile = `${base}.mp3`;
  fs.writeFileSync(textFile, String(text), "utf8");
  try {
    let lastError = null;
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const edgeArgs = ["--voice", voiceName, "--file", textFile, "--write-media", mediaFile];
        // Optional tuning, e.g. EDGE_RATE=-5% (slower) / EDGE_PITCH=-2Hz (calmer).
        const edgeRate = (process.env.EDGE_RATE || "").trim();
        const edgePitch = (process.env.EDGE_PITCH || "").trim();
        if (edgeRate) edgeArgs.push(`--rate=${edgeRate}`);
        if (edgePitch) edgeArgs.push(`--pitch=${edgePitch}`);
        const launchers = [
          ["edge-tts", edgeArgs],
          ["python3", ["-m", "edge_tts", ...edgeArgs]],
          ["python", ["-m", "edge_tts", ...edgeArgs]]
        ];
        let launched = false;
        let launchError = null;
        for (const [cmd, args] of launchers) {
          try {
            await runCommand(cmd, args);
            launched = true;
            break;
          } catch (error) {
            launchError = error;
            if (error?.code !== "ENOENT" && !/ENOENT|No module named/i.test(String(error?.message || error) + String(error?.stderr || ""))) throw error;
          }
        }
        if (!launched && !edgeInstallTried) {
          // One-time automatic install attempt (needs pip + internet on the server).
          edgeInstallTried = true;
          const pipTries = [
            ["python3", ["-m", "pip", "install", "--user", "--break-system-packages", "edge-tts"]],
            ["python", ["-m", "pip", "install", "--user", "--break-system-packages", "edge-tts"]],
            ["pip3", ["install", "--user", "--break-system-packages", "edge-tts"]],
            ["pip", ["install", "--user", "--break-system-packages", "edge-tts"]]
          ];
          for (const [cmd, args] of pipTries) {
            try {
              await runCommand(cmd, args);
              console.log("[EDGE TTS] edge-tts installed automatically");
              break;
            } catch (error) {
              console.warn(`[EDGE TTS] auto-install via ${cmd} failed:`, String(error?.message || error).slice(0, 160));
            }
          }
        }
        if (!launched && edgeInstallTried) {
          for (const [cmd, args] of launchers.slice(1)) {
            try { await runCommand(cmd, args); launched = true; break; } catch (error) { launchError = error; }
          }
        }
        if (!launched) {
          throw new Error("edge-tts မထည့်သွင်းရသေးပါ။ Server မှာ `pip install edge-tts` run ပါ။ (" + String(launchError?.message || launchError).slice(0, 100) + ")");
        }
        if (fs.existsSync(mediaFile) && fs.statSync(mediaFile).size > 100) { lastError = null; break; }
        lastError = new Error("Edge TTS empty audio");
      } catch (error) {
        lastError = error;
      }
      await new Promise(r => setTimeout(r, 1500 * (attempt + 1)));
    }
    if (lastError) throw lastError;
    await audioBuffersToWav([fs.readFileSync(mediaFile)], outputPath, "mp3");
  } finally {
    for (const f of [textFile, mediaFile]) { try { fs.unlinkSync(f); } catch {} }
  }
}

async function azureTts(text, outputPath, voice) {
  const key = (process.env.AZURE_SPEECH_KEY || "").trim();
  const region = (process.env.AZURE_SPEECH_REGION || "").trim();
  if (!key || !region) throw new Error("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION မထည့်ရသေးပါ");
  const voiceName = isMaleVoice(voice) ? "my-MM-ThihaNeural" : "my-MM-NilarNeural";
  const buffers = [];
  for (const part of splitTtsText(text, 1200)) {
    const ssml = `<speak version='1.0' xml:lang='my-MM'><voice xml:lang='my-MM' name='${voiceName}'>${xmlEscapeTts(part)}</voice></speak>`;
    let audio = null;
    let lastError = null;
    for (let attempt = 0; attempt < 3 && !audio; attempt++) {
      const res = await fetch(`https://${region}.tts.speech.microsoft.com/cognitiveservices/v1`, {
        method: "POST",
        headers: {
          "Ocp-Apim-Subscription-Key": key,
          "Content-Type": "application/ssml+xml",
          "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
          "User-Agent": "YNTRecapTool"
        },
        body: ssml
      });
      if (res.ok) {
        audio = Buffer.from(await res.arrayBuffer());
      } else {
        lastError = new Error(`Azure TTS HTTP ${res.status}`);
        if (res.status !== 429 && res.status < 500) break;
        await new Promise(r => setTimeout(r, 2000 * (attempt + 1)));
      }
    }
    if (!audio || audio.length < 100) throw lastError || new Error("Azure TTS empty audio");
    buffers.push(audio);
  }
  await audioBuffersToWav(buffers, outputPath, "mp3");
}

async function googleTranslateTts(text, outputPath) {
  const buffers = [];
  for (const part of splitTtsText(text, 180)) {
    const url = `https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl=my&q=${encodeURIComponent(part)}`;
    const res = await fetch(url, { headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36" } });
    if (!res.ok) throw new Error(`Google Translate TTS HTTP ${res.status}`);
    const audio = Buffer.from(await res.arrayBuffer());
    if (audio.length < 100) throw new Error("Google Translate TTS empty audio");
    buffers.push(audio);
    await new Promise(r => setTimeout(r, 250));
  }
  await audioBuffersToWav(buffers, outputPath, "mp3");
}

// Join scene narrations into one text. A scene that does not end with
// sentence punctuation gets "။" so the voice pauses instead of running on.
function joinNarrationsForTts(scenes) {
  const narrations = scenes
    .map(scene => String(scene.narration || "").trim())
    .filter(Boolean);

  // A scene boundary is not necessarily the end of a spoken thought. Use a
  // light Burmese comma at boundaries so TTS flows into the next scene rather
  // than inserting a full-stop pause after every scene. Keep the final stop.
  return narrations
    .map((text, index) => {
      if (index === narrations.length - 1) {
        return /[။!?…]$/.test(text) ? text : `${text}။`;
      }
      return text.replace(/[။!?…]+$/u, "၊").replace(/[၊,]+$/u, "၊");
    })
    .join(" ")
    .trim();
}

async function generateFullTTS(ai, scenes, outputPath, voice, engine = "auto", recapStyle) {
  const fullText = joinNarrationsForTts(scenes);
  if (!fullText) throw new Error("No narration text was available for TTS.");

  const nativeBurmese = voice === "Thiha" || voice === "Nilar";
  const order = engine === "auto"
    ? (nativeBurmese ? ["edge", "azure", "gemini", "google"] : ["gemini", "edge", "azure", "google"])
    : [engine];
  const errors = [];

  for (const name of order) {
    try {
      if (name === "gemini" && isModelCoolingDown(GEMINI_TTS_MODEL)) throw new Error("Gemini TTS quota cooldown — skipped");
      if (name === "gemini") await generateGeminiTTS(ai, scenes, outputPath, voice, recapStyle);
      else if (name === "edge") await edgeTts(fullText, outputPath, voice);
      else if (name === "elevenlabs") await elevenLabsTts(fullText, outputPath, voice);
      else if (name === "azure") await azureTts(fullText, outputPath, voice);
      else if (name === "google") await googleTranslateTts(fullText, outputPath);
      else throw new Error("မသိတဲ့ voice engine");

      if (!fs.existsSync(outputPath) || fs.statSync(outputPath).size < 100) throw new Error("audio file invalid");
      console.log(`[TTS] engine=${name}`);
      return outputPath;
    } catch (error) {
      const message = String(error?.message || error).slice(0, 200);
      if (name === "gemini" && isQuotaMessage(message)) markModelCooldown(GEMINI_TTS_MODEL, String(error?.message || error));
      errors.push(`${name}: ${message}`);
      console.error(`[TTS] ${name} failed: ${message}`);
    }
  }
  throw new Error(`အသံထုတ်လို့မရပါ။ ${errors.join(" | ")}`);
}

async function detectSilences(audioPath) {
  const result = await runCommand("ffmpeg", [
    "-i", audioPath,
    "-af", `silencedetect=noise=${process.env.SILENCE_DB || "-35dB"}:d=0.12`,
    "-f", "null", "-"
  ]);
  const text = `${result.stderr || ""}\n${result.stdout || ""}`;
  const starts = [...text.matchAll(/silence_start: (-?[\d.]+)/g)].map(m => Math.max(0, Number(m[1])));
  const ends = [...text.matchAll(/silence_end: ([\d.]+) \| silence_duration: ([\d.]+)/g)].map(m => ({ end: Number(m[1]), dur: Number(m[2]) }));
  const silences = [];
  for (let i = 0; i < Math.min(starts.length, ends.length); i++) {
    silences.push({ mid: (starts[i] + ends[i].end) / 2, dur: ends[i].dur });
  }
  return silences;
}

// Cut the narration at real pauses (not mid-word): start from the proportional
// estimate, then snap each boundary to the best nearby silence.
function chooseSceneCuts(silences, total, weights) {
  const n = weights.length;
  const cuts = [];
  let prev = 0;

  for (let i = 0; i < n - 1; i++) {
    const remainingWeight = weights.slice(i).reduce((sum, value) => sum + value, 0);
    const estimate = prev + (total - prev) * (weights[i] / remainingWeight);
    const segment = (total - prev) / (n - i);

    const pick = window => {
      let found = null;
      let foundScore = Infinity;
      for (const silence of silences) {
        if (silence.mid <= prev + 0.4 || silence.mid >= total - 0.3) continue;
        const distance = Math.abs(silence.mid - estimate);
        if (distance > window) continue;
        const score = distance - 2 * Math.min(silence.dur, 1);
        if (score < foundScore) {
          foundScore = score;
          found = silence.mid;
        }
      }
      return found;
    };

    const best = pick(Math.max(2.5, 0.6 * segment)) ?? pick(Math.max(4, 1.2 * segment));

    const cut = best ?? Math.min(Math.max(estimate, prev + 0.2), total - 0.1);
    cuts.push(cut);
    prev = cut;
  }
  return cuts;
}

async function splitFullTTSForScenes(fullAudio, scenes, sceneDir) {
  const totalAudioDuration = await getAudioDuration(fullAudio);
  const weights = scenes.map(scene => Math.max(1, String(scene.narration || "").trim().length));

  let silences = [];
  try {
    silences = await detectSilences(fullAudio);
  } catch (error) {
    console.warn("[SPLIT] silence detection failed, using proportional split:", String(error?.message || error).slice(0, 160));
  }

  const cuts = chooseSceneCuts(silences, totalAudioDuration, weights);
  const bounds = [0, ...cuts, totalAudioDuration];
  console.log(`[SPLIT] ${silences.length} pauses found, cuts at: ${cuts.map(c => c.toFixed(1)).join(", ")}`);

  const files = [];
  for (let i = 0; i < scenes.length; i++) {
    const scene = scenes[i];
    const start = bounds[i];
    const duration = Math.max(0.05, bounds[i + 1] - start);
    const number = String(scene.index).padStart(3, "0");
    const rawPath = path.join(sceneDir, `scene-${number}-raw.wav`);

    await runCommand("ffmpeg", [
      "-y", "-ss", start.toFixed(3), "-i", fullAudio,
      "-t", duration.toFixed(3), "-vn", "-ar", "24000", "-ac", "1",
      "-c:a", "pcm_s16le", rawPath
    ]);

    files.push(rawPath);
  }
  return files;
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
   RECAP VOICE SETTINGS + SPEED BY STYLE
========================================================= */

// Reference voice settings for stability-based TTS APIs (e.g. ElevenLabs).
// Gemini TTS does not take these numbers, so for Gemini the same intent is
// sent as the style prompt below (see recapStyleText).
const recapVoiceSettings = {
  stability: 0.35,          // more expressive / emotional delivery
  similarity_boost: 0.75,
  style_exaggeration: 0.25, // more "recap" feel
  use_speaker_boost: true
};

const styleByJob = new Map();
// Voice speed picked in the UI: multiplies the style speed (slow / normal / fast).
const voiceSpeedByJob = new Map();
const VOICE_SPEEDS = { slow: 0.92, normal: 1, fast: 1.12 };
const voiceSpeedOf = jobId => VOICE_SPEEDS[voiceSpeedByJob.get(jobId)] || 1;

// Speed (ffmpeg atempo) per recap style. Default recap speed is 1.1x.
function getRecapSpeed(recapStyle) {
  switch (String(recapStyle || "").toLowerCase()) {
    case "sync":
    case "natural":
      return 1.0; // storyteller / faithful dialogue: keep natural speed
    case "fast":
    case "short":
      return 1.25; // Shorts / TikTok pace
    case "dramatic":
    case "cinematic":
      return 1.05; // tense, clear storytelling
    default:
      return 1.1;
  }
}

function recapStyleText(recapStyle) {
  const base = "Professional Burmese-language YouTube movie recap, not a plot-summary readout. Clear Burmese pronunciation, idiomatic conversational Myanmar, smooth connected storytelling, natural phrase boundaries, steady medium pace, controlled emotion, no robotic cadence, no clipped syllables, no rushed words. Sound like a skilled human narrator explaining the movie to viewers.";
  switch (String(recapStyle || "").toLowerCase()) {
    case "sync":
      return "Natural spoken Burmese, clear pronunciation, natural conversational pacing, like a dubbing narrator following the scene.";
    case "fast":
    case "short":
      return base + " Energetic, punchy, fast TikTok/Shorts-style delivery.";
    case "dramatic":
    case "cinematic":
      return base + " Tense, dramatic, expressive delivery with emphasis on key moments.";
    default:
      return base + " Slightly expressive, recap-style energy.";
  }
}

// Standalone speed change (keeps pitch) for any audio file.
async function adjustAudioSpeed(inputAudioPath, outputAudioPath, recapStyle) {
  const filters = buildAtempoFilters(getRecapSpeed(recapStyle));
  await runCommand("ffmpeg", [
    "-y", "-i", inputAudioPath,
    ...(filters.length ? ["-filter:a", filters.join(",")] : []),
    "-vn", outputAudioPath
  ]);
  return outputAudioPath;
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

// Timing limits (see sync.js planSceneTiming). Env vars still work:
//   MIN_TEMPO / MAX_TEMPO   voice speed range relative to the style speed
//   MAX_VIDEO_SLOW          how much the video may be slowed before freezing
const TIMING_OPTS = {
  minTempo: Number(process.env.MIN_TEMPO || 0.9),
  maxTempo: Number(process.env.MAX_TEMPO || 1.25),
  maxVideoSlow: Number(process.env.MAX_VIDEO_SLOW || 1.2)
};

// highpass removes low rumble, acompressor evens out loud/quiet words, fades stop clicks at scene joins.
const VOICE_POLISH_FILTER = "highpass=f=70,acompressor=threshold=-20dB:ratio=2.5:attack=8:release=120:makeup=2,afade=t=in:d=0.012";

// Apply the planned tempo to one scene's voice and pad it with silence to the
// exact length of the scene (plan.outDur) so audio and video end together.
async function fitAudioToScene(inputAudio, outputAudio, plan) {
  const filters = [
    ...buildAtempoFilters(plan.tempo),
    // Voice polish: cut rumble/pops, tame loud peaks, soften clip edges (no clicks).
    VOICE_POLISH_FILTER,
    "apad",
    `atrim=duration=${plan.outDur.toFixed(3)}`
  ];
  await runCommand("ffmpeg", [
    "-y", "-i", inputAudio,
    "-af", filters.join(","),
    "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le",
    outputAudio
  ]);
}


/* =========================================================
   CUT VIDEO SCENE — ORIGINAL VIDEO ONLY
========================================================= */

async function renderSceneVideo(
  moviePath,
  scene,
  outputPath,
  filterComplex = null
) {
  // -t is an INPUT option here (before -i): it reads exactly this scene's
  // footage. The output may be longer than that when the filter slows the video
  // or freezes the last frame to hold the narration.
  const duration = Math.max(0.05, scene.end - scene.start);

  await runCommand(
    "ffmpeg",
    [
      "-y",
      "-ss", scene.start.toFixed(3),
      "-t", duration.toFixed(3),
      "-i", moviePath,
      ...(filterComplex
        ? ["-filter_complex", filterComplex, "-map", "[vout]"]
        : []),
      "-an",
      "-c:v", "libx264",
      "-preset", "veryfast",
      "-crf", "23",
      "-pix_fmt", "yuv420p",
      "-movflags", "+faststart",
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

function readBurn(jobFolder) {
  try {
    const burn = normalizeBurn(fs.readFileSync(path.join(jobFolder, "burn.json"), "utf8"));
    // Subtitles are delivered as a separate SRT file now, never burned into
    // the video (blur boxes and aspect ratio still are).
    if (burn) burn.subtitle.enabled = false;
    return burn;
  } catch {
    return null;
  }
}

/* =========================================================
   PER-SCENE NARRATION
   Every scene is spoken on its own, so the audio of a scene can only contain
   that scene's words and its real length is known exactly. (The old approach
   spoke the whole script at once and guessed where to cut it.)
========================================================= */

const TTS_CONCURRENCY = Math.max(1, Number(process.env.TTS_CONCURRENCY || 3));

async function mapLimit(items, limit, worker) {
  const results = new Array(items.length);
  let next = 0;
  let failed = false;
  async function run() {
    while (!failed) {
      const i = next++;
      if (i >= items.length) return;
      try {
        results[i] = await worker(items[i], i);
      } catch (error) {
        failed = true;
        throw error;
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, run));
  return results;
}

// Cut leading/trailing silence so the measured length is the real speech length.
async function trimAndMeasure(inputPath, outputPath) {
  await runCommand("ffmpeg", [
    "-y", "-i", inputPath,
    "-af", TRIM_SILENCE_FILTER,
    "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le",
    outputPath
  ]);
  return getAudioDuration(outputPath);
}

async function synthesizeSceneAudios(jobId, ai, scenes, sceneDir, voice) {
  const engine = ttsEngineByJob.get(jobId) || "auto";
  const style = styleByJob.get(jobId);
  const numberOf = scene => String(scene.index).padStart(3, "0");

  // Escape hatch: TTS_MODE=full speaks everything in one request (fewer API
  // calls, e.g. on Gemini's free tier) and cuts it at pauses. Less accurate.
  if (String(process.env.TTS_MODE || "scene").toLowerCase() === "full") {
    const fullTTS = path.join(sceneDir, "full-narration.wav");
    await generateFullTTS(ai, scenes, fullTTS, voice, engine, style);
    const parts = await splitFullTTSForScenes(fullTTS, scenes, sceneDir);
    const out = [];
    for (let i = 0; i < scenes.length; i++) {
      const trimPath = path.join(sceneDir, `scene-${numberOf(scenes[i])}-trim.wav`);
      out.push({ rawPath: trimPath, speech: Math.max(0.2, await trimAndMeasure(parts[i], trimPath)) });
    }
    return out;
  }

  let done = 0;
  return mapLimit(scenes, TTS_CONCURRENCY, async scene => {
    const ttsPath = path.join(sceneDir, `scene-${numberOf(scene)}-tts.wav`);
    const trimPath = path.join(sceneDir, `scene-${numberOf(scene)}-trim.wav`);
    let lastError = null;
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        await generateFullTTS(ai, [scene], ttsPath, voice, engine, style);
        const speech = await trimAndMeasure(ttsPath, trimPath);
        if (speech < 0.15) throw new Error(`Scene ${scene.index} မှာ အသံ မထွက်ပါ။`);
        done++;
        updateJob(jobId, {
          stage: "Voice",
          progress: 65 + Math.round((done / scenes.length) * 3),
          message: `Scene ${done} / ${scenes.length} — generating narration voice...`
        });
        return { rawPath: trimPath, speech };
      } catch (error) {
        lastError = error;
        console.warn(`[TTS] scene ${scene.index} attempt ${attempt + 1} failed: ${String(error?.message || error).slice(0, 160)}`);
        if (attempt === 0) await sleep(1500);
      }
    }
    throw lastError;
  });
}

async function renderSceneSyncVideo(
  jobId,
  moviePath,
  scenePlan,
  voice,
  jobFolder
) {
  const ai = new GoogleGenAI({ apiKey: requireEnv("GEMINI_API_KEY") });
  const sceneDir = path.join(jobFolder, "scenes");
  fs.mkdirSync(sceneDir, { recursive: true });

  updateJob(jobId, { stage: "Voice", progress: 65, message: "Generating narration voice..." });

  const voiced = await synthesizeSceneAudios(jobId, ai, scenePlan.scenes, sceneDir, voice);

  const burn = readBurn(jobFolder);
  const burnOn = burnActive(burn);
  const videoSize = burnOn ? await probeVideoSize(moviePath) : null;
  if (burnOn) {
    console.log(`[BURN] blurBoxes=${burn.blur.enabled ? burn.blur.boxes.length : 0} aspect=${burn.aspectRatio} size=${videoSize.width}x${videoSize.height}`);
  }

  const baseSpeed = getRecapSpeed(styleByJob.get(jobId)) * voiceSpeedOf(jobId);
  const renderedScenes = [];
  const cues = [];
  let offset = 0; // seconds from the start of the final video

  for (let i = 0; i < scenePlan.scenes.length; i++) {
    const scene = scenePlan.scenes[i];
    const progress = 68 + Math.round((i / scenePlan.scenes.length) * 23);
    updateJob(jobId, {
      stage: "FFmpeg", progress,
      message: `Scene ${scene.index} / ${scenePlan.scenes.length} — syncing video and voice...`
    });

    const number = String(scene.index).padStart(3, "0");
    const fittedTTS = path.join(sceneDir, `scene-${number}-audio.wav`);
    const sceneVideo = path.join(sceneDir, `scene-${number}-video.mp4`);
    const sceneFinal = path.join(sceneDir, `scene-${number}-final.mp4`);

    const slot = Math.max(0.3, scene.end - scene.start);
    const plan = planSceneTiming({ slot, speech: voiced[i].speech, base: baseSpeed, ...TIMING_OPTS });
    console.log(
      `[SYNC] scene ${scene.index}: speech ${voiced[i].speech.toFixed(1)}s, slot ${slot.toFixed(1)}s, ` +
      `voice x${plan.tempo.toFixed(2)}, video x${plan.videoStretch.toFixed(2)}, freeze ${plan.padSec.toFixed(1)}s`
    );

    await fitAudioToScene(voiced[i].rawPath, fittedTTS, plan);

    const baseFilter = burnOn ? buildBurnFilter(burn, videoSize, null) : null;
    const filterComplex = appendVideoTiming(baseFilter, plan.videoStretch, plan.padSec);

    await renderSceneVideo(moviePath, scene, sceneVideo, filterComplex);
    await muxScene(sceneVideo, fittedTTS, sceneFinal);
    renderedScenes.push(sceneFinal);

    // Subtitle cues follow the real speech inside this scene; the next scene
    // starts where this scene's finished clip really ends.
    cues.push(...makeSceneCues(scene.narration, offset, plan.speechDur));
    offset += await getVideoDuration(sceneFinal);
  }

  updateJob(jobId, { stage: "FFmpeg", progress: 94, message: "Joining synchronized scenes..." });

  const listPath = path.join(jobFolder, "scene-list.txt");
  const outputPath = path.join(jobFolder, "YNT-One-Clips-Recap.mp4");
  await concatScenes(renderedScenes, outputPath, listPath);
  await normalizeLoudness(outputPath);

  const srtPath = path.join(jobFolder, "subtitles.srt");
  fs.writeFileSync(srtPath, buildSrt(cues), "utf8");
  console.log(`[SRT] ${cues.length} cues written`);

  return { outputPath, srtPath, cueCount: cues.length };
}

// One loudness pass over the whole video (-16 LUFS, the usual level for YouTube /
// TikTok) so every scene sounds equally loud. Video is copied, never re-encoded.
// If anything goes wrong the un-normalized file is kept.
async function normalizeLoudness(videoPath) {
  const tmp = videoPath.replace(/\.mp4$/i, ".loud.mp4");
  try {
    await runCommand("ffmpeg", [
      "-y", "-i", videoPath,
      "-map", "0:v", "-map", "0:a:0",
      "-c:v", "copy",
      "-af", "loudnorm=I=-16:TP=-1.5:LRA=9",
      "-ar", "44100", "-c:a", "aac", "-b:a", "192k",
      "-movflags", "+faststart",
      tmp
    ]);
    if (fs.existsSync(tmp) && fs.statSync(tmp).size > 0) fs.renameSync(tmp, videoPath);
  } catch (error) {
    console.error("[LOUDNESS] skipped:", String(error?.message || error).slice(0, 200));
    try { fs.rmSync(tmp, { force: true }); } catch {}
  }
}

/* =========================================================
   PROCESS JOB
========================================================= */

async function finishRecap(jobId, moviePath, scenePlan, voice, jobFolder) {
    /* -----------------------------------------
       RENDER
    ----------------------------------------- */

    const rendered =
      await renderSceneSyncVideo(
        jobId,
        moviePath,
        scenePlan,
        voice,
        jobFolder
      );
    const outputPath = rendered.outputPath;

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
          `/api/download/${jobId}`,
        srt: {
          path: rendered.srtPath,
          cues: rendered.cueCount,
          url: `/api/srt/${jobId}`
        }
      }
    });

    console.log(
      `[JOB ${jobId}] COMPLETED`
    );
}

async function processOneClip(
  jobId,
  moviePath,
  language,
  style,
  voice,
  reviewScript = false
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
       GEMINI
    ----------------------------------------- */

    const scenePlan =
      await generateScenePlan(
        jobId,
        moviePath,
        duration,
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

    if (reviewScript) {
      updateJob(jobId, {
        status: "script_ready",
        stage: "Script",
        progress: 62,
        message: "Script ရပါပြီ။ ကြည့်ပြီး လိုသလို ပြင်ပါ၊ ပြီးရင် Video ထုတ်ပါ။",
        scenes: scenePlan.scenes.map((scene, i) => ({
          index: scene.index ?? i + 1,
          start: scene.start,
          end: scene.end,
          narration: scene.narration
        }))
      });
      console.log(`[JOB ${jobId}] SCRIPT READY`);
      return;
    }

    await finishRecap(jobId, moviePath, scenePlan, voice, jobFolder);

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
        recap: GEMINI_MODEL,
        tts: GEMINI_TTS_MODEL
      },

      sceneSync: true,
      sceneFPS: SCENE_FPS,
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
   SCRIPT REVIEW + VOICE PREVIEW
========================================================= */

function jobFolderFor(id) {
  if (!/^[0-9a-f-]{36}$/i.test(String(id))) return null;
  return path.join(JOB_DIR, String(id));
}

app.get("/api/script/:id", (req, res) => {
  const folder = jobFolderFor(req.params.id);
  const planPath = folder && path.join(folder, "scene-plan.json");
  if (!planPath || !fs.existsSync(planPath)) return res.status(404).json({ error: "Script မတွေ့ပါ။" });
  const plan = JSON.parse(fs.readFileSync(planPath, "utf8"));
  return res.json({
    scenes: plan.scenes.map((scene, i) => ({
      index: scene.index ?? i + 1,
      start: scene.start,
      end: scene.end,
      narration: scene.narration
    }))
  });
});

app.post("/api/render/:id", async (req, res) => {
  try {
    const folder = jobFolderFor(req.params.id);
    const job = await getJob(req.params.id);
    if (!folder || !job) return res.status(404).json({ error: "Job not found." });
    if (job.status === "processing") return res.status(409).json({ error: "Job က အလုပ်လုပ်နေဆဲပါ။" });

    const planPath = path.join(folder, "scene-plan.json");
    if (!fs.existsSync(planPath)) return res.status(404).json({ error: "Script မတွေ့ပါ။" });
    const plan = JSON.parse(fs.readFileSync(planPath, "utf8"));

    const edits = new Map((Array.isArray(req.body?.scenes) ? req.body.scenes : []).map(item => [Number(item.index), String(item.narration || "").trim()]));
    plan.scenes.forEach((scene, i) => {
      const key = Number(scene.index ?? i + 1);
      if (edits.has(key) && edits.get(key)) scene.narration = edits.get(key).slice(0, 2000);
    });
    if (plan.scenes.some(scene => !String(scene.narration || "").trim())) {
      return res.status(400).json({ error: "Scene တစ်ခုမှာ စာမရှိပါ။" });
    }
    fs.writeFileSync(planPath, JSON.stringify(plan, null, 2), "utf8");
    saveBurn(folder, req.body?.burn);

    const voice = String(req.body?.voice || "female");
    const engine = String(req.body?.ttsEngine || "auto").toLowerCase();
    ttsEngineByJob.set(job.id, TTS_ENGINES.includes(engine) ? engine : "auto");
    if (req.body?.style) styleByJob.set(job.id, String(req.body.style));
    voiceSpeedByJob.set(job.id, String(req.body?.voiceSpeed || "normal"));

    const movieName = fs.readdirSync(folder).find(name => /^movie\./i.test(name));
    if (!movieName) return res.status(404).json({ error: "Movie file မတွေ့တော့ပါ။ ပြန်တင်ပါ။" });

    updateJob(job.id, { status: "processing", stage: "Voice", progress: 63, message: "Script အတည်ပြုပြီး၊ အသံထုတ်ပါမယ်...", error: null });
    res.status(202).json({ success: true, jobId: job.id });

    setImmediate(() => {
      finishRecap(job.id, path.join(folder, movieName), plan, voice, folder).catch(error => {
        console.error(`[JOB ${job.id}] ERROR:`, error);
        updateJob(job.id, { status: "error", stage: "Error", progress: 0, message: error.message || "Render failed.", error: error.message || "Render failed." });
      });
    });
  } catch (error) {
    console.error("RENDER ERROR:", error);
    return res.status(500).json({ error: "Render မစနိုင်ပါ။" });
  }
});

const previewCache = new Map();

function saveBurn(folder, raw) {
  if (raw === undefined || raw === null || raw === "") return;
  const burn = normalizeBurn(raw);
  if (!burn) return;
  fs.writeFileSync(path.join(folder, "burn.json"), JSON.stringify(burn), "utf8");
}

app.post("/api/tts-preview", async (req, res) => {
  const text = String(req.body?.text || "").trim().slice(0, 220);
  if (!text) return res.status(400).json({ error: "စာမပါပါ။" });
  const voice = String(req.body?.voice || "female");
  const engine = String(req.body?.ttsEngine || "auto").toLowerCase();
  const cacheKey = `${engine}|${voice}|${text}`;
  const cached = previewCache.get(cacheKey);
  if (cached) {
    res.setHeader("Content-Type", "audio/wav");
    res.setHeader("Cache-Control", "no-store");
    return res.send(cached);
  }
  const tmp = path.join(os.tmpdir(), `ynt-preview-${crypto.randomUUID()}.wav`);
  try {
    const ai = new GoogleGenAI({ apiKey: requireEnv("GEMINI_API_KEY") });
    await generateFullTTS(ai, [{ narration: text }], tmp, voice, TTS_ENGINES.includes(engine) ? engine : "auto");
    const audioData = fs.readFileSync(tmp);
    previewCache.set(cacheKey, audioData);
    if (previewCache.size > 40) previewCache.delete(previewCache.keys().next().value);
    res.setHeader("Content-Type", "audio/wav");
    res.setHeader("Cache-Control", "no-store");
    return res.send(audioData);
  } catch (error) {
    return res.status(502).json({ error: String(error?.message || error).slice(0, 300) });
  } finally {
    try { fs.unlinkSync(tmp); } catch {}
  }
});

/* =========================================================
   SRT SUBTITLES (copy / download — never burned into the video)
========================================================= */

app.get("/api/srt/:id", (req, res) => {
  const folder = jobFolderFor(req.params.id);
  const srtPath = folder && path.join(folder, "subtitles.srt");
  if (!srtPath || !fs.existsSync(srtPath)) {
    return res.status(404).json({ error: "SRT မတွေ့ပါ။ Video ပြီးမှ ရပါမယ်။" });
  }
  res.setHeader("Content-Type", "text/plain; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");
  if (req.query.download) {
    res.setHeader("Content-Disposition", `attachment; filename="YNT-Recap-${req.params.id}.srt"`);
  }
  return res.send(fs.readFileSync(srtPath, "utf8"));
});

/* =========================================================
   SRT TOOL — any video in, SRT out (separate from the recap flow)
========================================================= */

const SRT_LANGS = { auto: "", en: "en", my: "my", th: "th", zh: "zh", ja: "ja", ko: "ko", hi: "hi", id: "id", vi: "vi" };

// Whisper segments can be long; split them into readable subtitle cues.
function segmentsToCues(segments, maxChars = 42) {
  const cues = [];
  for (const seg of segments) {
    const text = String(seg.text || "").replace(/\s+/g, " ").trim();
    if (!text) continue;
    const start = Number(seg.start) || 0;
    const end = Math.max(start + 0.3, Number(seg.end) || start + 0.3);
    const parts = text.length > maxChars ? splitNarration(text, maxChars) : [text];
    const total = parts.reduce((n, part) => n + part.length, 0) || 1;
    let t = start;
    parts.forEach((part, i) => {
      const dur = i === parts.length - 1 ? end - t : (end - start) * part.length / total;
      cues.push({ start: t, end: t + Math.max(0.2, dur), text: part });
      t += dur;
    });
  }
  return cues;
}

// Translate cue texts to Myanmar in batches; the model must return the same count.
async function translateCuesToMyanmar(ai, cues, jobId) {
  const BATCH = 40;
  const out = cues.map(cue => ({ ...cue }));
  for (let from = 0; from < cues.length; from += BATCH) {
    updateJob(jobId, {
      stage: "Translate",
      progress: 60 + Math.round((from / cues.length) * 32),
      message: `မြန်မာလို ဘာသာပြန်နေပါတယ်... ${Math.min(from + BATCH, cues.length)} / ${cues.length}`
    });
    const batch = cues.slice(from, from + BATCH).map(cue => cue.text);
    const prompt =
      "Translate each subtitle line below into natural, spoken Burmese (Myanmar, Unicode). " +
      "Keep every line short enough to read on screen, keep names as they sound, do not merge or skip lines. " +
      `Return ONLY a JSON array of exactly ${batch.length} strings in the same order.\n\n` +
      JSON.stringify(batch);
    try {
      const response = await callWithModelFallback("SRT Translate", model =>
        ai.models.generateContent({
          model,
          contents: [{ role: "user", parts: [{ text: prompt }] }],
          config: { responseMimeType: "application/json" }
        })
      );
      const raw = String(response?.text || "").trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
      const arr = JSON.parse(raw);
      if (Array.isArray(arr) && arr.length === batch.length) {
        arr.forEach((text, i) => {
          const clean = String(text || "").trim();
          if (clean) out[from + i].text = clean;
        });
      } else {
        console.warn("[SRT] translation count mismatch, keeping original for this batch");
      }
    } catch (error) {
      console.warn("[SRT] translate batch failed:", String(error?.message || error).slice(0, 160));
      // Quota is gone: stop sending the remaining batches, keep original text for them.
      if (isQuotaMessage(error?.message || error)) break;
    }
  }
  return out;
}

async function processSrtJob(jobId, moviePath, language, translate) {
  try {
    const folder = path.dirname(moviePath);
    updateJob(jobId, { status: "processing", stage: "Audio", progress: 8, message: "Video ထဲက အသံကို ထုတ်နေပါတယ်..." });

    const duration = await getVideoDuration(moviePath);
    const audioPath = path.join(folder, "audio.mp3");
    try {
      await extractAudio(moviePath, audioPath);
    } catch {
      throw new Error("ဒီ Video ထဲမှာ အသံ မတွေ့ပါ။");
    }
    const chunks = await splitAudio(audioPath, path.join(folder, "chunks"));

    const transcript = await transcribeMovie(jobId, chunks, SRT_LANGS[language] || "");
    let cues = segmentsToCues(transcript.segments);
    if (!cues.length) throw new Error("စကားပြောသံ မတွေ့ပါ။ (အသံ မရှင်းလို့ ဖြစ်နိုင်ပါတယ်)");

    if (translate) {
      const ai = new GoogleGenAI({ apiKey: requireEnv("GEMINI_API_KEY") });
      cues = await translateCuesToMyanmar(ai, cues, jobId);
    }

    fs.writeFileSync(path.join(folder, "subtitles.srt"), buildSrt(cues), "utf8");
    updateJob(jobId, {
      status: "completed",
      stage: "Ready",
      progress: 100,
      duration,
      message: "SRT ပြီးပါပြီ။",
      output: { srt: { cues: cues.length, url: `/api/srt/${jobId}` } }
    });
    console.log(`[SRT JOB ${jobId}] ${cues.length} cues`);
  } catch (error) {
    console.error(`[SRT JOB ${jobId}] FAILED:`, error);
    updateJob(jobId, { status: "failed", stage: "Failed", message: "SRT မထုတ်နိုင်ပါ။", error: String(error?.message || error).slice(0, 400) });
  }
}

app.post("/api/srt-tool", upload.single("movie"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "Video ဖိုင် ရွေးပါ။" });
    const job = await createJob();
    const folder = path.join(JOB_DIR, job.id);
    fs.mkdirSync(folder, { recursive: true });
    const extension = path.extname(req.file.originalname || "") || ".mp4";
    const moviePath = path.join(folder, `movie${extension}`);
    fs.renameSync(req.file.path, moviePath);

    const language = String(req.body?.language || "auto").toLowerCase();
    const translate = String(req.body?.translate || "") === "1" && language !== "my";
    updateJob(job.id, { originalFilename: req.file.originalname || "video", status: "processing", stage: "Upload", progress: 5, message: "Video လက်ခံပြီးပါပြီ..." });

    processSrtJob(job.id, moviePath, language in SRT_LANGS ? language : "auto", translate);
    return res.json({ success: true, jobId: job.id });
  } catch (error) {
    return res.status(500).json({ error: String(error?.message || error).slice(0, 300) });
  }
});

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

      saveBurn(jobFolder, req.body?.burn);

      const language =
        req.body?.language ||
        "my";

      const style =
        req.body?.style ||
        "natural";

      const voice =
        req.body?.voice ||
        "female";

      const reviewScript = String(req.body?.reviewScript || "") === "1";
      const requestedEngine = String(req.body?.ttsEngine || "auto").toLowerCase();
      ttsEngineByJob.set(job.id, TTS_ENGINES.includes(requestedEngine) ? requestedEngine : "auto");
      styleByJob.set(job.id, style);
      voiceSpeedByJob.set(job.id, String(req.body?.voiceSpeed || "normal"));


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
          voice,
          reviewScript
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
   THUMBNAIL STUDIO (AI frame picking + AI poster polish)
========================================================= */

const GEMINI_IMAGE_MODELS = (process.env.GEMINI_IMAGE_MODELS || process.env.GEMINI_IMAGE_MODEL || "gemini-2.5-flash-image")
  .split(",").map(name => name.trim()).filter(Boolean);

function parseDataUrl(value, maxBytes = 6 * 1024 * 1024) {
  const m = /^data:(image\/(?:jpeg|png|webp));base64,([A-Za-z0-9+/=]+)$/.exec(String(value || ""));
  if (!m) return null;
  if (m[2].length * 0.75 > maxBytes) return null;
  return { mimeType: m[1], data: m[2] };
}

function extractJson(text) {
  const raw = String(text || "").trim().replace(/^```(?:json)?/i, "").replace(/```$/, "").trim();
  try { return JSON.parse(raw); } catch {}
  const a = raw.indexOf("{"), b = raw.lastIndexOf("}");
  if (a >= 0 && b > a) { try { return JSON.parse(raw.slice(a, b + 1)); } catch {} }
  return null;
}

// Looks at candidate frames and returns the 3 best "main character / key moment"
// frames plus ready-made Burmese title ideas.
app.post("/api/thumbnail/analyze", async (req, res) => {
  try {
    const frames = (Array.isArray(req.body?.frames) ? req.body.frames : [])
      .slice(0, 16).map(v => parseDataUrl(v));
    if (frames.length < 1 || frames.some(f => !f)) {
      return res.status(400).json({ error: "Frame ပုံများ မမှန်ပါ။" });
    }
    const context = String(req.body?.context || "").replace(/\s+/g, " ").slice(0, 1800);
    const hint = String(req.body?.titleHint || "").slice(0, 80);
    const ai = new GoogleGenAI({ apiKey: requireEnv("GEMINI_API_KEY") });

    const prompt = `You are a viral YouTube/TikTok movie-recap thumbnail editor.
You get ${frames.length} candidate frames from one movie, numbered 0 to ${frames.length - 1} in the order given.
${context ? `Story context (may be Burmese): ${context}\n` : ""}${hint ? `Current working title: ${hint}\n` : ""}
Tasks:
1. Identify the main characters/locations that matter most to the story.
2. Choose exactly ${Math.min(3, frames.length)} DIFFERENT frames that would make the strongest thumbnail: clear sharp faces or a striking key place/moment, strong emotion or action, good lighting, not blurry, not black, not a credits/logo frame. Prefer variety (e.g. hero, villain, key location).
3. For each pick give a SHORT Burmese label (max 24 characters) naming the character or place, and a short Burmese reason.
4. Write 6 catchy Burmese thumbnail titles (each max 28 Burmese characters, curiosity or drama driven, no quotes, no emojis).
Return ONLY JSON:
{"movie":"<guess in English or empty>","picks":[{"frame":0,"label":"","reason":""}],"titles":["",""],"subtitles":["",""]}`;

    const parts = [{ text: prompt }];
    frames.forEach((f, i) => {
      parts.push({ text: `Frame ${i}:` });
      parts.push({ inlineData: { mimeType: f.mimeType, data: f.data } });
    });

    const response = await thumbCall([GEMINI_MODEL, ...GEMINI_FALLBACK_MODELS.filter(m => m !== GEMINI_MODEL)].slice(0, 3), model =>
      ai.models.generateContent({
        model,
        contents: [{ role: "user", parts }],
        config: { responseMimeType: "application/json", temperature: 0.8 }
      })
    );
    const data = extractJson(response?.text || response?.candidates?.[0]?.content?.parts?.map(p => p.text || "").join(""));
    if (!data) throw new Error("AI အဖြေကို နားမလည်နိုင်ပါ။ ပြန်စမ်းပါ။");

    const seen = new Set();
    const picks = (Array.isArray(data.picks) ? data.picks : [])
      .map(p => ({
        frame: Number.parseInt(p?.frame, 10),
        label: String(p?.label || "").slice(0, 40),
        reason: String(p?.reason || "").slice(0, 120)
      }))
      .filter(p => Number.isInteger(p.frame) && p.frame >= 0 && p.frame < frames.length && !seen.has(p.frame) && seen.add(p.frame))
      .slice(0, 3);
    const clean = list => (Array.isArray(list) ? list : []).map(t => String(t || "").trim().slice(0, 60)).filter(Boolean).slice(0, 8);
    return res.json({ success: true, movie: String(data.movie || "").slice(0, 80), picks, titles: clean(data.titles), subtitles: clean(data.subtitles) });
  } catch (error) {
    return thumbFail(res, error);
  }
});

// Writes Burmese thumbnail titles that follow the actual story (script / dialogue / summary).
app.post("/api/thumbnail/titles", async (req, res) => {
  try {
    const story = String(req.body?.story || "").replace(/\s+/g, " ").trim().slice(0, 7000);
    const frames = (Array.isArray(req.body?.frames) ? req.body.frames : []).slice(0, 8).map(v => parseDataUrl(v)).filter(Boolean);
    if (!story && !frames.length) return res.status(400).json({ error: "ဇာတ်လမ်းစာသား မရှိပါ။" });
    const ai = new GoogleGenAI({ apiKey: requireEnv("GEMINI_API_KEY") });
    const prompt = `You write viral Burmese (Myanmar language) YouTube/TikTok movie-recap thumbnail titles.
${story ? `Here is the movie's story (script, dialogue or summary, may be Burmese or another language):\n"""${story}"""` : "No story text is available; infer the story from the attached frames."}

Write 8 DIFFERENT Burmese titles that clearly match THIS story's main characters, conflict and twist:
- 2 curiosity/question style, 2 dramatic/emotional, 2 short punchy (max 14 Burmese characters), 2 that name the main character or situation.
- Each max 30 Burmese characters, natural spoken Burmese, easy to read on a thumbnail.
- Hook the viewer but do NOT reveal the final ending. Do not invent facts that are not in the story.
- No quotes, no emojis, no hashtags.
Also write 4 short subtitles (max 20 characters) like "Movie Recap မြန်မာ", genre, or "အပိုင်းတစ်".
Return ONLY JSON: {"titles":["",""],"subtitles":["",""]}`;
    const parts = [{ text: prompt }];
    frames.forEach(f => parts.push({ inlineData: { mimeType: f.mimeType, data: f.data } }));
    const response = await thumbCall([GEMINI_MODEL, ...GEMINI_FALLBACK_MODELS.filter(m => m !== GEMINI_MODEL)].slice(0, 3), model =>
      ai.models.generateContent({ model, contents: [{ role: "user", parts }], config: { responseMimeType: "application/json", temperature: 0.9 } })
    );
    const data = extractJson(response?.text || response?.candidates?.[0]?.content?.parts?.map(p => p.text || "").join(""));
    if (!data) throw new Error("AI အဖြေကို နားမလည်နိုင်ပါ။ ပြန်စမ်းပါ။");
    const clean = list => (Array.isArray(list) ? list : []).map(t => String(t || "").trim().slice(0, 60)).filter(Boolean).slice(0, 10);
    return res.json({ success: true, titles: clean(data.titles), subtitles: clean(data.subtitles) });
  } catch (error) {
    return thumbFail(res, error);
  }
});

const THUMB_MOODS = {
  cinematic: "epic cinematic blockbuster look, teal and orange grading, dramatic rim light, volumetric haze",
  fire: "explosive action look, fiery orange and red glow, sparks and embers, intense contrast",
  neon: "vivid neon cyberpunk look, magenta and cyan glow, glossy reflections, night city energy",
  dark: "dark thriller / horror look, deep shadows, cold desaturated blue-green grading, eerie fog",
  gold: "luxurious golden-hour look, warm glowing highlights, rich premium color grading",
  fantasy: "magical fantasy look, glowing particles, rich purple and gold light, dreamy depth"
};

// Takes the composed thumbnail background (no text) and returns a polished poster-quality version.
app.post("/api/thumbnail/ai", async (req, res) => {
  try {
    const img = parseDataUrl(req.body?.image, 8 * 1024 * 1024);
    if (!img) return res.status(400).json({ error: "ပုံ မမှန်ပါ။" });
    const ratio = req.body?.ratio === "9:16" ? "9:16" : "16:9";
    const mood = THUMB_MOODS[String(req.body?.mood || "")] || THUMB_MOODS.cinematic;
    const textPos = ["top", "middle", "bottom"].includes(req.body?.textPos) ? req.body.textPos : "bottom";
    const title = String(req.body?.title || "").slice(0, 80);
    const ai = new GoogleGenAI({ apiKey: requireEnv("GEMINI_API_KEY") });

    const prompt = `Turn the provided image into a stunning, click-worthy ${ratio === "9:16" ? "9:16 vertical" : "16:9 landscape"} movie-recap thumbnail background.
Style: ${mood}.
Rules:
- Keep every person/character EXACTLY recognizable: same faces, hair, clothes, count and positions. If the image is a collage, keep every panel's subject and blend the seams into one seamless, dramatic composition.
- Make the main subjects pop: bigger contrast, rim lighting, rich saturated color, sharp detail, cinematic depth of field, subtle glow.
- Improve the background and atmosphere (light rays, smoke, sparks, bokeh) to match the story${title ? ` titled "${title}"` : ""}.
- Keep the ${textPos} area of the image a bit darker and less busy so a big title can be overlaid there.
- Output exactly ${ratio} aspect ratio.
- REMOVE any burned-in subtitles, captions, watermarks or text overlays that are in the source image and rebuild that area naturally.
- ABSOLUTELY NO text, letters, captions, subtitles, logos, watermarks or UI.`;

    const response = await thumbCall(GEMINI_IMAGE_MODELS, model =>
      ai.models.generateContent({
        model,
        contents: [{ role: "user", parts: [{ text: prompt }, { inlineData: { mimeType: img.mimeType, data: img.data } }] }],
        config: { responseModalities: ["IMAGE", "TEXT"], imageConfig: { aspectRatio: ratio } }
      })
    );
    const parts = response?.candidates?.[0]?.content?.parts || [];
    const out = parts.find(p => p.inlineData?.data);
    if (!out) throw new Error("AI က ပုံမထုတ်ပေးနိုင်ပါ။ နောက်တစ်ကြိမ် ပြန်နှိပ်ကြည့်ပါ။");
    return res.json({ success: true, image: `data:${out.inlineData.mimeType || "image/png"};base64,${out.inlineData.data}` });
  } catch (error) {
    return thumbFail(res, error);
  }
});

const isQuotaError = e => /429|quota|rate limit|RESOURCE_EXHAUSTED/i.test(String(e?.message || e));

// One request per model, never retried on quota errors (so a full quota is not hammered).
async function thumbCall(models, makeCall) {
  let lastError = null;
  for (const model of models) {
    if (isModelCoolingDown(model)) continue;
    try {
      return await makeCall(model);
    } catch (error) {
      lastError = error;
      if (isQuotaError(error)) markModelCooldown(model, String(error?.message || error));
      if (!/429|quota|rate limit|RESOURCE_EXHAUSTED|404|not found|not supported|503|UNAVAILABLE|overloaded/i.test(String(error?.message || error))) break;
    }
  }
  if (!lastError) lastError = new Error("429 quota cooldown");
  const err = new Error(isQuotaError(lastError) ? "QUOTA: Gemini quota ပြည့်နေပါတယ်။" : String(lastError?.message || lastError || "Gemini error"));
  err.quota = isQuotaError(lastError);
  throw err;
}

const thumbFail = (res, error) =>
  res.status(error?.quota ? 429 : 502).json({ error: String(error?.message || error).slice(0, 300), quota: !!error?.quota });

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
      { method: req.method, url: req.originalUrl, contentLength: req.headers["content-length"] || null, error: err?.stack || err?.message || String(err) }
    );

    if (err instanceof multer.MulterError) {
      const status = err.code === "LIMIT_FILE_SIZE" ? 413 : 400;
      return res.status(status).json({
        error: err.code === "LIMIT_FILE_SIZE"
          ? "Video ဖိုင်အရွယ်အစား 500MB ထက် မကျော်ရပါ။"
          : err.message
      });
    }

    return res.status(500).json({
      error: err.message || "Server error."
    });
  }
);

/* =========================================================
   START SERVER
========================================================= */

async function startServer() {
  try {
    await initDatabase();
    await checkSubtitleFont();

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
          `Gemini Recap: ${GEMINI_MODEL}`
        );

        console.log(
          `Gemini TTS: ${GEMINI_TTS_MODEL}`
        );

        console.log(
          `Scene Analysis FPS: ${SCENE_FPS}`
        );

        console.log(
          `Maximum Scenes: ${MAX_SCENES}`
        );

        console.log(
          `Maximum Gemini Requests/Movie: 7`
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
