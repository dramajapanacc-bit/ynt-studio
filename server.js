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
import fsp from "node:fs/promises";

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
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-3.8-flash";
const GEMINI_FALLBACK_MODELS = (process.env.GEMINI_FALLBACK_MODELS || "gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite")
  .split(",").map(name => name.trim()).filter(Boolean);
const GEMINI_TTS_MODEL = "gemini-3.8-flash-lite-tts";

/* =========================================================
   SETTINGS
========================================================= */

const AUDIO_CHUNK_SECONDS = 90;
const SCENE_FPS = 2;
const MAX_SCENES = 18;
const MIN_SCENE_SECONDS = 3;
const MAX_VIDEO_SIZE = 500 * 1024 * 1024;

const GEMINI_MAX_RETRIES = 4;
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
        maxBuffer: 100 * 1024 * 1024,
        timeout: 15 * 60 * 1000,
        killSignal: "SIGKILL"
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

const exhaustedModels = new Map();

async function callWithModelFallback(operationName, makeCall) {
  const models = [GEMINI_MODEL, ...GEMINI_FALLBACK_MODELS.filter(name => name !== GEMINI_MODEL)];
  let lastError = null;

  for (const model of models) {
    if ((exhaustedModels.get(model) || 0) > Date.now()) continue;

    try {
      const result = await callGeminiWithRetry(`${operationName} [${model}]`, () => makeCall(model));
      console.log(`[GEMINI] ${operationName}: ${model}`);
      return result;
    } catch (error) {
      lastError = error;
      const message = String(error?.message || error);
      if (/429|quota|rate limit|per day|404|not found|not supported|NOT_FOUND/i.test(message)) {
        if (/per day|perday|daily/i.test(message)) {
          exhaustedModels.set(model, Date.now() + 3 * 60 * 60 * 1000);
        }
        console.warn(`[GEMINI] ${model} မရပါ၊ နောက် model ကို စမ်းပါမယ်။`);
        continue;
      }
      throw error;
    }
  }

  if (lastError) throw lastError;
  exhaustedModels.clear();
  throw new Error("Gemini model အားလုံး quota ကုန်နေပါတယ်။ နာရီအနည်းငယ် စောင့်ပြီး ပြန်စမ်းပါ။");
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

      if (
        status === 429 &&
        /per day|perday|daily/i.test(String(error?.message || ""))
      ) {
        throw error;
      }

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
  duration
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
    MAX_SCENES
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

  let styleInstruction =
    "Use cinematic movie recap narration.";

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
9. Prefer scenes around 6 to 14 seconds.
10. Create approximately 8 to ${MAX_SCENES} scenes.
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
21. NARRATION LENGTH BUDGET (very important): the narration of each scene must be speakable in LESS than that scene's duration at a normal speaking pace.
    - Myanmar: about 9 to 11 characters per second of the scene (a 10 second scene is about 100 characters). Never go above 13 characters per second.
    - English: about 2 words per second.
    - Do NOT describe every small action. Summarize only the key story events and skip minor details.
    - Leave short natural pauses. It is better to say less than to rush.
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

  const response =
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
      duration
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
  voice
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
            style: "Natural cinematic movie recap narration. Clear Burmese pronunciation, smooth natural pacing, emotional but controlled storyteller voice. Speak continuously as one natural movie recap."
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
const MALE_VOICES = new Set(["male","Puck","Charon","Fenrir","Orus","Enceladus","Iapetus","Umbriel","Algenib","Rasalgethi","Alnilam","Schedar","Achird","Zubenelgenubi","Sadachbia","Sadaltager"]);
function isMaleVoice(voice) { return MALE_VOICES.has(String(voice)); }
const TTS_ENGINES = ["auto", "gemini", "azure", "google"];

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

async function generateFullTTS(ai, scenes, outputPath, voice, engine = "auto") {
  const fullText = scenes.map(scene => String(scene.narration || "").trim()).join(" ").trim();
  if (!fullText) throw new Error("No narration text was available for TTS.");

  const order = engine === "auto" ? ["gemini", "azure", "google"] : [engine];
  const errors = [];

  for (const name of order) {
    try {
      if (name === "gemini") await generateGeminiTTS(ai, scenes, outputPath, voice);
      else if (name === "azure") await azureTts(fullText, outputPath, voice);
      else if (name === "google") await googleTranslateTts(fullText, outputPath);
      else throw new Error("မသိတဲ့ voice engine");

      if (!fs.existsSync(outputPath) || fs.statSync(outputPath).size < 100) throw new Error("audio file invalid");
      console.log(`[TTS] engine=${name}`);
      return outputPath;
    } catch (error) {
      const message = String(error?.message || error).slice(0, 200);
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

const MIN_TEMPO = Number(process.env.MIN_TEMPO || 0.88);
const MAX_TEMPO = Number(process.env.MAX_TEMPO || 1.3);

async function fitAudioToScene(
  inputAudio,
  outputAudio,
  targetDuration
) {
  const sourceDuration = await getAudioDuration(inputAudio);

  if (sourceDuration <= 0) {
    throw new Error("Generated TTS has invalid duration.");
  }

  // Natural speed: never stretch the voice a lot. Short narration keeps its
  // normal speed (silence is added after it); long narration is sped up only
  // a little and the scene video is extended instead of cutting words off.
  const factor = sourceDuration / targetDuration;
  const tempo = Math.min(MAX_TEMPO, Math.max(MIN_TEMPO, factor));
  const speechDuration = sourceDuration / tempo;
  const finalDuration = Math.max(targetDuration, speechDuration);

  const filters = [];
  if (Math.abs(tempo - 1) > 0.01) filters.push(`atempo=${tempo.toFixed(4)}`);
  filters.push("apad");
  filters.push(`atrim=duration=${finalDuration.toFixed(3)}`);

  await runCommand("ffmpeg", [
    "-y", "-i", inputAudio,
    "-af", filters.join(","),
    "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le",
    outputAudio
  ]);

  return { sourceDuration, targetDuration, factor, tempo, finalDuration };
}


/* =========================================================
   CUT VIDEO SCENE — ORIGINAL VIDEO ONLY
========================================================= */


/* =========================================================
   BURN-IN HELPERS (inlined: Myanmar subtitles + blur boxes)
========================================================= */
// Burn-in helpers: blur regions + Myanmar subtitles -> ffmpeg arguments.

const FONT_FILE = 'NotoSansMyanmar-Bold.ttf';
const FONT_FAMILY = 'Noto Sans Myanmar';
const FONT_URLS = [
  'https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSansMyanmar/full/ttf/NotoSansMyanmar-Bold.ttf',
  'https://raw.githubusercontent.com/notofonts/notofonts.github.io/main/fonts/NotoSansMyanmar/full/ttf/NotoSansMyanmar-Bold.ttf',
  'https://github.com/google/fonts/raw/main/ofl/notosansmyanmar/NotoSansMyanmar%5Bwdth%2Cwght%5D.ttf'
];

// Make sure a Myanmar-capable font exists in fontsDir. Users can also drop their own
// .ttf/.otf file into the fonts folder (the first Myanmar-capable font found is used).
async function ensureFont(fontsDir) {
  await fsp.mkdir(fontsDir, { recursive: true });
  const target = path.join(fontsDir, FONT_FILE);
  try { const s = await fsp.stat(target); if (s.size > 50_000) return { ok: true, family: FONT_FAMILY }; } catch {}
  for (const url of FONT_URLS) {
    try {
      const r = await fetch(url, { redirect: 'follow' });
      if (!r.ok) continue;
      const buf = Buffer.from(await r.arrayBuffer());
      if (buf.length < 50_000) continue;
      await fsp.writeFile(target, buf);
      return { ok: true, family: FONT_FAMILY };
    } catch {}
  }
  return { ok: false, family: FONT_FAMILY };
}


const bClamp = (v, a, b) => Math.min(b, Math.max(a, v));
const bNum = (v, d) => (Number.isFinite(Number(v)) ? Number(v) : d);

function assTime(sec) {
  const cs = Math.max(0, Math.round(sec * 100));
  const h = Math.floor(cs / 360000), m = Math.floor(cs % 360000 / 6000), s = Math.floor(cs % 6000 / 100), c = cs % 100;
  return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(c).padStart(2, '0')}`;
}
// ASS colour is &HAABBGGRR
function assColor(hex, fallback = '#ffffff') {
  const h = /^#?[0-9a-f]{6}$/i.test(String(hex || '')) ? String(hex).replace('#', '') : fallback.replace('#', '');
  const r = h.slice(0, 2), g = h.slice(2, 4), b = h.slice(4, 6);
  return `&H00${b}${g}${r}`.toUpperCase();
}
function escapeAss(t) {
  return String(t).replace(/\\/g, '').replace(/[{}]/g, '').replace(/\n/g, '\\N');
}

function buildAss(cues, W, H, sub, fontFamily = FONT_FAMILY) {
  const sizePct = bClamp(bNum(sub?.sizePct, 3.6), 1.5, 12);
  const outlinePct = bClamp(bNum(sub?.outlinePct, 0.28), 0, 2);
  const fontSize = Math.round(H * sizePct / 100);
  const outline = Math.max(0, +(H * outlinePct / 100).toFixed(2));
  const px = Math.round(bClamp(bNum(sub?.x, 0.5), 0, 1) * W);
  const py = Math.round(bClamp(bNum(sub?.y, 0.8), 0, 1) * H);
  const bold = sub?.bold === false ? 0 : -1;
  const marginLR = Math.round(W * 0.06);
  const primary = assColor(sub?.color, '#ffe600');
  const outlineC = assColor(sub?.outlineColor, '#000000');
  const header = `[Script Info]
ScriptType: v4.00+
PlayResX: ${W}
PlayResY: ${H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,${fontFamily},${fontSize},${primary},${primary},${outlineC},&H80000000,${bold},0,0,0,100,100,0,0,1,${outline},0,5,${marginLR},${marginLR},10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
`;
  const lines = cues.map(c => `Dialogue: 0,${assTime(c.start)},${assTime(c.end)},Default,,0,0,0,,{\\an5\\pos(${px},${py})}${escapeAss(c.text)}`);
  return header + lines.join('\n') + '\n';
}

function escapeFilterPath(p) {
  return String(p).replace(/\\/g, '/').replace(/:/g, '\\:').replace(/'/g, "\\'").replace(/,/g, '\\,').replace(/\[/g, '\\[').replace(/\]/g, '\\]');
}

// boxes: [{x,y,w,h}] normalised 0..1 relative to the frame.
function buildFilterGraph({ W, H, boxes, strength, assPath, fontsDir }) {
  const sigma = bClamp(bNum(strength, 14), 2, 60);
  const steps = [];
  const valid = [];
  for (const b of Array.isArray(boxes) ? boxes : []) {
    let x = Math.round(bClamp(bNum(b.x, 0), 0, 1) * W), y = Math.round(bClamp(bNum(b.y, 0), 0, 1) * H);
    let w = Math.round(bClamp(bNum(b.w, 0), 0, 1) * W), h = Math.round(bClamp(bNum(b.h, 0), 0, 1) * H);
    x = bClamp(x, 0, W - 2); y = bClamp(y, 0, H - 2);
    w = bClamp(w, 2, W - x); h = bClamp(h, 2, H - y);
    w -= w % 2; h -= h % 2; x -= x % 2; y -= y % 2;
    if (w >= 2 && h >= 2) valid.push({ x, y, w, h });
  }
  let cur = 'base';
  steps.push(`[0:v]scale=${W}:${H},setsar=1[${cur}]`);
  valid.forEach((b, i) => {
    const a = `a${i}`, s = `s${i}`, bl = `b${i}`, out = `o${i}`;
    steps.push(`[${cur}]split=2[${a}][${s}]`);
    steps.push(`[${s}]crop=${b.w}:${b.h}:${b.x}:${b.y},gblur=sigma=${sigma}:steps=2[${bl}]`);
    steps.push(`[${a}][${bl}]overlay=${b.x}:${b.y}[${out}]`);
    cur = out;
  });
  if (assPath) {
    steps.push(`[${cur}]ass='${escapeFilterPath(assPath)}':fontsdir='${escapeFilterPath(fontsDir)}'[vsub]`);
    cur = 'vsub';
  }
  steps.push(`[${cur}]format=yuv420p[vout]`);
  return { graph: steps.join(';'), blurCount: valid.length };
}


// BURN-HELPERS-START
/* =========================================================
   SUBTITLE + BLUR OPTIONS (optional, chosen in the UI)
========================================================= */

const burnOptionsByJob = new Map();
const SUB_POS_Y = { top: 0.14, middle: 0.5, bottom: 0.84 };

function sanitizeBurn(raw) {
  let b = raw;
  if (typeof b === "string") {
    try { b = JSON.parse(b); } catch { b = null; }
  }
  if (!b || typeof b !== "object") return null;

  const num = (v, d, lo, hi) => {
    const n = Number(v);
    return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : d;
  };

  const sub = b.subtitle && typeof b.subtitle === "object" ? b.subtitle : {};
  const blur = b.blur && typeof b.blur === "object" ? b.blur : {};

  const boxes = (Array.isArray(blur.boxes) ? blur.boxes : [])
    .slice(0, 6)
    .map(x => ({ x: num(x?.x, 0, 0, 1), y: num(x?.y, 0, 0, 1), w: num(x?.w, 0, 0, 1), h: num(x?.h, 0, 0, 1) }))
    .filter(x => x.w > 0.02 && x.h > 0.01);

  return {
    subtitle: {
      enabled: sub.enabled === true,
      sizePct: num(sub.sizePct, 3.6, 1.5, 9),
      color: /^#?[0-9a-f]{6}$/i.test(String(sub.color || "")) ? String(sub.color) : "#ffe600",
      position: ["top", "middle", "bottom"].includes(sub.position) ? sub.position : "bottom"
    },
    blur: {
      enabled: blur.enabled === true && boxes.length > 0,
      boxes,
      strength: num(blur.strength, 14, 2, 60)
    }
  };
}

// Break an over-long Burmese phrase at syllable starts (never inside a syllable).
function breakLongPhrase(piece, maxLen) {
  const chars = Array.from(piece);
  const out = [];
  let start = 0;
  while (chars.length - start > maxLen) {
    const cut = start + maxLen;
    let k = cut;
    while (k > start + 8) {
      const c = chars[k].codePointAt(0);
      const nx = chars[k + 1] ? chars[k + 1].codePointAt(0) : 0;
      const pv = chars[k - 1].codePointAt(0);
      if (c >= 0x1000 && c <= 0x1021 && nx !== 0x103A && nx !== 0x1039 && pv !== 0x1039) break;
      k--;
    }
    if (k <= start + 8) k = cut;
    out.push(chars.slice(start, k).join(""));
    start = k;
  }
  out.push(chars.slice(start).join(""));
  return out.filter(Boolean);
}

function splitSubtitleText(text, maxLen = 30) {
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  if (!clean) return [];

  const pieces = clean.split(/(?<=[။၊!?])|\s+/u).map(x => x.trim()).filter(Boolean);
  const lines = [];
  let current = "";

  for (const piece of pieces) {
    const parts = Array.from(piece).length > maxLen ? breakLongPhrase(piece, maxLen) : [piece];
    for (const part of parts) {
      if (!current) { current = part; continue; }
      if (Array.from(current + " " + part).length <= maxLen) current += " " + part;
      else { lines.push(current); current = part; }
    }
  }
  if (current) lines.push(current);
  return lines;
}

// Cues are timed relative to the START of the scene clip (each scene is encoded separately).
function buildSceneCues(text, speechSeconds) {
  const lines = splitSubtitleText(text);
  if (!lines.length) return [];
  const total = Math.max(0.8, Number(speechSeconds) || 0);
  const weights = lines.map(line => Math.max(4, Array.from(line).length));
  const sum = weights.reduce((a, b) => a + b, 0);
  let t = 0;
  return lines.map((line, i) => {
    const d = total * weights[i] / sum;
    const cue = { start: t, end: t + d, text: line };
    t += d;
    return cue;
  });
}

async function probeVideoSize(moviePath) {
  const result = await runCommand("ffprobe", [
    "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", moviePath
  ]);
  const stream = JSON.parse(result.stdout).streams?.[0];
  if (!stream) throw new Error("Video stream not found.");

  let w = Number(stream.width);
  let h = Number(stream.height);
  const rotation = Math.abs(Number(
    stream.tags?.rotate ??
    (stream.side_data_list || []).find(d => d.rotation !== undefined)?.rotation ??
    0
  ));
  if (rotation === 90 || rotation === 270) [w, h] = [h, w];

  // Same size cap for every scene (keeps encoding light and all scenes joinable).
  const cap = w >= h ? 1280 : 720;
  let outW = Math.min(w, cap);
  outW -= outW % 2;
  let outH = Math.round((h * outW) / w);
  outH -= outH % 2;
  return { W: Math.max(2, outW), H: Math.max(2, outH) };
}
// BURN-HELPERS-END

async function renderSceneVideo(
  moviePath,
  scene,
  outputPath,
  burn = null
) {
  const duration = Math.max(0.05, scene.end - scene.start);

  const args = [
    "-y",
    "-ss", scene.start.toFixed(3),
    "-i", moviePath,
    "-t", duration.toFixed(3),
    "-an",
    "-nostats", "-loglevel", "error"
  ];

  if (burn) {
    const { graph } = buildFilterGraph(burn);
    args.push("-filter_complex", graph, "-map", "[vout]");
  } else {
    args.push("-vf", "scale='if(gt(iw,ih),min(1280,iw),min(720,iw))':-2");
  }

  args.push(
    "-threads", "2",
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-crf", "24",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    outputPath
  );

  await runCommand("ffmpeg", args);
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
  const ai = new GoogleGenAI({ apiKey: requireEnv("GEMINI_API_KEY") });
  const sceneDir = path.join(jobFolder, "scenes");
  fs.mkdirSync(sceneDir, { recursive: true });

  const fullTTS = path.join(sceneDir, "full-narration.wav");

  updateJob(jobId, { stage: "Voice", progress: 65, message: "Generating one continuous narration voice..." });

  // ONE Gemini TTS request for the entire recap.
  await generateFullTTS(ai, scenePlan.scenes, fullTTS, voice, ttsEngineByJob.get(jobId) || "auto");

  const rawSceneAudio = await splitFullTTSForScenes(fullTTS, scenePlan.scenes, sceneDir);
  const renderedScenes = [];

  // Optional Myanmar subtitles + blur boxes (chosen in the UI).
  let burnPlan = null;
  const burnOpts = burnOptionsByJob.get(jobId);
  if (burnOpts && (burnOpts.subtitle.enabled || burnOpts.blur.enabled)) {
    try {
      const size = await probeVideoSize(moviePath);
      const fontsDir = path.join(DATA_DIR, "fonts");
      let subOk = false;
      if (burnOpts.subtitle.enabled) {
        subOk = (await ensureFont(fontsDir)).ok;
        if (!subOk) console.warn(`[JOB ${jobId}] Myanmar subtitle font could not be downloaded. Subtitles skipped.`);
      }
      burnPlan = {
        ...size,
        fontsDir,
        sub: subOk ? burnOpts.subtitle : null,
        boxes: burnOpts.blur.enabled ? burnOpts.blur.boxes : [],
        strength: burnOpts.blur.strength
      };
      console.log(`[JOB ${jobId}] Burn-in: subtitles=${Boolean(burnPlan.sub)} blurBoxes=${burnPlan.boxes.length} size=${size.W}x${size.H}`);
    } catch (error) {
      console.warn(`[JOB ${jobId}] Burn-in setup failed, continuing without it:`, String(error?.message || error).slice(0, 200));
      burnPlan = null;
    }
  }

  for (let i = 0; i < scenePlan.scenes.length; i++) {
    const scene = scenePlan.scenes[i];
    const progress = 68 + Math.round((i / scenePlan.scenes.length) * 23);
    updateJob(jobId, {
      stage: "Voice", progress,
      message: `Scene ${scene.index} / ${scenePlan.scenes.length} — syncing one-shot narration...`
    });

    const number = String(scene.index).padStart(3, "0");
    const rawTTS = rawSceneAudio[i];
    const fittedTTS = path.join(sceneDir, `scene-${number}-audio.wav`);
    const sceneVideo = path.join(sceneDir, `scene-${number}-video.mp4`);
    const sceneFinal = path.join(sceneDir, `scene-${number}-final.mp4`);

    const fit = await fitAudioToScene(rawTTS, fittedTTS, scene.duration);
    const sceneForVideo = fit.finalDuration > scene.duration + 0.05
      ? { ...scene, end: scene.start + fit.finalDuration }
      : scene;
    console.log(`[SYNC] scene ${scene.index}: speech ${fit.sourceDuration.toFixed(1)}s, slot ${scene.duration.toFixed(1)}s, tempo ${fit.tempo.toFixed(2)}`);

    updateJob(jobId, {
      stage: "FFmpeg", progress: progress + 1,
      message: `Scene ${scene.index} / ${scenePlan.scenes.length} — syncing video and voice...`
    });

    let burn = null;
    if (burnPlan && (burnPlan.sub || burnPlan.boxes.length)) {
      let assPath = null;
      if (burnPlan.sub) {
        const speechSeconds = Math.min(fit.finalDuration, fit.sourceDuration / Math.max(fit.tempo, 0.5)) + 0.1;
        const cues = buildSceneCues(scene.narration, speechSeconds);
        if (cues.length) {
          assPath = path.join(sceneDir, `scene-${number}.ass`);
          fs.writeFileSync(assPath, buildAss(cues, burnPlan.W, burnPlan.H, {
            sizePct: burnPlan.sub.sizePct,
            color: burnPlan.sub.color,
            x: 0.5,
            y: SUB_POS_Y[burnPlan.sub.position] ?? 0.84,
            bold: true
          }, FONT_FAMILY), "utf8");
        }
      }
      burn = { W: burnPlan.W, H: burnPlan.H, boxes: burnPlan.boxes, strength: burnPlan.strength, assPath, fontsDir: burnPlan.fontsDir };
    }

    await renderSceneVideo(moviePath, sceneForVideo, sceneVideo, burn);
    await muxScene(sceneVideo, fittedTTS, sceneFinal);
    renderedScenes.push(sceneFinal);
  }

  updateJob(jobId, { stage: "FFmpeg", progress: 94, message: "Joining synchronized scenes..." });

  const listPath = path.join(jobFolder, "scene-list.txt");
  const outputPath = path.join(jobFolder, "YNT-One-Clips-Recap.mp4");
  await concatScenes(renderedScenes, outputPath, listPath);
  return outputPath;
}

/* =========================================================
   PROCESS JOB
========================================================= */

async function finishRecap(jobId, moviePath, scenePlan, voice, jobFolder) {
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

    const voice = String(req.body?.voice || "female");
    const engine = String(req.body?.ttsEngine || "auto").toLowerCase();
    ttsEngineByJob.set(job.id, TTS_ENGINES.includes(engine) ? engine : "auto");
    const burnOptions = sanitizeBurn(req.body?.burn);
    if (burnOptions) burnOptionsByJob.set(job.id, burnOptions);

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

      const reviewScript = String(req.body?.reviewScript || "") === "1";
      const requestedEngine = String(req.body?.ttsEngine || "auto").toLowerCase();
      ttsEngineByJob.set(job.id, TTS_ENGINES.includes(requestedEngine) ? requestedEngine : "auto");
      const burnOptions = sanitizeBurn(req.body?.burn);
      if (burnOptions) burnOptionsByJob.set(job.id, burnOptions);


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
