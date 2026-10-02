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
const GROQ_RECAP_MODEL = "openai/gpt-oss-120b";
const GEMINI_TTS_MODEL = process.env.GEMINI_TTS_MODEL_ID || "gemini-3.8-flash-tts";
const GEMINI_TTS_VOICE = process.env.GEMINI_TTS_VOICE || "Kore";

/* =========================================================
   SETTINGS
========================================================= */

const AUDIO_CHUNK_SECONDS = 90;
const SCENE_FPS = 2;
const VISUAL_SAMPLE_SECONDS = 5;
const MAX_VISUAL_SAMPLES = 72;
const GROQ_VISION_MODEL = process.env.GROQ_VISION_MODEL || "qwen/qwen3.8-27b";
const MAX_SCENES = 24;
const MIN_SCENE_SECONDS = 3;
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
   TIMELINE + GROQ RECAP
   ---------------------------------------------------------
   Video timing comes from Groq Whisper timestamps. Gemini
   video scene analysis is intentionally not used here, so
   the pipeline does not depend on Gemini video quota.
========================================================= */

async function extractVisualFrames(moviePath, duration, workDir) {
  const frameDir = path.join(workDir, "visual-frames");
  fs.mkdirSync(frameDir, { recursive: true });

  const sampleEvery = Math.max(
    3,
    Math.ceil(Number(duration) / MAX_VISUAL_SAMPLES)
  );
  const fps = 1 / sampleEvery;

  await runCommand("ffmpeg", [
    "-y",
    "-i", moviePath,
    "-vf", `fps=${fps.toFixed(6)},scale=768:-2:flags=lanczos`,
    "-q:v", "4",
    path.join(frameDir, "frame-%04d.jpg")
  ]);

  const files = fs.readdirSync(frameDir)
    .filter(name => /^frame-\d+\.jpg$/i.test(name))
    .sort();

  const frames = files.map((name, index) => ({
    index,
    timestamp: Math.min(duration, index * sampleEvery),
    path: path.join(frameDir, name)
  }));

  if (!frames.length) {
    throw new Error("Unable to extract visual timeline frames.");
  }

  return { frames, sampleEvery };
}

async function analyzeVisualFrames(frames, groq, jobId) {
  const results = [];

  // Groq vision accepts up to 3 images per request. We overlap batches by
  // one frame so changes at batch boundaries are not missed.
  const step = 2;
  const totalBatches = Math.max(1, Math.ceil((frames.length - 1) / step));

  for (let start = 0, batchNo = 0; start < frames.length; start += step, batchNo++) {
    const batch = frames.slice(start, Math.min(start + 3, frames.length));
    if (!batch.length) break;

    const content = [{
      type: "text",
      text: `You are analyzing consecutive movie frames for a recap editor.\n\n` +
        `The frames are in chronological order. Each image has an exact timestamp. ` +
        `For every frame, describe ONLY what is visibly happening: people, movement, ` +
        `objects, location, camera framing, and important actions. Do not invent dialogue ` +
        `or off-screen events. For each frame also estimate whether a major visual change ` +
        `has occurred since the previous frame in this batch.\n\n` +
        `Return ONLY JSON:\n` +
        `{"frames":[{"timestamp":0,"visual":"...","action":"...","importance":0.0,"change_from_previous":0.0}]}\n\n` +
        `change_from_previous: 0 means essentially the same shot/action; 1 means a clear ` +
        `new shot, location, character focus, or major action. For the first frame of the ` +
        `batch use 0 unless it is obviously a new scene compared with its timestamp context.`
    }];

    for (const frame of batch) {
      const base64 = fs.readFileSync(frame.path).toString("base64");
      content.push({
        type: "text",
        text: `FRAME ${frame.index} — TIMESTAMP ${frame.timestamp.toFixed(2)} seconds`
      });
      content.push({
        type: "image_url",
        image_url: {
          url: `data:image/jpeg;base64,${base64}`
        }
      });
    }

    try {
      const response = await groq.chat.completions.create({
        model: GROQ_VISION_MODEL,
        temperature: 0.15,
        max_completion_tokens: 1800,
        response_format: { type: "json_object" },
        messages: [{
          role: "system",
          content: "You are a precise visual timeline analyst for movie recap editing."
        }, {
          role: "user",
          content
        }]
      });

      const parsed = parseJsonObject(
        response?.choices?.[0]?.message?.content || ""
      );

      if (Array.isArray(parsed?.frames)) {
        for (const item of parsed.frames) {
          const timestamp = Number(item?.timestamp);
          if (!Number.isFinite(timestamp)) continue;
          results.push({
            timestamp,
            visual: String(item?.visual || "").trim(),
            action: String(item?.action || "").trim(),
            importance: Math.max(0, Math.min(1, Number(item?.importance) || 0)),
            change: Math.max(0, Math.min(1, Number(item?.change_from_previous) || 0))
          });
        }
      }
    } catch (error) {
      console.warn(`[VISION] batch ${batchNo + 1}/${totalBatches} failed:`, error?.message || error);
    }

    updateJob(jobId, {
      stage: "Gemini",
      progress: Math.min(52, 28 + Math.round(((batchNo + 1) / totalBatches) * 22)),
      message: `Analyzing visual timeline ${batchNo + 1}/${totalBatches}...`
    });
  }

  const byTime = new Map();
  for (const item of results) {
    const key = item.timestamp.toFixed(2);
    // Overlapping batches can analyze the same frame twice. Keep the richer one.
    const previous = byTime.get(key);
    if (!previous || (item.visual.length + item.action.length) > (previous.visual.length + previous.action.length)) {
      byTime.set(key, item);
    }
  }

  return [...byTime.values()].sort((a, b) => a.timestamp - b.timestamp);
}

function buildVisualTimelineBlocks(visualFrames, transcriptSegments, duration) {
  if (!visualFrames.length) return [];

  const points = visualFrames.map((frame, index) => ({
    ...frame,
    next: visualFrames[index + 1]?.timestamp ?? duration
  }));

  const boundaries = [0];
  const MIN_BLOCK = 8;
  const MAX_BLOCK = 22;

  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    const sinceLast = point.timestamp - boundaries[boundaries.length - 1];
    const isVisualChange = point.change >= 0.62 || point.importance >= 0.9;
    if (sinceLast >= MIN_BLOCK && isVisualChange) {
      boundaries.push(point.timestamp);
    } else if (sinceLast >= MAX_BLOCK) {
      boundaries.push(point.timestamp);
    }
  }

  if (boundaries[boundaries.length - 1] !== duration) boundaries.push(duration);

  const blocks = [];
  for (let i = 0; i < boundaries.length - 1; i++) {
    const start = boundaries[i];
    const end = boundaries[i + 1];
    if (end - start < 3) continue;

    const relevant = points.filter(p => p.timestamp >= start && p.timestamp < end);
    const visual = relevant.map(p => p.visual).filter(Boolean).join("; ");
    const actions = relevant.map(p => p.action).filter(Boolean).join("; ");

    const transcript = (transcriptSegments || [])
      .filter(seg => Number(seg.end) > start && Number(seg.start) < end)
      .map(seg => String(seg.text || "").trim())
      .filter(Boolean)
      .join(" ");

    blocks.push({
      index: blocks.length + 1,
      start: Number(start.toFixed(3)),
      end: Number(end.toFixed(3)),
      duration: Number((end - start).toFixed(3)),
      visual,
      action: actions,
      transcript
    });
  }

  return blocks;
}

function parseJsonObject(text) {
  const clean = String(text || "")
    .trim()
    .replace(/^```json\s*/i, "")
    .replace(/^```\s*/i, "")
    .replace(/\s*```$/i, "");

  try {
    return JSON.parse(clean);
  } catch {
    const first = clean.indexOf("{");
    const last = clean.lastIndexOf("}");
    if (first >= 0 && last > first) {
      return JSON.parse(clean.slice(first, last + 1));
    }
    throw new Error("Groq returned invalid recap JSON.");
  }
}

async function generateScenePlan(
  jobId,
  moviePath,
  duration,
  transcript,
  language = "my",
  style = "cinematic"
) {
  updateJob(jobId, {
    stage: "Gemini",
    progress: 25,
    message: "Building visual + speech timeline..."
  });

  const groq = new Groq({
    apiKey: requireEnv("GROQ_API_KEY")
  });

  const jobFolder = path.dirname(moviePath);
  const { frames } = await extractVisualFrames(moviePath, duration, jobFolder);
  const visualFrames = await analyzeVisualFrames(frames, groq, jobId);

  if (!visualFrames.length) {
    throw new Error("Visual analysis returned no usable frames.");
  }

  const blocks = buildVisualTimelineBlocks(
    visualFrames,
    transcript?.segments || [],
    duration
  );

  if (!blocks.length) {
    throw new Error("Unable to build visual timeline blocks.");
  }

  updateJob(jobId, {
    stage: "Gemini",
    progress: 54,
    message: `Writing recap against ${blocks.length} visual timeline sections...`
  });

  const languageInstruction = language === "en"
    ? "Write natural spoken English."
    : "Write natural spoken Myanmar (Burmese) using Burmese script.";

  const styleInstruction = {
    short: "Concise and fast-paced movie recap narration.",
    storytelling: "Smooth natural storytelling. Explain what is happening as it happens, with clear chronological flow and natural transitions.",
    detailed: "Detailed but natural movie recap narration.",
    cinematic: "Cinematic movie recap narration with clear story flow."
  }[style] || "Smooth natural storytelling with clear chronological flow.";

  const blockText = blocks.map(block =>
    `SECTION ${block.index} | ${block.start.toFixed(2)}s-${block.end.toFixed(2)}s\n` +
    `VISIBLE: ${block.visual || "No reliable visual description"}\n` +
    `ACTIONS: ${block.action || "No reliable action description"}\n` +
    `DIALOGUE/TRANSCRIPT: ${block.transcript || "No speech in this section"}`
  ).join("\n\n");

  const prompt = `
Create a professional movie recap narration that is synchronized to the ORIGINAL VIDEO TIMELINE below.

CRITICAL SYNCHRONIZATION RULES:
- Each SECTION is a real video interval. Do not move events between sections.
- Describe the visible action in the same section where it occurs.
- Do not say an action has happened before its visual section.
- Do not describe an action that is not supported by the visual evidence or transcript.
- Use the transcript only for dialogue/context; visual evidence determines WHEN an action is described.
- Every section must receive exactly one narration.
- Keep each narration short enough to finish naturally inside that section. Prefer 1-3 spoken sentences.
- Do not repeat the same event across neighboring sections unless the action genuinely continues.
- Do not invent names, dialogue, actions, locations, or motives.
- ${languageInstruction}
- Style: ${styleInstruction}

Return ONLY JSON:
{
  "scenes": [
    {"index": 1, "narration": "..."}
  ]
}

TIMELINE:
${blockText}
`;

  let parsed = { scenes: [] };
  try {
    const response = await groq.chat.completions.create({
      model: GROQ_RECAP_MODEL,
      temperature: 0.25,
      max_tokens: 9000,
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: "You are an exacting movie-recap editor. Visual timing is authoritative. Never invent unsupported events."
        },
        {
          role: "user",
          content: prompt
        }
      ]
    });
    parsed = parseJsonObject(response?.choices?.[0]?.message?.content || "");
  } catch (error) {
    console.error("[GROQ RECAP] visual-timeline JSON failed:", error?.message || error);
  }

  const generated = Array.isArray(parsed?.scenes) ? parsed.scenes : [];

  const scenes = blocks.map((block, index) => {
    const item = generated[index] || generated.find(x => Number(x?.index) === block.index);
    const narration = String(item?.narration || "").trim() || block.transcript || block.action || block.visual;

    return {
      index: block.index,
      start: block.start,
      end: block.end,
      duration: block.duration,
      visual: block.visual,
      event: block.action || block.transcript,
      narration
    };
  });

  return { scenes };
}

/* =========================================================
   GEMINI TTS
========================================================= */

function resolveGeminiTTSVoice(voice) {
  const configured = String(process.env.GEMINI_TTS_VOICE || "").trim();
  if (configured) return configured;

  // Keep the UI voice selector compatible. The Gemini TTS voice is
  // controlled server-side so the existing HTML does not need changes.
  return GEMINI_TTS_VOICE || "Kore";
}

async function generateSceneTTS(
  ai,
  scene,
  outputPath,
  voice
) {
  const actualVoice = resolveGeminiTTSVoice(voice);
  const text = String(scene.narration || "").trim();

  if (!text) {
    throw new Error(`Scene ${scene.index} has no narration.`);
  }

  const response = await ai.models.generateContent({
    model: GEMINI_TTS_MODEL,
    contents: [{
      role: "user",
      parts: [{
        text,
        speechMetadata: {
          style:
            "Natural Myanmar movie recap narration. Clear Burmese pronunciation, smooth natural pacing, cinematic and expressive, but never rushed or unnaturally slow."
        }
      }]
    }],
    config: {
      responseModalities: ["AUDIO"],
      speechConfig: {
        voiceConfig: { voice: actualVoice }
      }
    }
  });

  const base64 =
    response?.candidates?.[0]?.content?.parts?.find(
      part => part?.inlineData?.data
    )?.inlineData?.data;

  if (!base64) {
    throw new Error(`Gemini returned no audio for scene ${scene.index}.`);
  }

  const audio = Buffer.from(base64, "base64");
  fs.writeFileSync(outputPath, audio);

  if (audio.length < 100) {
    throw new Error(`Gemini returned invalid audio for scene ${scene.index}.`);
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
  const sourceDuration = await getAudioDuration(inputAudio);

  if (sourceDuration <= 0) {
    throw new Error("Generated TTS has invalid duration.");
  }

  // IMPORTANT: do not time-stretch Gemini speech. This was the reason
  // the old pipeline could sound slow, word-by-word, or get cut off.
  // We only pad shorter speech with silence. If speech is longer than
  // the original scene, the caller extends the scene instead.
  const effectiveDuration = Math.max(
    Number(targetDuration) || 0,
    sourceDuration
  );

  await runCommand(
    "ffmpeg",
    [
      "-y",
      "-i",
      inputAudio,
      "-af",
      `apad,atrim=duration=${effectiveDuration.toFixed(3)}`,
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
    targetDuration: Number(targetDuration) || 0,
    effectiveDuration
  };
}

/* =========================================================
   CUT VIDEO SCENE
========================================================= */

async function renderSceneVideo(
  moviePath,
  scene,
  outputPath,
  outputDuration = null
) {
  const sourceDuration = scene.end - scene.start;
  const finalDuration = Math.max(
    sourceDuration,
    Number(outputDuration) || sourceDuration
  );

  const extraFreeze = Math.max(0, finalDuration - sourceDuration);

  const args = [
    "-y",
    "-ss",
    scene.start.toFixed(3),
    "-i",
    moviePath,
    "-t",
    sourceDuration.toFixed(3),
    "-an",
    "-vf",
    extraFreeze > 0
      ? `tpad=stop_mode=clone:stop_duration=${extraFreeze.toFixed(3)}`
      : "null",
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
  ];

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
  const ai = new GoogleGenAI({
    apiKey: requireEnv("GEMINI_API_KEY")
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

    const audioFit = await fitAudioToScene(
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
      sceneVideo,
      audioFit.effectiveDuration
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
        tts: `Gemini ${GEMINI_TTS_MODEL}`
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
          `Scene Analysis FPS: ${SCENE_FPS}`
        );

        console.log(
          `Maximum Scenes: ${MAX_SCENES}`
        );

        console.log(
          "Gemini Video Analysis: DISABLED"
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
