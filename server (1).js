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
const GROQ_VISION_MODEL = "qwen/qwen3.8-27b";

/* =========================================================
   SETTINGS
========================================================= */

const AUDIO_CHUNK_SECONDS = 90;
const SCENE_FPS = 2;
const MAX_SCENES = 18;
const MIN_SCENE_SECONDS = 3;
const VISUAL_SAMPLES = 12;
const VISION_IMAGES_PER_REQUEST = 2;
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
   GROQ VISUAL TIMELINE + RECAP SCENE PLAN
========================================================= */

async function extractFrameAtTime(videoPath, timeSec, outputPath) {
  await runCommand("ffmpeg", [
    "-y",
    "-ss", Math.max(0, Number(timeSec)).toFixed(3),
    "-i", videoPath,
    "-frames:v", "1",
    "-vf", "scale=512:-2",
    "-q:v", "6",
    outputPath
  ]);
}

function imageDataUrl(filePath) {
  const base64 = fs.readFileSync(filePath).toString("base64");
  return `data:image/jpeg;base64,${base64}`;
}

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

async function analyzeVisualFrames(jobId, videoPath, duration, jobFolder) {
  const groq = new Groq({ apiKey: requireEnv("GROQ_API_KEY") });
  const frameDir = path.join(jobFolder, "visual-frames");
  fs.mkdirSync(frameDir, { recursive: true });

  const count = Math.max(6, Math.min(VISUAL_SAMPLES, Math.ceil(duration / 8)));
  const frames = [];

  for (let i = 0; i < count; i++) {
    const time = count === 1 ? 0 : (duration * i) / (count - 1);
    const file = path.join(frameDir, `frame-${String(i + 1).padStart(3, "0")}.jpg`);
    await extractFrameAtTime(videoPath, time, file);
    frames.push({ index: i + 1, time, path: file });
    updateJob(jobId, {
      stage: "Visual Sync",
      progress: 54 + Math.round(((i + 1) / count) * 10),
      message: `Analyzing visual timeline ${i + 1}/${count}`
    });
  }

  const observations = [];

  for (let start = 0; start < frames.length; start += VISION_IMAGES_PER_REQUEST) {
    const group = frames.slice(start, start + VISION_IMAGES_PER_REQUEST);
    const content = [{
      type: "text",
      text: `Analyze these movie frames in chronological order. Each frame has an exact timestamp in its label. Describe ONLY visible events, characters, actions, setting, camera-relevant changes, and important visual transitions. Do not invent dialogue or hidden events. Return JSON only as {"frames":[{"index":1,"time_sec":0,"description":"..."}]}.`
    }];

    for (const frame of group) {
      content.push({
        type: "text",
        text: `FRAME ${frame.index} — ${frame.time.toFixed(2)} seconds`
      });
      content.push({
        type: "image_url",
        image_url: { url: imageDataUrl(frame.path) }
      });
    }

    const response = await groq.chat.completions.create({
      model: GROQ_VISION_MODEL,
      messages: [{ role: "user", content }],
      temperature: 0.2,
      max_completion_tokens: 700,
      reasoning_effort: "none",
      response_format: { type: "json_object" }
    });

    const parsed = parseJsonObject(response?.choices?.[0]?.message?.content, { frames: [] });
    if (Array.isArray(parsed?.frames)) observations.push(...parsed.frames);
  }

  observations.sort((a, b) => Number(a.time_sec || 0) - Number(b.time_sec || 0));
  return observations;
}

function buildVisualTimelineText(observations) {
  return observations.map((item, index) =>
    `VISUAL ${index + 1}: [${Number(item.time_sec || 0).toFixed(2)}s] ${String(item.description || "").trim()}`
  ).join("\n");
}

async function generateScenePlan(jobId, moviePath, duration, transcript, language = "my", style = "cinematic") {
  const visualObservations = await analyzeVisualFrames(
    jobId,
    moviePath,
    duration,
    path.dirname(moviePath)
  );

  if (!visualObservations.length) {
    throw new Error("Groq Vision did not return a visual timeline.");
  }

  const groq = new Groq({ apiKey: requireEnv("GROQ_API_KEY") });
  const transcriptText = buildTranscriptTimeline(transcript.segments);
  const visualText = buildVisualTimelineText(visualObservations);

  let languageInstruction = language === "en"
    ? "Write natural spoken English."
    : "Write natural spoken Myanmar Burmese.";

  let styleInstruction = "Use cinematic movie recap narration with natural spoken pacing.";
  if (style === "short") styleInstruction = "Use concise, fast-paced cinematic movie recap narration.";
  if (style === "storytelling") styleInstruction = "Use smooth storytelling with suspense and emotional flow.";
  if (style === "detailed") styleInstruction = "Use detailed but natural movie recap narration.";

  updateJob(jobId, {
    stage: "Gemini",
    progress: 66,
    message: "Creating synchronized recap script from visual + speech timeline..."
  });

  const prompt = `
Create a professional synchronized movie recap scene plan.

MOVIE DURATION: ${duration.toFixed(2)} seconds
LANGUAGE: ${languageInstruction}
STYLE: ${styleInstruction}

IMPORTANT:
- Use BOTH the visual timeline and Whisper transcript.
- Visual events are authoritative for where important actions happen.
- Do not invent events, characters, dialogue, or locations.
- Create 8 to ${MAX_SCENES} scenes when the movie is long enough.
- Each scene should normally be 5 to 18 seconds.
- Scene boundaries must be chronological, non-overlapping, and cover 0 to ${duration.toFixed(2)} seconds.
- narration must be concise enough to speak naturally inside its scene without forced speed changes.
- Prefer one or two natural Burmese sentences per scene.
- Keep important story events and the ending.
- Return JSON only.

JSON:
{"scenes":[{"start_sec":0,"end_sec":8,"visual_summary":"...","event_summary":"...","narration":"..."}]}

VISUAL TIMELINE:
${visualText}

WHISPER TIMELINE:
${transcriptText}
`.trim();

  const response = await groq.chat.completions.create({
    model: GROQ_RECAP_MODEL,
    messages: [{ role: "user", content: prompt }],
    temperature: 0.35,
    max_completion_tokens: 7000,
    response_format: { type: "json_object" }
  });

  const parsed = parseJsonObject(response?.choices?.[0]?.message?.content, null);
  const scenes = normalizeScenes(parsed?.scenes, duration);

  if (!scenes.length) throw new Error("Groq recap scene plan is empty.");

  return { scenes };
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
            style: "Natural cinematic Myanmar movie recap narration. Clear pronunciation, smooth natural pacing, emotional but controlled storyteller voice. Do not rush. Do not read labels or headings."
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

async function renderSceneFromAudio(
  moviePath,
  scene,
  audioPath,
  outputPath
) {
  const videoDuration = Math.max(0.5, scene.audioEnd - scene.audioStart);
  await runCommand("ffmpeg", [
    "-y",
    "-ss", Number(scene.start).toFixed(3),
    "-i", moviePath,
    "-t", videoDuration.toFixed(3),
    "-an",
    "-vf", "fps=24,format=yuv420p",
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-crf", "23",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    outputPath
  ]);

  const sceneAudio = path.join(path.dirname(outputPath), `${path.basename(outputPath, ".mp4")}-audio.wav`);
  await runCommand("ffmpeg", [
    "-y",
    "-ss", Number(scene.audioStart).toFixed(3),
    "-i", audioPath,
    "-t", videoDuration.toFixed(3),
    "-ar", "24000",
    "-ac", "1",
    "-c:a", "pcm_s16le",
    sceneAudio
  ]);

  await runCommand("ffmpeg", [
    "-y",
    "-i", outputPath,
    "-i", sceneAudio,
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-c:v", "copy",
    "-c:a", "aac",
    "-b:a", "128k",
    "-shortest",
    "-movflags", "+faststart",
    `${outputPath}.mux.mp4`
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

function groupScenesForTTS(scenes) {
  const groups = [];
  let current = [];
  let chars = 0;

  for (const scene of scenes) {
    const n = normalizeSpeechText(scene.narration).length;
    if (current.length && chars + n > MAX_TTS_CHARS_PER_REQUEST) {
      groups.push(current);
      current = [];
      chars = 0;
    }
    current.push(scene);
    chars += n;
  }
  if (current.length) groups.push(current);
  return groups;
}

async function renderSceneSyncVideo(jobId, moviePath, scenePlan, voice, jobFolder) {
  const ai = new GoogleGenAI({ apiKey: requireEnv("GEMINI_API_KEY") });
  const groq = new Groq({ apiKey: requireEnv("GROQ_API_KEY") });
  const sceneDir = path.join(jobFolder, "scenes");
  fs.mkdirSync(sceneDir, { recursive: true });
  const groups = groupScenesForTTS(scenePlan.scenes);
  const renderedScenes = [];
  const alignedScenes = [];

  for (let g = 0; g < groups.length; g++) {
    const group = groups[g];
    const narrationText = group.map(s => s.narration.trim()).join("\n\n");
    const rawAudio = path.join(sceneDir, `tts-group-${String(g + 1).padStart(2, "0")}.wav`);

    updateJob(jobId, {
      stage: "Voice",
      progress: 70 + Math.round((g / Math.max(1, groups.length)) * 15),
      message: `Gemini TTS group ${g + 1}/${groups.length} — generating natural voice...`
    });

    await generateGeminiTTSBatch(ai, narrationText, rawAudio, voice, `TTS group ${g + 1}`);

    const audioDuration = await getAudioDuration(rawAudio);
    const whisperSegments = await transcribeGeneratedAudio(groq, rawAudio);
    const aligned = alignNarrationToAudio(group, whisperSegments, audioDuration);

    alignedScenes.push(...aligned);

    for (const scene of aligned) {
      const number = String(scene.index).padStart(3, "0");
      const sceneFinal = path.join(sceneDir, `scene-${number}-final.mp4`);
      await renderSceneFromAudio(moviePath, scene, rawAudio, sceneFinal);
      renderedScenes.push(sceneFinal);
    }
  }

  const scenePlanAligned = {
    scenes: alignedScenes.map(s => ({
      ...s,
      audioStart: Number(s.audioStart.toFixed(3)),
      audioEnd: Number(s.audioEnd.toFixed(3))
    }))
  };

  fs.writeFileSync(path.join(jobFolder, "scene-plan-aligned.json"), JSON.stringify(scenePlanAligned, null, 2), "utf8");

  updateJob(jobId, {
    stage: "FFmpeg",
    progress: 94,
    message: "Joining visually synchronized narration scenes..."
  });

  const listPath = path.join(jobFolder, "scene-list.txt");
  const outputPath = path.join(jobFolder, "YNT-One-Clips-Recap.mp4");
  await concatScenes(renderedScenes, outputPath, listPath);
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
      progress: 68,
      message:
        `${scenePlan.scenes.length} visual-synchronized scenes created.`,

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
        tts: GEMINI_TTS_MODEL,
        vision: GROQ_VISION_MODEL
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
          `Groq Vision: ${GROQ_VISION_MODEL}`
        );

        console.log(
          `Maximum Scenes: ${MAX_SCENES}`
        );

        console.log(
          `Gemini TTS batching: ${MAX_TTS_CHARS_PER_REQUEST} chars/group`
        );

        console.log(
          `Gemini Scene Analysis: DISABLED`
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
