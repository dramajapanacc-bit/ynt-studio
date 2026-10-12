import express from "express";
import multer from "multer";
import dotenv from "dotenv";
import os from "os";
import WebSocket from "ws";
import fs from "fs";
import path from "path";
import crypto from "crypto";
import { execFile } from "child_process";
import { promisify } from "util";
import Groq from "groq-sdk";
import { GoogleGenAI } from "@google/genai";
import pg from "pg";
import { EdgeTTS } from "node-edge-tts";

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
const EDGE_TTS_DEFAULT_VOICE = "my-MM-NilarNeural";
const EDGE_TTS_VOICES = {
  female: "my-MM-NilarNeural",
  male: "my-MM-ThihaNeural",
  nilar: "my-MM-NilarNeural",
  thiha: "my-MM-ThihaNeural"
};

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

const USE_DB = Boolean(DATABASE_URL);

if (!USE_DB) {
  console.log("[JOB STORAGE] DATABASE_URL မရှိပါ - memory ထဲမှာပဲ job သိမ်းမည်။");
}

const db = !USE_DB ? null : new Pool({
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
  if (!USE_DB) return;
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

  await db.query(
    `UPDATE jobs SET status = 'error', message = $1, error = $1, updated_at = NOW()
     WHERE status NOT IN ('completed', 'error')`,
    ["Server ပြန်စတင်သွားလို့ job ရပ်သွားပါတယ်။ Start ကို ပြန်နှိပ်ပါ။"]
  );

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
  if (!USE_DB) return;
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

  if (!USE_DB) {
    return null;
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
  const value = String(process.env[name] || "")
    .replace(/^\s*(export\s+)?[A-Z_]+\s*=\s*/, "")
    .replace(/^Bearer\s+/i, "")
    .replace(/["'\s]/g, "");

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
21. Narration length must be reasonable for its scene.
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
    await callGeminiWithRetry(
      "Scene-Synchronized Movie Analysis",
      () =>
        ai.interactions.create({
          model: GEMINI_MODEL,

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

  return {
    scenes
  };
}

/* =========================================================
   MICROSOFT EDGE TTS VOICE
========================================================= */

function resolveEdgeVoice(voice) {
  const key = String(voice || "female").trim().toLowerCase();
  return EDGE_TTS_VOICES[key] || EDGE_TTS_DEFAULT_VOICE;
}

function resolveEdgeRate(rate) {
  const value = Number(rate);
  if (!Number.isFinite(value)) return 1;
  return Math.max(0.7, Math.min(1.3, value));
}

/* =========================================================
   MICROSOFT EDGE TTS
========================================================= */

const EDGE_TOKEN = "6A5AA1D4EAFF4E9FB37E23D68491D6F4";

function edgeGec() {
  let ticks = Math.floor(Date.now() / 1000) + 11644473600;
  ticks -= ticks % 300;
  ticks *= 10000000;
  return crypto.createHash("sha256").update(`${ticks}${EDGE_TOKEN}`).digest("hex").toUpperCase();
}

function xmlEscape(text) {
  return String(text)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&apos;");
}

function splitForEdge(text, maxBytes = 2500) {
  const pieces = text.split(/(?<=[။!?.\n])\s*/).filter(Boolean);
  const parts = [];
  let current = "";
  const push = () => { if (current.trim()) parts.push(current.trim()); current = ""; };
  for (const piece of pieces) {
    if (Buffer.byteLength(piece) > maxBytes) {
      push();
      let chunk = "";
      for (const ch of piece) {
        if (Buffer.byteLength(chunk + ch) > maxBytes) { parts.push(chunk); chunk = ""; }
        chunk += ch;
      }
      current = chunk;
      continue;
    }
    if (Buffer.byteLength(current + " " + piece) > maxBytes) push();
    current = current ? `${current} ${piece}` : piece;
  }
  push();
  return parts;
}

function edgeChunk(text, voice, ratePercent, version) {
  return new Promise((resolve, reject) => {
    const major = version.split(".")[0];
    const id = crypto.randomUUID().replace(/-/g, "");
    const url =
      `wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1` +
      `?TrustedClientToken=${EDGE_TOKEN}&ConnectionId=${id}` +
      `&Sec-MS-GEC=${edgeGec()}&Sec-MS-GEC-Version=1-${version}`;
    const ws = new WebSocket(url, {
      headers: {
        Pragma: "no-cache",
        "Cache-Control": "no-cache",
        Origin: "chrome-extension://jdiccldimpdaibmpdkjnbmckianbfold",
        "User-Agent": `Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/${major}.0.0.0 Safari/537.36 Edg/${major}.0.0.0`,
        "Accept-Language": "en-US,en;q=0.9",
        Cookie: `muid=${crypto.randomBytes(16).toString("hex").toUpperCase()};`
      }
    });
    const audio = [];
    let finished = false;
    const fail = error => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      try { ws.terminate(); } catch {}
      reject(error);
    };
    const timer = setTimeout(() => fail(new Error("Edge TTS timeout")), 45000);

    ws.on("open", () => {
      const stamp = new Date().toISOString();
      ws.send(
        `X-Timestamp:${stamp}\r\nContent-Type:application/json; charset=utf-8\r\nPath:speech.config\r\n\r\n` +
        `{"context":{"synthesis":{"audio":{"metadataoptions":{"sentenceBoundaryEnabled":"false","wordBoundaryEnabled":"false"},"outputFormat":"audio-24khz-48kbitrate-mono-mp3"}}}}\r\n`
      );
      const rate = `${ratePercent >= 0 ? "+" : ""}${ratePercent}%`;
      ws.send(
        `X-RequestId:${id}\r\nContent-Type:application/ssml+xml\r\nX-Timestamp:${stamp}Z\r\nPath:ssml\r\n\r\n` +
        `<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'><voice name='${voice}'><prosody pitch='+0Hz' rate='${rate}' volume='+0%'>${xmlEscape(text)}</prosody></voice></speak>`
      );
    });
    ws.on("message", (data, isBinary) => {
      if (!isBinary) {
        if (data.toString().includes("Path:turn.end")) {
          finished = true;
          clearTimeout(timer);
          try { ws.close(); } catch {}
          resolve(Buffer.concat(audio));
        }
        return;
      }
      const buf = Buffer.from(data);
      const headLen = buf.readUInt16BE(0);
      if (buf.subarray(2, 2 + headLen).toString().includes("Path:audio")) {
        audio.push(buf.subarray(2 + headLen));
      }
    });
    ws.on("unexpected-response", (req, res) => fail(new Error(`Edge TTS HTTP ${res.statusCode}`)));
    ws.on("error", fail);
    ws.on("close", () => fail(new Error("Edge TTS connection closed")));
  });
}

async function edgeTtsDirect(text, outputPath, voice, ratePercent) {
  const versions = [process.env.EDGE_CHROMIUM_VERSION, "143.0.3650.75", "130.0.2849.68"].filter(Boolean);
  const buffers = [];
  for (const part of splitForEdge(text)) {
    let result = null;
    let lastError = null;
    for (const version of versions) {
      try {
        const audio = await edgeChunk(part, voice, ratePercent, version);
        if (audio.length > 100) { result = audio; break; }
        lastError = new Error("Edge TTS empty audio");
      } catch (error) {
        lastError = error;
      }
    }
    if (!result) throw lastError || new Error("Edge TTS failed");
    buffers.push(result);
  }
  fs.writeFileSync(outputPath, Buffer.concat(buffers));
}

async function geminiTtsToMp3(text, outputPath, voiceName, ratePercent) {
  const ai = new GoogleGenAI({ apiKey: requireEnv("GEMINI_API_KEY") });
  const models = [
    process.env.GEMINI_TTS_MODEL,
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts"
  ].filter(Boolean);
  const pcmParts = [];
  let sampleRate = 24000;

  for (const part of splitForEdge(text, 1500)) {
    let pcm = null;
    let lastError = null;
    for (const model of models) {
      for (let attempt = 0; attempt < 2 && !pcm; attempt++) {
        try {
          const res = await ai.models.generateContent({
            model,
            contents: [{ role: "user", parts: [{ text: part }] }],
            config: {
              responseModalities: ["AUDIO"],
              speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName } } }
            }
          });
          const inline = res?.candidates?.[0]?.content?.parts?.find(p => p?.inlineData?.data)?.inlineData;
          if (inline?.data) {
            pcm = Buffer.from(inline.data, "base64");
            const m = String(inline.mimeType || "").match(/rate=(\d+)/);
            if (m) sampleRate = Number(m[1]);
          } else {
            lastError = new Error("Gemini TTS audio မပါလာပါ");
          }
        } catch (error) {
          lastError = error;
          await new Promise(r => setTimeout(r, 1500));
        }
      }
      if (pcm) break;
    }
    if (!pcm) throw lastError || new Error("Gemini TTS failed");
    pcmParts.push(pcm);
  }

  const pcmPath = `${outputPath}.pcm`;
  fs.writeFileSync(pcmPath, Buffer.concat(pcmParts));
  const tempo = Math.min(2, Math.max(0.5, 1 + ratePercent / 100));
  try {
    await runCommand("ffmpeg", [
      "-y", "-f", "s16le", "-ar", String(sampleRate), "-ac", "1", "-i", pcmPath,
      ...(Math.abs(tempo - 1) > 0.01 ? ["-filter:a", `atempo=${tempo.toFixed(3)}`] : []),
      "-codec:a", "libmp3lame", "-b:a", "64k", outputPath
    ]);
  } finally {
    try { fs.unlinkSync(pcmPath); } catch {}
  }
}

async function generateFullTTS(scenes, outputPath, voice, rate = 1) {
  const actualVoice = resolveEdgeVoice(voice);
  const actualRate = resolveEdgeRate(rate);
  const narrations = scenes.map(scene => String(scene.narration || "").trim());

  if (narrations.some(text => !text)) {
    const bad = scenes.find(scene => !String(scene.narration || "").trim());
    throw new Error(`Scene ${bad?.index ?? "?"} has no narration.`);
  }

  const fullText = narrations.join(" ").trim();
  if (!fullText) throw new Error("No narration text was available for TTS.");

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });

  const ratePercent = Math.round((actualRate - 1) * 100);
  const tts = new EdgeTTS({
    voice: actualVoice,
    lang: "my-MM",
    outputFormat: "audio-24khz-48kbitrate-mono-mp3",
    saveSubtitles: false,
    rate: `${ratePercent >= 0 ? "+" : ""}${ratePercent}%`,
    timeout: 30000
  });

  const geminiVoice = /Nilar/i.test(actualVoice) ? "Kore" : "Charon";
  const errors = [];
  let done = false;

  try {
    await geminiTtsToMp3(fullText, outputPath, geminiVoice, ratePercent);
    done = fs.existsSync(outputPath) && fs.statSync(outputPath).size > 100;
    if (done) console.log(`[GEMINI TTS] voice=${geminiVoice}`);
  } catch (error) {
    errors.push(`gemini: ${String(error?.message || error).slice(0, 200)}`);
    console.error("[GEMINI TTS] failed:", String(error?.message || error).slice(0, 300));
  }

  if (!done) {
    try {
      await edgeTtsDirect(fullText, outputPath, actualVoice, ratePercent);
      done = true;
    } catch (error) {
      errors.push(`edge: ${String(error?.message || error).slice(0, 120)}`);
      try {
        await tts.ttsPromise(fullText, outputPath);
        done = true;
      } catch (oldError) {
        errors.push(`edge-old: ${oldError?.message || oldError}`);
      }
    }
  }

  if (!done) {
    throw new Error(`အသံထုတ်လို့မရပါ။ ${errors.join(" | ")}`);
  }

  if (!fs.existsSync(outputPath)) {
    throw new Error("Microsoft Edge TTS did not return an audio file.");
  }

  if (fs.statSync(outputPath).size < 100) {
    throw new Error("Generated Edge TTS audio is invalid.");
  }

  console.log(`[EDGE TTS] voice=${actualVoice} rate=${actualRate}`);
  return outputPath;
}

async function splitFullTTSForScenes(fullAudio, scenes, sceneDir) {
  const totalAudioDuration = await getAudioDuration(fullAudio);
  const weights = scenes.map(scene => Math.max(1, String(scene.narration || "").trim().length));
  const totalWeight = weights.reduce((sum, value) => sum + value, 0);
  const files = [];
  let offset = 0;

  for (let i = 0; i < scenes.length; i++) {
    const scene = scenes[i];
    const remaining = Math.max(0, totalAudioDuration - offset);
    const allocated = i === scenes.length - 1 ? remaining : totalAudioDuration * (weights[i] / totalWeight);
    const duration = Math.max(0.05, Math.min(remaining, allocated));
    const number = String(scene.index).padStart(3, "0");
    const rawPath = path.join(sceneDir, `scene-${number}-raw.wav`);

    await runCommand("ffmpeg", [
      "-y", "-ss", offset.toFixed(3), "-i", fullAudio,
      "-t", duration.toFixed(3), "-vn", "-ar", "24000", "-ac", "1",
      "-c:a", "pcm_s16le", rawPath
    ]);

    files.push(rawPath);
    offset += duration;
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
   CUT VIDEO SCENE — ORIGINAL VIDEO ONLY
========================================================= */

async function renderSceneVideo(
  moviePath,
  scene,
  outputPath
) {
  const duration = Math.max(0.05, scene.end - scene.start);

  const maxHeight = Number(process.env.RENDER_MAX_HEIGHT || 720);

  await runCommand(
    "nice",
    [
      "-n",
      "10",
      "ffmpeg",
      "-y",
      "-ss",
      scene.start.toFixed(3),
      "-i",
      moviePath,
      "-t",
      duration.toFixed(3),
      "-an",
      "-vf",
      `scale=-2:'min(${maxHeight},ih)'`,
      "-c:v",
      "libx264",
      "-preset",
      "ultrafast",
      "-crf",
      "26",
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
  jobFolder,
  voiceRate = 1
) {
  const sceneDir = path.join(jobFolder, "scenes");
  fs.mkdirSync(sceneDir, { recursive: true });

  const fullTTS = path.join(sceneDir, "full-narration.mp3");

  updateJob(jobId, { stage: "Voice", progress: 65, message: "Microsoft Edge TTS နဲ့ Myanmar narration ထုတ်နေပါတယ်..." });

  // ONE continuous Edge TTS request for the entire recap. Gemini TTS is not used.
  await generateFullTTS(scenePlan.scenes, fullTTS, voice, voiceRate);

  const rawSceneAudio = await splitFullTTSForScenes(fullTTS, scenePlan.scenes, sceneDir);
  const renderedScenes = [];

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

    await fitAudioToScene(rawTTS, fittedTTS, scene.duration);

    updateJob(jobId, {
      stage: "FFmpeg", progress: progress + 1,
      message: `Scene ${scene.index} / ${scenePlan.scenes.length} — syncing video and voice...`
    });

    await renderSceneVideo(moviePath, scene, sceneVideo);
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
   YOUTUBE INGESTION + SHORTS EXPORT
========================================================= */

function isAllowedVideoUrl(value) {
  try {
    const u = new URL(value);
    return ["youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "youtube-nocookie.com", "www.youtube-nocookie.com"].includes(u.hostname.toLowerCase());
  } catch {
    return false;
  }
}

async function downloadYouTubeVideo(url, jobFolder) {
  if (!isAllowedVideoUrl(url)) {
    throw new Error("YouTube URL မမှန်ပါ။ youtube.com သို့မဟုတ် youtu.be link ကိုသာ ထည့်ပါ။");
  }
  const template = path.join(jobFolder, "source.%(ext)s");
  const ytExtra = [];
  const cookieText = process.env.YT_COOKIES_CONTENT;
  if (cookieText) {
    const cookiePath = path.join(os.tmpdir(), "yt-cookies.txt");
    fs.writeFileSync(cookiePath, cookieText);
    ytExtra.push("--cookies", cookiePath);
  }
  await runCommand("yt-dlp", [
    ...ytExtra,
    "--no-playlist",
    "--format", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
    "--merge-output-format", "mp4",
    "--max-filesize", "500M",
    "--no-warnings",
    "--output", template,
    url
  ]);
  const candidates = fs.readdirSync(jobFolder)
    .filter(name => name.startsWith("source.") && /\.(mp4|mkv|webm|mov)$/i.test(name))
    .map(name => path.join(jobFolder, name));
  if (!candidates.length) throw new Error("yt-dlp က video ဖိုင်မထုတ်ပေးနိုင်ပါ။ YouTube က server IP ကို ပိတ်ထားနိုင်ပါတယ်။ MP4 ဖိုင်ကို တိုက်ရိုက်တင်ပါ၊ ဒါမှမဟုတ် YT_COOKIES_CONTENT ထည့်ပါ။");
  candidates.sort((a,b) => fs.statSync(b).size - fs.statSync(a).size);
  const chosen = candidates[0];
  const finalPath = path.join(jobFolder, "movie.mp4");
  if (chosen !== finalPath) fs.renameSync(chosen, finalPath);
  const size = fs.statSync(finalPath).size;
  if (!size || size > MAX_VIDEO_SIZE) throw new Error("YouTube video သည် 500 MB limit ထက် ကျော်နေသည် သို့မဟုတ် ဖိုင်အလွတ်ဖြစ်နေသည်။");
  return finalPath;
}

async function renderVerticalShorts(jobId, inputPath, outputPath) {
  updateJob(jobId, { stage: "Shorts Render", progress: 94, message: "9:16 vertical Shorts ကို 1080×1920 အဖြစ် render လုပ်နေသည်..." });
  await runCommand("ffmpeg", [
    "-y", "-i", inputPath,
    "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1",
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
    "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
    outputPath
  ]);
  if (!fs.existsSync(outputPath) || fs.statSync(outputPath).size <= 0) throw new Error("9:16 Shorts MP4 ကို FFmpeg က မထုတ်နိုင်ပါ။");
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
  voice,
  voiceRate = 1,
  generateShorts = true
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
        jobFolder,
        voiceRate
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

    let shortsOutput = null;
    if (generateShorts) {
      const shortsPath = path.join(jobFolder, `YNT-Shorts-${jobId}-9x16.mp4`);
      await renderVerticalShorts(jobId, outputPath, shortsPath);
      shortsOutput = {
        path: shortsPath,
        filename: `YNT-Shorts-${jobId}-9x16.mp4`,
        size: fs.statSync(shortsPath).size,
        url: `/api/download/${jobId}/shorts`
      };
    }

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
        url: `/api/download/${jobId}`,
        shorts: shortsOutput
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

function maskKey(name) {
  try {
    const v = requireEnv(name);
    return `${v.slice(0, 4)}...${v.slice(-3)} (${v.length} chars)`;
  } catch {
    return "NOT SET";
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
        tts: "Microsoft Edge TTS"
      },

      sceneSync: true,
      sceneFPS: SCENE_FPS,
      jobStorage: USE_DB ? "PostgreSQL" : "memory",
      keys: {
        groq: maskKey("GROQ_API_KEY"),
        gemini: maskKey("GEMINI_API_KEY")
      },
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
   DOWNLOAD 9:16 SHORTS
========================================================= */
app.get("/api/download/:id/shorts", async (req, res) => {
  try {
    await waitForJobWrites(req.params.id);
    const job = await getJob(req.params.id);
    const shorts = job?.output?.shorts;
    if (!job || job.status !== "completed" || !shorts) {
      return res.status(404).json({ error: "9:16 Shorts မထုတ်ရသေးပါ။" });
    }
    if (!fs.existsSync(shorts.path)) return res.status(404).json({ error: "Shorts ဖိုင်မတွေ့ပါ။ Render storage ပျောက်သွားနိုင်ပါတယ်။" });
    return res.download(shorts.path, shorts.filename);
  } catch (error) {
    console.error("SHORTS DOWNLOAD ERROR:", error);
    return res.status(500).json({ error: "Shorts ကို download မလုပ်နိုင်ပါ။" });
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
      const youtubeUrl = String(req.body?.url || req.body?.youtubeUrl || "").trim();
      if (!req.file && !youtubeUrl) {
        return res.status(400).json({ error: "MP4 ဖိုင်တင်ပါ သို့မဟုတ် YouTube URL ထည့်ပါ။" });
      }
      if (!req.file && !isAllowedVideoUrl(youtubeUrl)) {
        return res.status(400).json({ error: "YouTube URL မမှန်ပါ။ youtube.com သို့မဟုတ် youtu.be link ကိုသာ ထည့်ပါ။" });
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

      const originalName = req.file?.originalname || (youtubeUrl ? "YouTube-video.mp4" : "movie.mp4");
      let moviePath = null;
      if (req.file) {
        const extension = path.extname(originalName) || ".mp4";
        moviePath = path.join(jobFolder, `movie${extension}`);
        fs.renameSync(req.file.path, moviePath);
      }

      const language =
        req.body?.language ||
        "my";

      const style =
        req.body?.style ||
        "cinematic";

      const voice =
        req.body?.voice ||
        "female";

      const voiceRate = Number(req.body?.voiceRate || 1);
      const generateShorts = !["false", "0", "off", "no"].includes(String(req.body?.shorts ?? "true").toLowerCase());


      updateJob(job.id, {
        originalFilename: originalName,
        status: "processing",
        stage: "Upload",
        progress: 5,
        message: youtubeUrl ? "YouTube link လက်ခံပြီး video download စတင်ပါမည်။" : "Movie uploaded. Processing started."
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

      setImmediate(async () => {
        try {
          if (!moviePath && youtubeUrl) {
            updateJob(job.id, { stage: "YouTube Download", progress: 3, message: "YouTube မှ video ကို download လုပ်နေသည်..." });
            moviePath = await downloadYouTubeVideo(youtubeUrl, jobFolder);
          }
          await processOneClip(job.id, moviePath, language, style, voice, voiceRate, generateShorts);
        } catch (error) {
          console.error(`[JOB ${job.id}] UNHANDLED ERROR:`, error);
          updateJob(job.id, { status: "error", stage: "Error", progress: 0, message: error.message || "Movie processing failed.", error: error.message || "Movie processing failed." });
        }
      });
/* old async handler removed */
/*
*/

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
          "Microsoft Edge TTS: my-MM-NilarNeural / my-MM-ThihaNeural"
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
