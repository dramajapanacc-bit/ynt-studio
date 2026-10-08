import express from "express";
import multer from "multer";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import crypto from "crypto";
import { execFile } from "child_process";
import { promisify } from "util";
import Groq from "groq-sdk";
import { EdgeTTS } from "node-edge-tts";
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
const GROQ_VISION_MODEL = process.env.GROQ_VISION_MODEL || "qwen/qwen3.8-27b";
const GROQ_SCRIPT_MODEL = process.env.GROQ_SCRIPT_MODEL || "openai/gpt-oss-20b";
const GEMINI_SCRIPT_MODEL = process.env.GEMINI_SCRIPT_MODEL || "gemini-3.8-flash";
const OPENROUTER_VISION_MODEL = process.env.OPENROUTER_VISION_MODEL || "openrouter/free";

const EDGE_TTS_DEFAULT_VOICE = "my-MM-NilarNeural";
const VOICE_PROVIDER_DEFAULT = process.env.VOICE_PROVIDER || "edge";
const VOICE_CLONE_NAME = process.env.VOICE_CLONE_NAME || "Custom Voice Clone";
const VOICE_CLONE_ENDPOINT = String(process.env.VOICE_CLONE_ENDPOINT || "").trim();
const VOICE_CLONE_API_KEY = String(process.env.VOICE_CLONE_API_KEY || "").trim();
const VOICE_CLONE_TIMEOUT_MS = Math.max(30000, Number(process.env.VOICE_CLONE_TIMEOUT_MS || 180000));
const VOICE_CLONE_MAX_REFERENCE_BYTES = 30 * 1024 * 1024;

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

const GROQ_AI_MAX_RETRIES = 1;
const GROQ_AI_INITIAL_RETRY_DELAY = 1000;

const LOCAL_SCENE_THRESHOLD = 0.30;
const VISUAL_FRAMES_PER_SCENE = 1;
const VISUAL_FRAME_WIDTH = 384;
const VISUAL_FRAME_QUALITY = 7;
const MAX_VIDEO_SPEED = 1.25;
const SMART_SYNC_TARGET_SECONDS = 4.2;
const SMART_SYNC_MAX_SECONDS = 5.6;

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
      "video/x-msvideo",
      "audio/wav",
      "audio/x-wav",
      "audio/mpeg",
      "audio/mp3",
      "audio/mp4",
      "audio/x-m4a",
      "audio/aac",
      "audio/ogg",
      "audio/webm"
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
   GROQ RETRY
========================================================= */

function getGroqErrorStatus(error) {
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
    if (Number.isFinite(number)) return number;
  }

  const message = String(error?.message || error || "");
  const match = message.match(/\b(400|401|403|408|409|429|500|502|503|504)\b/);
  return match ? Number(match[1]) : null;
}

function isRetryableGroqError(error) {
  return [408, 409, 429, 500, 502, 503, 504].includes(getGroqErrorStatus(error));
}

async function callGroqWithRetry(operationName, operation) {
  let lastError;

  for (let attempt = 0; attempt <= GROQ_AI_MAX_RETRIES; attempt++) {
    try {
      if (attempt > 0) {
        const wait = GROQ_AI_INITIAL_RETRY_DELAY * Math.pow(2, attempt - 1);
        console.log(`[GROQ AI RETRY] ${operationName} waiting ${wait}ms`);
        await sleep(wait);
      }

      console.log(`[GROQ AI] ${operationName} attempt ${attempt + 1}/${GROQ_AI_MAX_RETRIES + 1}`);
      return await operation();
    } catch (error) {
      lastError = error;
      console.error(`[GROQ AI ERROR] ${operationName}`, getGroqErrorStatus(error) || "", error?.message || error);
      if (!isRetryableGroqError(error) || attempt >= GROQ_AI_MAX_RETRIES) break;
    }
  }

  const status = getGroqErrorStatus(lastError);
  throw new Error(`Groq ${operationName} failed after ${GROQ_AI_MAX_RETRIES + 1} attempts${status ? ` (HTTP ${status})` : ""}. ${lastError?.message || "Groq API error."}`);
}

function parseAIJSON(text) {
  let clean = String(text || "").trim();
  clean = clean.replace(/^```json\s*/i, "").replace(/^```\s*/i, "").replace(/\s*```$/i, "");
  try {
    return JSON.parse(clean);
  } catch {
    const first = clean.indexOf("{");
    const last = clean.lastIndexOf("}");
    if (first >= 0 && last > first) return JSON.parse(clean.slice(first, last + 1));
    throw new Error("AI returned invalid JSON.");
  }
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
        "segment",
        "word"
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
      start: Number(segment.start || 0) + offset,
      end: Number(segment.end || 0) + offset,
      text: String(segment.text || "").trim(),
      words: Array.isArray(segment.words)
        ? segment.words.map(word => ({
            start: Number(word.start || 0) + offset,
            end: Number(word.end || word.start || 0) + offset,
            text: String(word.word || word.text || "").trim()
          })).filter(word => word.text && word.end > word.start)
        : []
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
   RECAP-EN-TO-MM SMART TIMELINE HELPERS
========================================================= */
function cleanText(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

function atomicizeDialogueSegments(segments, {
  targetDuration = SMART_SYNC_TARGET_SECONDS,
  maxDuration = SMART_SYNC_MAX_SECONDS,
  maxWords = 24
} = {}) {
  const units = [];
  const tokens = [];

  for (const [segmentIndex, segment] of (segments || []).entries()) {
    const text = cleanText(segment?.text);
    const start = Number(segment?.start);
    const end = Number(segment?.end);
    if (!text || !Number.isFinite(start) || !Number.isFinite(end) || end <= start) continue;

    const words = Array.isArray(segment?.words) ? segment.words.filter(w =>
      cleanText(w?.text) && Number.isFinite(Number(w?.start)) && Number.isFinite(Number(w?.end)) && Number(w.end) > Number(w.start)
    ) : [];

    if (words.length) {
      for (const w of words) tokens.push({
        text: cleanText(w.text),
        start: Number(w.start),
        end: Number(w.end),
        source: segmentIndex
      });
    } else {
      const parts = text.split(/\s+/).filter(Boolean);
      const span = end - start;
      parts.forEach((word, i) => {
        const a = start + span * (i / parts.length);
        const b = start + span * ((i + 1) / parts.length);
        tokens.push({ text: word, start: a, end: b, source: segmentIndex });
      });
    }
  }

  tokens.sort((a,b) => a.start - b.start || a.end - b.end);
  let current = [];
  const close = (reason) => {
    if (!current.length) return;
    units.push({
      id: units.length + 1,
      speech_start: Number(current[0].start.toFixed(3)),
      speech_end: Number(current[current.length - 1].end.toFixed(3)),
      source_text: cleanText(current.map(x => x.text).join(" ")),
      boundary: reason
    });
    current = [];
  };

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    current.push(token);
    const duration = current[current.length - 1].end - current[0].start;
    const nextGap = i + 1 < tokens.length ? Math.max(0, tokens[i + 1].start - token.end) : 999;
    const wordCount = current.length;
    const sentenceEnd = /[.!?。！？]$/.test(token.text);

    if (nextGap >= 1.0) close("pause");
    else if (sentenceEnd && duration >= 1.2) close("sentence");
    else if (duration >= targetDuration && nextGap >= 0.16) close("pause");
    else if (duration >= maxDuration || wordCount >= maxWords) close("forced_local");
  }
  close("end");
  return units;
}

function dialogueForScene(scene, units) {
  const start = Number(scene.start_sec);
  const end = Number(scene.end_sec);
  return (units || [])
    .filter(u => Number(u.speech_end) > start && Number(u.speech_start) < end)
    .map(u => `[${Number(u.speech_start).toFixed(2)}s-${Number(u.speech_end).toFixed(2)}s] ${u.source_text}`)
    .join(" ");
}

function findSuspiciousDuplicateNarration(scenes) {
  const norm = value => cleanText(value).replace(/[^\u1000-\u109F\w]+/g, "").toLowerCase();
  const bad = [];
  for (let i = 1; i < scenes.length; i++) {
    const a = norm(scenes[i - 1]?.narration);
    const b = norm(scenes[i]?.narration);
    if (a.length < 12 || b.length < 12) continue;
    const shorter = Math.min(a.length, b.length);
    let same = 0;
    for (let j = 0; j < shorter; j++) if (a[j] === b[j]) same++;
    if (same / shorter > 0.94) bad.push([i, i + 1]);
  }
  return bad;
}

/* =========================================================
   LOCAL SCENE DETECTION
========================================================= */

async function detectLocalSceneBoundaries(moviePath, duration) {
  const threshold = LOCAL_SCENE_THRESHOLD;

  try {
    const result = await runCommand("ffmpeg", [
      "-hide_banner",
      "-i", moviePath,
      "-vf", `select='gt(scene,${threshold})',showinfo`,
      "-an",
      "-f",
      "null",
      "-"
    ]);

    const combined = `${result.stdout || ""}\n${result.stderr || ""}`;
    const matches = [...combined.matchAll(/pts_time:([0-9]+(?:\.[0-9]+)?)/g)];
    const rawCuts = matches.map(m => Number(m[1])).filter(t => Number.isFinite(t) && t > 0.5 && t < duration - 0.5);
    const cuts = [];

    for (const t of rawCuts) {
      if (!cuts.length || t - cuts[cuts.length - 1] >= MIN_SCENE_SECONDS) cuts.push(t);
    }

    let selected = cuts;
    if (selected.length > MAX_SCENES - 1) {
      const stride = Math.ceil(selected.length / (MAX_SCENES - 1));
      selected = selected.filter((_t, i) => i % stride === stride - 1).slice(0, MAX_SCENES - 1);
    }

    const boundaries = [0, ...selected, duration];
    const candidates = [];

    for (let i = 0; i < boundaries.length - 1; i++) {
      const start = boundaries[i];
      const end = boundaries[i + 1];
      if (end - start < MIN_SCENE_SECONDS && candidates.length) {
        candidates[candidates.length - 1].end_sec = Number(end.toFixed(3));
      } else {
        candidates.push({
          index: candidates.length + 1,
          start_sec: Number(start.toFixed(3)),
          end_sec: Number(end.toFixed(3))
        });
      }
    }

    if (candidates.length) {
      candidates[0].start_sec = 0;
      candidates[candidates.length - 1].end_sec = Number(duration.toFixed(3));
    }

    console.log(`[LOCAL SCENE] ${candidates.length} candidate intervals detected.`);
    return candidates;
  } catch (error) {
    console.warn(`[LOCAL SCENE] Detection failed; using fallback intervals: ${error?.message || error}`);
    const count = Math.min(MAX_SCENES, Math.max(1, Math.ceil(duration / 12)));
    const step = duration / count;
    return Array.from({ length: count }, (_v, i) => ({
      index: i + 1,
      start_sec: Number((i * step).toFixed(3)),
      end_sec: Number(((i + 1) * step).toFixed(3))
    }));
  }
}

/* =========================================================
   REPRESENTATIVE FRAME EXTRACTION
========================================================= */

async function extractVisualFrames(moviePath, scenes, jobFolder) {
  const framesDir = path.join(jobFolder, "visual-frames");
  fs.mkdirSync(framesDir, { recursive: true });
  const results = [];

  for (const scene of scenes) {
    const sceneDuration = Math.max(0.1, scene.end_sec - scene.start_sec);
    const offset = scene.start_sec + sceneDuration * 0.5;
    const filename = `scene-${String(scene.index).padStart(3, "0")}.jpg`;
    const framePath = path.join(framesDir, filename);

    await runCommand("ffmpeg", [
      "-y",
      "-hide_banner",
      "-ss", offset.toFixed(3),
      "-i", moviePath,
      "-frames:v", "1",
      "-vf", `scale=${VISUAL_FRAME_WIDTH}:-2:force_original_aspect_ratio=decrease`,
      "-q:v", String(VISUAL_FRAME_QUALITY),
      framePath
    ]);

    if (!fs.existsSync(framePath)) continue;
    const data = fs.readFileSync(framePath).toString("base64");
    results.push({
      ...scene,
      frames: [{
        offset_sec: Number(offset.toFixed(3)),
        path: framePath,
        mime_type: "image/jpeg",
        data
      }]
    });
  }

  return results;
}

/* =========================================================
   GROQ VISUAL ANALYSIS
========================================================= */

async function analyzeVisualBatches(jobId, visualSceneData, groq) {
  const evidence = new Map();
  const batchSize = 3;

  async function openRouterBatch(batch, rangeLabel) {
    const apiKey = String(process.env.OPENROUTER_API_KEY || "").trim();
    if (!apiKey) throw new Error("OPENROUTER_API_KEY is not configured.");

    const content = [{
      type: "text",
      text: `Analyze these movie frames in chronological order. Return ONLY JSON: {"scenes":[{"index":1,"visual_summary":"...","important_event":"..."}]}. Use only visible evidence. Focus on characters, actions, setting, emotion, objects, important story events and reveals. Do not invent dialogue or unseen events. Batch: ${rangeLabel}`
    }];
    for (const scene of batch) {
      content.push({ type: "text", text: `SCENE ${scene.index} | ${scene.start_sec.toFixed(2)}s-${scene.end_sec.toFixed(2)}s` });
      const frame = scene.frames?.[0];
      if (frame) content.push({ type: "image_url", image_url: { url: `data:${frame.mime_type};base64,${frame.data}` } });
    }

    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json",
        "HTTP-Referer": process.env.PUBLIC_APP_URL || "https://ynt-studio.onrender.com",
        "X-Title": "YNT Studio Movie Recap"
      },
      body: JSON.stringify({
        model: OPENROUTER_VISION_MODEL,
        messages: [{ role: "user", content }],
        temperature: 0.2,
        max_tokens: 500
      })
    });
    const raw = await response.text();
    if (!response.ok) throw new Error(`OpenRouter HTTP ${response.status}: ${raw.slice(0, 1200)}`);
    const data = JSON.parse(raw);
    return parseAIJSON(data?.choices?.[0]?.message?.content || "");
  }

  async function groqBatch(batch) {
    const content = [{
      type: "text",
      text: `Analyze these movie scene frames in chronological order. Return ONLY JSON in this exact shape: {"scenes":[{"index":1,"visual_summary":"...","important_event":"..."}]}. Do not invent anything outside the visible evidence. Focus on people, actions, setting, objects, emotions, and important story events.`
    }];
    for (const scene of batch) {
      content.push({ type: "text", text: `SCENE ${scene.index} | ${scene.start_sec.toFixed(2)}s-${scene.end_sec.toFixed(2)}s` });
      const frame = scene.frames?.[0];
      if (frame) content.push({ type: "image_url", image_url: { url: `data:${frame.mime_type};base64,${frame.data}` } });
    }
    const response = await callGroqWithRetry(`Visual Analysis ${batch[0].index}-${batch[batch.length - 1].index}`, () =>
      groq.chat.completions.create({
        model: GROQ_VISION_MODEL,
        messages: [{ role: "user", content }],
        temperature: 0.2,
        max_completion_tokens: 500
      })
    );
    return parseAIJSON(response?.choices?.[0]?.message?.content || "");
  }

  for (let start = 0; start < visualSceneData.length; start += batchSize) {
    const batch = visualSceneData.slice(start, start + batchSize);
    const rangeLabel = `${batch[0]?.index || start + 1}-${batch[batch.length - 1]?.index || start + batch.length}`;
    updateJob(jobId, {
      stage: "Visual Analysis",
      progress: 54 + Math.round((start / Math.max(1, visualSceneData.length)) * 6),
      message: `Free Vision: scenes ${rangeLabel} / ${visualSceneData.length}...`
    });

    let parsed = null;
    try {
      if (process.env.OPENROUTER_API_KEY) {
        parsed = await openRouterBatch(batch, rangeLabel);
        console.log(`[VISION] OpenRouter free router → ${rangeLabel}`);
      } else {
        parsed = await groqBatch(batch);
        console.log(`[VISION] Groq fallback → ${rangeLabel}`);
      }
    } catch (primaryError) {
      console.warn(`[VISION] Primary failed for ${rangeLabel}: ${primaryError?.message || primaryError}`);
      try {
        if (process.env.OPENROUTER_API_KEY) {
          parsed = await groqBatch(batch);
          console.log(`[VISION] Groq fallback → ${rangeLabel}`);
        }
      } catch (fallbackError) {
        console.warn(`[VISION] Groq fallback failed for ${rangeLabel}: ${fallbackError?.message || fallbackError}`);
        // Important: a vision quota failure must not kill the whole recap. Whisper
        // transcript remains valid evidence for the script writer.
      }
    }

    for (const item of Array.isArray(parsed?.scenes) ? parsed.scenes : []) {
      const index = Number(item?.index);
      if (!Number.isFinite(index)) continue;
      evidence.set(index, {
        visual_summary: cleanText(item?.visual_summary),
        important_event: cleanText(item?.important_event)
      });
    }
  }
  return evidence;
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
      "Groq did not return a scene list."
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
      "No usable scenes were returned by Groq."
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
   GROQ SCENE PLAN
========================================================= */

async function generateGeminiScript(prompt) {
  const key = String(process.env.GEMINI_API_KEY || "").trim();
  if (!key) throw new Error("GEMINI_API_KEY is not configured.");
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(GEMINI_SCRIPT_MODEL)}:generateContent?key=${encodeURIComponent(key)}`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: { temperature: 0.35, maxOutputTokens: 1800 }
    })
  });
  const raw = await response.text();
  if (!response.ok) throw new Error(`Gemini HTTP ${response.status}: ${raw.slice(0, 1400)}`);
  const data = JSON.parse(raw);
  const text = (data?.candidates?.[0]?.content?.parts || []).map(p => p?.text || "").join("\n").trim();
  if (!text) throw new Error("Gemini returned an empty recap script.");
  return text;
}

async function generateGroqScript(prompt) {
  const groq = new Groq({ apiKey: requireEnv("GROQ_API_KEY") });
  // Deliberately do NOT force response_format=json_object. Some Groq models reject
  // valid natural JSON under strict validation; we parse the returned text locally.
  const response = await callGroqWithRetry("Recap Script Fallback", () =>
    groq.chat.completions.create({
      model: GROQ_SCRIPT_MODEL,
      messages: [{ role: "user", content: prompt }],
      temperature: 0.35,
      max_completion_tokens: 1800
    })
  );
  return response?.choices?.[0]?.message?.content || "";
}

async function generateScenePlan(jobId, moviePath, duration, transcript, language = "myanmar", style = "natural") {
  const groq = new Groq({ apiKey: requireEnv("GROQ_API_KEY") });

  updateJob(jobId, {
    stage: "Visual Analysis",
    progress: 53,
    message: "FFmpeg scene cuts + Whisper dialogue + free vision analysis..."
  });

  const localScenes = await detectLocalSceneBoundaries(moviePath, duration);
  const visualSceneData = await extractVisualFrames(moviePath, localScenes, path.dirname(moviePath));
  const evidence = await analyzeVisualBatches(jobId, visualSceneData, groq);
  const timeline = buildTranscriptTimeline(transcript.segments);
  const dialogueUnits = atomicizeDialogueSegments(transcript.segments);

  const candidateTimeline = localScenes.map(scene => {
    const item = evidence.get(scene.index) || {};
    const dialogue = dialogueForScene(scene, dialogueUnits);
    return `SCENE ${scene.index}: ${scene.start_sec.toFixed(2)}s - ${scene.end_sec.toFixed(2)}s\nVISUAL: ${item.visual_summary || "No visual summary available."}\nIMPORTANT EVENT: ${item.important_event || "No visual event confirmed."}\nDIALOGUE ANCHORS: ${dialogue || "No dialogue in this window."}`;
  }).join("\n\n");

  let languageInstruction = "Write natural spoken Myanmar (Burmese).";
  if (language === "en" || language === "english") languageInstruction = "Write natural spoken English.";
  let styleInstruction = "Use cinematic movie recap narration.";
  if (style === "fast" || style === "short") styleInstruction = "Use concise, fast-paced movie recap narration.";
  if (style === "suspense" || style === "storytelling") styleInstruction = "Use suspenseful storytelling with emotional flow and strong reveals.";
  if (style === "detailed") styleInstruction = "Use detailed but natural movie recap narration while keeping every scene speakable.";

  const prompt = `You are the final movie recap script writer. Gemini is used ONLY for this script step.

SOURCE EVIDENCE:
- FFmpeg supplied chronological scene boundaries.
- Groq Whisper supplied timestamped dialogue/audio transcript.
- Whisper dialogue has been atomicized into small local sync anchors (target about 4.2s, hard cap about 5.6s when possible).
- Free vision/Groq vision supplied representative-frame evidence. Missing visual evidence must NOT be invented.

CORE RULES:
1. Combine visual evidence and dialogue evidence.
2. Never invent unsupported events, characters, locations, actions, or dialogue.
3. Preserve chronology, cause/effect, important visual events, dialogue-driven events, major reveals, and the ending.
4. Keep every output scene inside its supplied scene range. Do not create timestamps outside those ranges.
5. Use enough scenes to cover the whole story arc; do not collapse a long movie into only a few seconds of narration.
6. Narration should be natural when spoken aloud, not a literal translation or formal article.
7. For Myanmar, use conversational Burmese, short natural clauses, varied sentence endings, and spoken storytelling rhythm. Avoid bookish wording, bullet-like phrasing, and repetitive endings.
8. Do not repeat every dialogue line. Summarize dialogue when it advances the story.
9. Action-only scenes without useful dialogue may be narrated only when the visual evidence shows an important story event. Do not invent speech for silent scenes.
10. Each narration should normally be 1-3 spoken sentences and reasonably fit its selected source window.
11. Preserve important beginning, middle, climax/twist, and ending events.
12. Return ONLY JSON. No Markdown.

LANGUAGE: ${languageInstruction}
STYLE: ${styleInstruction}

JSON SHAPE:
{"scenes":[{"start_sec":0,"end_sec":8,"visual_summary":"...","event_summary":"...","narration":"..."}]}

SCENE + DIALOGUE EVIDENCE:
${candidateTimeline}

FULL WHISPER TIMELINE:
${timeline}`;

  updateJob(jobId, {
    stage: "Recap",
    progress: 61,
    message: `Gemini is writing the Myanmar recap script (${GEMINI_SCRIPT_MODEL})...`
  });

  let outputText = "";
  let scriptProvider = "Gemini";
  try {
    outputText = await generateGeminiScript(prompt);
  } catch (geminiError) {
    console.warn(`[GEMINI SCRIPT] ${geminiError?.message || geminiError}`);
    updateJob(jobId, {
      stage: "Recap",
      progress: 62,
      message: "Gemini quota/error detected — switching to Groq script fallback..."
    });
    scriptProvider = "Groq fallback";
    outputText = await generateGroqScript(prompt);
  }

  if (!outputText) throw new Error(`${scriptProvider} returned an empty scene plan.`);

  let parsed;
  try {
    parsed = parseAIJSON(outputText);
  } catch (parseError) {
    if (scriptProvider === "Gemini") {
      console.warn(`[GEMINI SCRIPT] JSON parse failed — retrying once with Groq fallback.`);
      scriptProvider = "Groq fallback";
      parsed = parseAIJSON(await generateGroqScript(prompt));
    } else {
      throw parseError;
    }
  }

  const scenes = normalizeScenes(parsed.scenes, duration);
  if (!scenes.length) throw new Error(`${scriptProvider} returned no usable scenes.`);
  const duplicates = findSuspiciousDuplicateNarration(scenes);
  if (duplicates.length) console.warn(`[RECAP AUDIT] suspicious adjacent narration duplicates: ${duplicates.map(x => x.join("/")).join(", ")}`);

  console.log(`[SCENE SYNC] ${scenes.length} scenes created by ${scriptProvider}.`);
  return { scenes, provider: scriptProvider, dialogueUnits: dialogueUnits.length };
}

/* =========================================================
   EDGE TTS VOICE
========================================================= */

function resolveEdgeVoice(voice) {
  if (EDGE_TTS_VOICES[voice]) return EDGE_TTS_VOICES[voice];
  if (EDGE_TTS_VOICES[String(voice || "").toLowerCase()]) return EDGE_TTS_VOICES[String(voice || "").toLowerCase()];
  return EDGE_TTS_DEFAULT_VOICE;
}

function resolveEdgeRate(rate) {
  const numeric = Number(rate);
  const safe = Number.isFinite(numeric) ? Math.max(0.5, Math.min(2, numeric)) : 1;
  const percent = Math.round((safe - 1) * 100);
  return `${percent >= 0 ? "+" : ""}${percent}%`;
}

async function fetchWithTimeout(url, options = {}, timeoutMs = VOICE_CLONE_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function generateSceneTTS(
  scene,
  outputPath,
  voice,
  rate = 1,
  cloneAudioPath = "",
  cloneReferenceText = ""
) {
  const text = String(scene?.narration || "").trim();
  if (!text) throw new Error(`Scene ${scene?.index || "?"} has no narration text.`);

  const selectedProvider = String(
    scene?.voiceProvider || VOICE_PROVIDER_DEFAULT || "edge"
  ).toLowerCase();

  if (selectedProvider === "clone") {
    if (!VOICE_CLONE_ENDPOINT) {
      throw new Error(
        "Voice Clone endpoint မသတ်မှတ်ရသေးပါ။ VOICE_CLONE_ENDPOINT ထည့်ပါ။"
      );
    }
    if (!cloneAudioPath || !fs.existsSync(cloneAudioPath)) {
      throw new Error(
        "Voice Clone အတွက် Reference Audio မတွေ့ပါ။ အသံဖိုင်တင်ပြီး ပြန်စမ်းပါ။"
      );
    }
    if (!String(cloneReferenceText || "").trim()) {
      throw new Error(
        "Voice Clone အတွက် Reference Text လိုအပ်ပါတယ်။ Reference အသံထဲမှာ ပြောထားတဲ့စာကို အတိအကျထည့်ပါ။"
      );
    }

    const stat = fs.statSync(cloneAudioPath);
    if (stat.size > VOICE_CLONE_MAX_REFERENCE_BYTES) {
      throw new Error(
        `Reference Audio က ${(VOICE_CLONE_MAX_REFERENCE_BYTES / 1024 / 1024).toFixed(0)}MB ထက် မကျော်ရပါ။`
      );
    }

    const form = new FormData();
    form.append("text", text);
    form.append("reference_text", String(cloneReferenceText).trim());
    form.append("voice", voice || VOICE_CLONE_NAME);
    form.append("speed", String(Number(rate) || 1));

    const audioBuffer = fs.readFileSync(cloneAudioPath);
    const ext = path.extname(cloneAudioPath).toLowerCase();
    const mime =
      ext === ".wav" ? "audio/wav" :
      ext === ".mp3" ? "audio/mpeg" :
      ext === ".m4a" ? "audio/mp4" :
      ext === ".ogg" ? "audio/ogg" :
      ext === ".webm" ? "audio/webm" :
      "application/octet-stream";

    form.append(
      "audio",
      new Blob([audioBuffer], { type: mime }),
      path.basename(cloneAudioPath)
    );

    const headers = {};
    if (VOICE_CLONE_API_KEY) {
      headers.Authorization = `Bearer ${VOICE_CLONE_API_KEY}`;
    }

    console.log(
      `[VOICE CLONE] scene=${scene.index} endpoint=${VOICE_CLONE_ENDPOINT} ref=${path.basename(cloneAudioPath)}`
    );

    let response;
    try {
      response = await fetchWithTimeout(
        VOICE_CLONE_ENDPOINT,
        {
          method: "POST",
          headers,
          body: form
        }
      );
    } catch (error) {
      if (error?.name === "AbortError") {
        throw new Error(
          `Voice Clone timeout (${Math.round(VOICE_CLONE_TIMEOUT_MS / 1000)}s).`
        );
      }
      throw error;
    }

    if (!response.ok) {
      const body = await response.text().catch(() => "");
      throw new Error(
        `Voice Clone HTTP ${response.status}${body ? `: ${body.slice(0, 800)}` : ""}`
      );
    }

    const buffer = Buffer.from(await response.arrayBuffer());
    if (buffer.length < 1000) {
      throw new Error("Voice Clone returned empty/invalid audio.");
    }

    fs.writeFileSync(outputPath, buffer);
    return outputPath;
  }

  const actualVoice = resolveEdgeVoice(voice);
  const tts = new EdgeTTS({
    voice: actualVoice,
    lang: "my-MM",
    outputFormat: "audio-24khz-48kbitrate-mono-mp3",
    saveSubtitles: false,
    rate: resolveEdgeRate(rate),
    timeout: 30000
  });
  console.log(
    `[EDGE TTS] scene=${scene.index} voice=${actualVoice} rate=${resolveEdgeRate(rate)}`
  );
  await tts.ttsPromise(text, outputPath);
  if (!fs.existsSync(outputPath) || fs.statSync(outputPath).size < 100) {
    throw new Error(`Edge TTS did not create valid audio for scene ${scene.index}.`);
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

async function prepareNarrationAudio(inputAudio, outputAudio, { shortPauses = true, rate = 1 } = {}) {
  const sourceDuration = await getAudioDuration(inputAudio);
  if (!Number.isFinite(sourceDuration) || sourceDuration <= 0) {
    throw new Error("Generated TTS has invalid duration.");
  }

  const filters = [];
  // recap-en-to-mm: trim the long silence Edge TTS can place around clips,
  // while keeping a small natural sentence break.
  if (shortPauses) {
    filters.push("silenceremove=start_periods=1:start_duration=0.18:start_threshold=-48dB:stop_periods=1:stop_duration=0.22:stop_threshold=-48dB");
  }
  const safeRate = Math.max(0.5, Math.min(2, Number(rate) || 1));
  if (Math.abs(safeRate - 1) > 0.001) {
    filters.push(...buildAtempoFilters(safeRate));
  }
  filters.push("loudnorm=I=-16:LRA=11:TP=-1.5");

  await runCommand("ffmpeg", [
    "-y", "-i", inputAudio,
    "-af", filters.join(","),
    "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", outputAudio
  ]);

  const finalDuration = await getAudioDuration(outputAudio);
  return { sourceDuration: finalDuration };
}


/* =========================================================
   CUT VIDEO SCENE — ORIGINAL VIDEO ONLY
========================================================= */

async function renderSceneVideo(
  moviePath,
  scene,
  outputPath,
  targetDuration
) {
  const sourceDuration = Math.max(0.05, Number(scene.end) - Number(scene.start));
  const target = Math.max(0.05, Number(targetDuration));
  const maxSpeed = MAX_VIDEO_SPEED;
  const minTargetForSpeedCap = sourceDuration / maxSpeed;
  const effectiveDuration = Math.max(target, minTargetForSpeedCap);
  const speed = sourceDuration / effectiveDuration;

  const args = [
    "-y", "-ss", Number(scene.start).toFixed(3), "-i", moviePath, "-an"
  ];

  if (effectiveDuration > sourceDuration + 0.05) {
    const extra = effectiveDuration - sourceDuration;
    args.push("-vf", `tpad=stop_mode=clone:stop_duration=${extra.toFixed(3)}`);
  } else if (speed > 1.001) {
    args.push("-vf", `setpts=PTS/${Math.min(maxSpeed, speed).toFixed(6)}`);
  }

  args.push(
    "-t", effectiveDuration.toFixed(3),
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart", outputPath
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
  voiceRate,
  jobFolder,
  voiceProvider = "edge",
  shortPauses = true,
  cloneAudioPath = "",
  cloneReferenceText = ""
) {
  const sceneDir = path.join(jobFolder, "scenes");
  fs.mkdirSync(sceneDir, { recursive: true });

  const renderedScenes = [];
  const totalScenes = scenePlan.scenes.length;

  updateJob(jobId, {
    stage: "Voice",
    progress: 65,
    message: "Generating natural Myanmar narration scene by scene..."
  });

  for (let i = 0; i < totalScenes; i++) {
    const scene = scenePlan.scenes[i];
    const number = String(scene.index).padStart(3, "0");
    const rawTTS = path.join(
      sceneDir,
      `scene-${number}-tts.${String(voiceProvider).toLowerCase() === "clone" ? "wav" : "mp3"}`
    );
    const narrationAudio = path.join(sceneDir, `scene-${number}-audio.wav`);
    const sceneVideo = path.join(sceneDir, `scene-${number}-video.mp4`);
    const sceneFinal = path.join(sceneDir, `scene-${number}-final.mp4`);

    const voiceProgress = 65 + Math.round((i / Math.max(1, totalScenes)) * 25);
    updateJob(jobId, {
      stage: "Voice",
      progress: voiceProgress,
      message: `Myanmar Voice: scene ${scene.index} / ${totalScenes}...`
    });

    await generateSceneTTS(
      { ...scene, voiceProvider },
      rawTTS,
      voice,
      voiceRate,
      cloneAudioPath,
      cloneReferenceText
    );
    const prepared = await prepareNarrationAudio(
      rawTTS,
      narrationAudio,
      { shortPauses, rate: voiceProvider === "clone" ? voiceRate : 1 }
    );

    updateJob(jobId, {
      stage: "FFmpeg",
      progress: Math.min(93, voiceProgress + 1),
      message: `Scene ${scene.index} / ${totalScenes} — matching video to narration (${prepared.sourceDuration.toFixed(1)}s)...`
    });

    // Final scene duration follows the natural narration duration, not the
    // original movie-scene duration. This makes the recap genuinely shorter.
    await renderSceneVideo(moviePath, scene, sceneVideo, prepared.sourceDuration);
    await muxScene(sceneVideo, narrationAudio, sceneFinal);
    renderedScenes.push(sceneFinal);
  }

  updateJob(jobId, {
    stage: "FFmpeg",
    progress: 94,
    message: "Joining recap scenes..."
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
  voice,
  voiceProvider = "edge",
  shortPauses = true,
  voiceRate = 1,
  cloneAudioPath = "",
  cloneReferenceText = ""
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
       GROQ VISUAL + RECAP
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
      stage: "Recap",
      progress: 64,
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
        voiceRate,
        jobFolder,
        voiceProvider,
        shortPauses,
        cloneAudioPath,
        cloneReferenceText
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
        vision: process.env.OPENROUTER_API_KEY ? `OpenRouter/${OPENROUTER_VISION_MODEL}` : `Groq/${GROQ_VISION_MODEL}`,
        recap: `Gemini/${GEMINI_SCRIPT_MODEL} → Groq/${GROQ_SCRIPT_MODEL}`,
        tts: "Microsoft Edge TTS / VoxCPM Voice Clone"
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
  upload.fields([
    { name: "movie", maxCount: 1 },
    { name: "cloneAudio", maxCount: 1 }
  ]),

  async (req, res) => {
    try {
      const movieFile = req.files?.movie?.[0];
      const cloneAudioFile = req.files?.cloneAudio?.[0];

      if (!movieFile) {
        return res
          .status(400)
          .json({
            error:
              "Movie file is required."
          });
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

      const voiceProvider =
        req.body?.voiceProvider ||
        VOICE_PROVIDER_DEFAULT;

      const shortPauses =
        String(req.body?.shortPauses ?? "true") !== "false";

      const voiceRate =
        Number(req.body?.voiceRate || 1);

      const cloneReferenceText = String(
        req.body?.cloneReferenceText || ""
      ).trim();

      if (String(voiceProvider).toLowerCase() === "clone") {
        if (!VOICE_CLONE_ENDPOINT) {
          return res.status(400).json({
            error: "Voice Clone service မချိတ်ထားသေးပါ။ VOICE_CLONE_ENDPOINT ထည့်ပါ။"
          });
        }
        if (!cloneAudioFile) {
          return res.status(400).json({
            error: "Voice Clone ရွေးထားရင် Reference Audio တင်ပေးရပါမယ်။"
          });
        }
        if (!cloneReferenceText) {
          return res.status(400).json({
            error: "Voice Clone ရွေးထားရင် Reference Text ထည့်ပေးရပါမယ်။"
          });
        }
        if (cloneAudioFile.size > VOICE_CLONE_MAX_REFERENCE_BYTES) {
          return res.status(400).json({
            error: `Reference Audio က ${(VOICE_CLONE_MAX_REFERENCE_BYTES / 1024 / 1024).toFixed(0)}MB ထက် မကျော်ရပါ။`
          });
        }
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
        movieFile.originalname ||
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
        movieFile.path,
        moviePath
      );

      let cloneAudioPath = "";

      if (String(voiceProvider).toLowerCase() === "clone") {
        cloneAudioPath = path.join(
          jobFolder,
          `voice-reference${path.extname(cloneAudioFile.originalname || ".wav") || ".wav"}`
        );
        fs.renameSync(cloneAudioFile.path, cloneAudioPath);
      } else if (cloneAudioFile) {
        try { fs.unlinkSync(cloneAudioFile.path); } catch {}
      }

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
        `[JOB ${job.id}] Voice Rate: ${voiceRate}`
      );

      console.log(
        `[JOB ${job.id}] Voice Provider: ${voiceProvider}`
      );

      console.log(
        `[JOB ${job.id}] Short Pauses: ${shortPauses}`
      );

      console.log(
        `[JOB ${job.id}] Clone Reference: ${voiceProvider === "clone" ? path.basename(cloneAudioPath) : "not used"}`
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
          voiceProvider,
          shortPauses,
          voiceRate,
          cloneAudioPath,
          cloneReferenceText
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

      const pendingFiles = [
        ...(req.files?.movie || []),
        ...(req.files?.cloneAudio || [])
      ];
      for (const pending of pendingFiles) {
        if (pending?.path && fs.existsSync(pending.path)) {
          try { fs.unlinkSync(pending.path); } catch {}
        }
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
          `Groq Vision: ${GROQ_VISION_MODEL}`
        );

        console.log(
          `Vision: ${process.env.OPENROUTER_API_KEY ? `OpenRouter/${OPENROUTER_VISION_MODEL}` : `Groq/${GROQ_VISION_MODEL}`}`
        );

        console.log(
          `Recap Script: Gemini/${GEMINI_SCRIPT_MODEL} → Groq/${GROQ_SCRIPT_MODEL}`
        );

        console.log(
          `Scene Analysis FPS: ${SCENE_FPS}`
        );

        console.log(
          `Maximum Scenes: ${MAX_SCENES}`
        );

        console.log(
          `Gemini Script: ${process.env.GEMINI_API_KEY ? `READY (${GEMINI_SCRIPT_MODEL})` : "NOT CONFIGURED — Groq fallback only"}`
        );

        console.log(
          `Groq Retries: ${GROQ_AI_MAX_RETRIES}`
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
