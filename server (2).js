import 'dotenv/config';

import express from 'express';
import multer from 'multer';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import Groq from 'groq-sdk';
import pg from 'pg';

const { Pool } = pg;
const execFileAsync = promisify(execFile);
const app = express();

const PORT = Number(process.env.PORT || 3000);
const ROOT = process.cwd();
const DATA_DIR = process.env.DATA_DIR ? path.resolve(process.env.DATA_DIR) : ROOT;
const PUBLIC_DIR = path.join(ROOT, 'public');
const UPLOAD_DIR = path.join(DATA_DIR, 'uploads');
const JOB_DIR = path.join(DATA_DIR, 'jobs');

for (const dir of [PUBLIC_DIR, UPLOAD_DIR, JOB_DIR]) {
  fs.mkdirSync(dir, { recursive: true });
}

/* =========================================================
   LIMITS / MODELS
========================================================= */
const MAX_VIDEO_SIZE = 500 * 1024 * 1024;
const MAX_VIDEO_SECONDS = 5 * 60;
const AUDIO_CHUNK_SECONDS = 90;
const MAX_SCENES = 18;
const MIN_SCENE_SECONDS = 3;
const SCENE_THRESHOLD = Number(process.env.SCENE_THRESHOLD || 0.30);

const GROQ_WHISPER_MODEL = 'whisper-large-v3-turbo';
const GROQ_RECAP_MODEL = process.env.GROQ_RECAP_MODEL || 'openai/gpt-oss-120b';
const GROQ_VISION_MODEL = process.env.GROQ_VISION_MODEL || 'qwen/qwen3.8-27b';

const AZURE_SPEECH_REGION = String(process.env.AZURE_SPEECH_REGION || '').trim();
const AZURE_SPEECH_KEY = String(process.env.AZURE_SPEECH_KEY || '').trim();
const AZURE_TTS_MALE = process.env.AZURE_TTS_VOICE_MALE || 'my-MM-ThihaNeural';
const AZURE_TTS_FEMALE = process.env.AZURE_TTS_VOICE_FEMALE || 'my-MM-NilarNeural';

/* =========================================================
   EXPRESS
========================================================= */
app.use(express.json({ limit: '20mb' }));
app.use(express.urlencoded({ extended: true, limit: '20mb' }));

/* =========================================================
   UPLOAD
========================================================= */
const allowedExt = new Set([
  '.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v', '.flv', '.wmv', '.mpeg', '.mpg'
]);

const upload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, UPLOAD_DIR),
    filename: (_req, file, cb) => {
      const ext0 = path.extname(file.originalname || '').toLowerCase();
      const ext = allowedExt.has(ext0) ? ext0 : '.mp4';
      cb(null, `${Date.now()}-${crypto.randomBytes(8).toString('hex')}${ext}`);
    }
  }),
  limits: { fileSize: MAX_VIDEO_SIZE, files: 1 },
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname || '').toLowerCase();
    if (!allowedExt.has(ext)) return cb(new Error('MP4 / MOV / MKV / WEBM / AVI video ကိုသုံးပါ။'));
    cb(null, true);
  }
});

/* =========================================================
   POSTGRES JOB STORAGE
========================================================= */
const DATABASE_URL = process.env.DATABASE_URL;
if (!DATABASE_URL) {
  throw new Error('DATABASE_URL is required.');
}

const db = new Pool({
  connectionString: DATABASE_URL,
  ssl: process.env.DATABASE_SSL === 'false' ? false : { rejectUnauthorized: false },
  max: 5,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 10000
});

const jobs = new Map();
const jobWriteQueues = new Map();

async function initDatabase() {
  await db.query(`
    CREATE TABLE IF NOT EXISTS one_clip_jobs (
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
      timeline JSONB,
      output JSONB,
      error TEXT
    )
  `);
  await db.query(`CREATE INDEX IF NOT EXISTS idx_one_clip_jobs_updated_at ON one_clip_jobs(updated_at DESC)`);
  console.log('[JOB STORAGE] PostgreSQL database ready.');
}

function jobToRow(job) {
  return {
    id: job.id,
    status: job.status || 'created',
    stage: job.stage || 'Waiting',
    progress: Number(job.progress || 0),
    message: job.message || null,
    created_at: job.createdAt,
    updated_at: job.updatedAt,
    original_filename: job.originalFilename || null,
    duration: job.duration == null ? null : Number(job.duration),
    total_chunks: job.totalChunks == null ? null : Number(job.totalChunks),
    transcript: job.transcript ?? null,
    timeline: job.timeline ?? null,
    output: job.output ?? null,
    error: job.error || null
  };
}

function rowToJob(row) {
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
    timeline: row.timeline,
    output: row.output,
    error: row.error
  };
}

async function persistJob(job) {
  const r = jobToRow(job);
  await db.query(`
    INSERT INTO one_clip_jobs (
      id,status,stage,progress,message,created_at,updated_at,
      original_filename,duration,total_chunks,transcript,timeline,output,error
    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb,$12::jsonb,$13::jsonb,$14)
    ON CONFLICT (id) DO UPDATE SET
      status=EXCLUDED.status,
      stage=EXCLUDED.stage,
      progress=EXCLUDED.progress,
      message=EXCLUDED.message,
      updated_at=EXCLUDED.updated_at,
      original_filename=EXCLUDED.original_filename,
      duration=EXCLUDED.duration,
      total_chunks=EXCLUDED.total_chunks,
      transcript=EXCLUDED.transcript,
      timeline=EXCLUDED.timeline,
      output=EXCLUDED.output,
      error=EXCLUDED.error
  `, [
    r.id, r.status, r.stage, r.progress, r.message,
    r.created_at, r.updated_at, r.original_filename,
    r.duration, r.total_chunks,
    JSON.stringify(r.transcript), JSON.stringify(r.timeline), JSON.stringify(r.output), r.error
  ]);
}

function queueJobWrite(job) {
  const previous = jobWriteQueues.get(job.id) || Promise.resolve();
  const next = previous.catch(() => {}).then(() => persistJob(job)).catch(err => {
    console.error(`[JOB STORAGE] ${job.id}:`, err?.message || err);
  });
  jobWriteQueues.set(job.id, next);
  next.finally(() => {
    if (jobWriteQueues.get(job.id) === next) jobWriteQueues.delete(job.id);
  }).catch(() => {});
  return next;
}

async function createJob() {
  const id = crypto.randomUUID();
  const now = new Date().toISOString();
  const job = {
    id,
    status: 'created',
    stage: 'Waiting',
    progress: 0,
    message: 'Job created.',
    createdAt: now,
    updatedAt: now,
    originalFilename: null,
    duration: null,
    totalChunks: null,
    transcript: null,
    timeline: null,
    output: null,
    error: null
  };
  jobs.set(id, job);
  await persistJob(job);
  return job;
}

function updateJob(id, data) {
  const job = jobs.get(id);
  if (!job) return null;
  Object.assign(job, data, { updatedAt: new Date().toISOString() });
  queueJobWrite(job);
  return job;
}

async function getJob(id) {
  if (jobs.has(id)) return jobs.get(id);
  const result = await db.query('SELECT * FROM one_clip_jobs WHERE id=$1 LIMIT 1', [id]);
  if (!result.rows.length) return null;
  const job = rowToJob(result.rows[0]);
  jobs.set(id, job);
  return job;
}

async function waitForJobWrites(id) {
  const pending = jobWriteQueues.get(id);
  if (pending) await pending;
}

/* =========================================================
   HELPERS
========================================================= */
function requireEnv(name) {
  const value = String(process.env[name] || '').trim();
  if (!value) throw new Error(`${name} is not configured.`);
  return value;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function runCommand(command, args) {
  try {
    const result = await execFileAsync(command, args, { maxBuffer: 100 * 1024 * 1024 });
    return result;
  } catch (error) {
    throw new Error(`${command} failed: ${error.stderr || error.message || 'Unknown error'}`);
  }
}

async function getVideoDuration(videoPath) {
  const r = await runCommand('ffprobe', [
    '-v', 'error', '-show_entries', 'format=duration',
    '-of', 'default=noprint_wrappers=1:nokey=1', videoPath
  ]);
  const duration = Number(r.stdout.trim());
  if (!Number.isFinite(duration) || duration <= 0) throw new Error('Unable to read video duration.');
  return duration;
}

async function extractAudio(videoPath, outputPath) {
  await runCommand('ffmpeg', [
    '-y', '-i', videoPath, '-vn', '-map', '0:a:0',
    '-ac', '1', '-ar', '16000', '-b:a', '64k', '-c:a', 'libmp3lame', outputPath
  ]);
}

async function splitAudio(audioPath, outputDir) {
  fs.mkdirSync(outputDir, { recursive: true });
  await runCommand('ffmpeg', [
    '-y', '-i', audioPath, '-f', 'segment', '-segment_time', String(AUDIO_CHUNK_SECONDS),
    '-reset_timestamps', '1', '-c', 'copy', path.join(outputDir, 'chunk-%04d.mp3')
  ]);
  const files = fs.readdirSync(outputDir)
    .filter(x => /^chunk-\d+\.mp3$/.test(x))
    .sort()
    .map(x => path.join(outputDir, x));
  if (!files.length) throw new Error('FFmpeg could not create audio chunks.');
  return files;
}

/* =========================================================
   GROQ WHISPER
========================================================= */
async function transcribeChunk(groq, audioPath, chunkIndex) {
  const response = await groq.audio.transcriptions.create({
    file: fs.createReadStream(audioPath),
    model: GROQ_WHISPER_MODEL,
    response_format: 'verbose_json',
    timestamp_granularities: ['segment'],
    temperature: 0
  });
  const offset = chunkIndex * AUDIO_CHUNK_SECONDS;
  const segments = Array.isArray(response.segments) ? response.segments : [];
  return {
    text: String(response.text || '').trim(),
    segments: segments.map(s => ({
      start: Number(s.start || 0) + offset,
      end: Number(s.end || 0) + offset,
      text: String(s.text || '').trim()
    })).filter(s => s.end > s.start && s.text)
  };
}

async function transcribeMovie(jobId, audioChunks) {
  const groq = new Groq({ apiKey: requireEnv('GROQ_API_KEY') });
  const segments = [];
  const texts = [];
  for (let i = 0; i < audioChunks.length; i++) {
    updateJob(jobId, {
      stage: 'Whisper',
      progress: 15 + Math.round((i / audioChunks.length) * 30),
      message: `Groq Whisper: ${i + 1} / ${audioChunks.length}`
    });
    const r = await transcribeChunk(groq, audioChunks[i], i);
    if (r.text) texts.push(r.text);
    segments.push(...r.segments);
  }
  segments.sort((a, b) => a.start - b.start);
  return { text: texts.join(' ').trim(), segments };
}

/* =========================================================
   VISUAL SCENE CUT DETECTION — NO GEMINI VIDEO ANALYSIS
========================================================= */
async function detectSceneCuts(videoPath, duration) {
  const filter = `select='gt(scene,${SCENE_THRESHOLD})',showinfo`;
  let stderr = '';
  try {
    await execFileAsync('ffmpeg', [
      '-hide_banner', '-i', videoPath,
      '-filter:v', filter,
      '-an', '-f', 'null', '-'
    ], { maxBuffer: 100 * 1024 * 1024 });
  } catch (error) {
    stderr = String(error.stderr || '');
  }

  const matches = [...stderr.matchAll(/pts_time:([0-9.]+)/g)];
  const rawCuts = matches.map(m => Number(m[1]))
    .filter(n => Number.isFinite(n) && n > 0.5 && n < duration - 0.5);

  const cuts = [0];
  for (const cut of rawCuts) {
    if (cut - cuts[cuts.length - 1] >= 0.75) cuts.push(cut);
  }
  cuts.push(duration);

  return reduceTimelineCuts(cuts, MAX_SCENES, MIN_SCENE_SECONDS);
}

function reduceTimelineCuts(cuts, maxScenes, minDuration) {
  let boundaries = [...cuts].sort((a, b) => a - b);
  boundaries = boundaries.filter((v, i) => i === 0 || v - boundaries[i - 1] >= 0.75);

  while (boundaries.length - 1 > maxScenes) {
    let best = 1;
    let bestScore = Infinity;
    for (let i = 1; i < boundaries.length - 1; i++) {
      const left = boundaries[i] - boundaries[i - 1];
      const right = boundaries[i + 1] - boundaries[i];
      const score = Math.min(left, right);
      if (score < bestScore) {
        bestScore = score;
        best = i;
      }
    }
    boundaries.splice(best, 1);
  }

  // Remove very short intervals while preserving the outer timeline.
  let changed = true;
  while (changed && boundaries.length > 2) {
    changed = false;
    for (let i = 1; i < boundaries.length - 1; i++) {
      const left = boundaries[i] - boundaries[i - 1];
      const right = boundaries[i + 1] - boundaries[i];
      if (left < minDuration || right < minDuration) {
        // Remove the boundary that creates the smaller neighboring scene.
        boundaries.splice(i, 1);
        changed = true;
        break;
      }
    }
  }

  return boundaries;
}

function overlapText(segments, start, end) {
  return segments
    .filter(s => s.end > start && s.start < end)
    .map(s => s.text)
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function buildTimeline(boundaries, transcriptSegments) {
  const blocks = [];
  for (let i = 0; i < boundaries.length - 1; i++) {
    const start = boundaries[i];
    const end = boundaries[i + 1];
    blocks.push({
      index: i + 1,
      start: Number(start.toFixed(3)),
      end: Number(end.toFixed(3)),
      duration: Number((end - start).toFixed(3)),
      sourceTranscript: overlapText(transcriptSegments, start, end),
      narration: ''
    });
  }
  return blocks;
}

function mergeBlockInto(blocks, index, targetIndex) {
  const a = blocks[index];
  const b = blocks[targetIndex];
  const start = Math.min(a.start, b.start);
  const end = Math.max(a.end, b.end);
  const merged = {
    ...a,
    start,
    end,
    duration: end - start,
    sourceTranscript: `${a.sourceTranscript} ${b.sourceTranscript}`.replace(/\s+/g, ' ').trim()
  };
  const out = blocks.filter((_, i) => i !== index && i !== targetIndex);
  out.push(merged);
  out.sort((x, y) => x.start - y.start);
  return out.map((x, i) => ({ ...x, index: i + 1 }));
}

function ensureTranscriptCoverage(blocks, transcriptSegments) {
  let out = blocks.map(b => ({ ...b }));
  for (let i = 0; i < out.length; i++) {
    if (out[i].sourceTranscript) continue;
    if (out.length === 1) continue;
    if (i < out.length - 1) out = mergeBlockInto(out, i, i + 1);
    else out = mergeBlockInto(out, i, i - 1);
    i = -1;
  }
  // Recompute transcript after merges.
  return out.map((b, i) => ({
    ...b,
    index: i + 1,
    sourceTranscript: overlapText(transcriptSegments, b.start, b.end)
  }));
}

/* =========================================================
   GROQ RECAP — TIMELINE-AWARE
========================================================= */
function chooseRecapStyle(style) {
  const styles = {
    cinematic: 'cinematic movie recap narration, natural and emotional but controlled',
    short: 'short, punchy, fast-paced movie recap narration',
    storytelling: 'smooth storytelling with suspense and emotional flow',
    detailed: 'detailed but natural movie recap narration'
  };
  return styles[style] || styles.cinematic;
}

async function generateNarration(groq, block, language, style) {
  if (language === 'en') {
    const prompt = `You are writing a professional movie recap voice-over.\n\nTIMELINE: ${block.start.toFixed(2)}s to ${block.end.toFixed(2)}s (${block.duration.toFixed(2)}s).\nORIGINAL TRANSCRIPT IN THIS EXACT TIME WINDOW:\n${block.sourceTranscript || '(no spoken dialogue in this window)'}\n\nWrite ONE narration for only this time window. ${chooseRecapStyle(style)}. Do not invent events, characters, dialogue, locations, or actions not supported by the supplied transcript. Keep the narration short enough to be spoken naturally within ${Math.max(3, block.duration - 0.2).toFixed(1)} seconds. Output only the narration text.`;
    const r = await groq.chat.completions.create({
      model: GROQ_RECAP_MODEL,
      temperature: 0.35,
      max_tokens: 350,
      messages: [
        { role: 'system', content: 'Return only one clean voice-over paragraph.' },
        { role: 'user', content: prompt }
      ]
    });
    return String(r.choices?.[0]?.message?.content || '').trim();
  }

  const prompt = `You are writing a PROFESSIONAL MYANMAR MOVIE RECAP voice-over.\n\nTIMELINE: ${block.start.toFixed(2)}s to ${block.end.toFixed(2)}s (${block.duration.toFixed(2)}s).\nORIGINAL WHISPER TRANSCRIPT FROM THIS EXACT TIME WINDOW:\n${block.sourceTranscript || '(ဒီအချိန်ပိုင်းမှာ စကားပြောသံမရှိပါ)'}\n\nWrite ONE natural spoken Myanmar narration for ONLY this time window. Style: ${chooseRecapStyle(style)}.\n\nSTRICT RULES:\n- Use natural spoken Burmese suitable for a movie recap narrator.\n- Do not invent characters, dialogue, places, actions, or story facts.\n- Do not move an event to another time.\n- Do not mention timestamps, AI, prompts, or these instructions.\n- Keep it concise enough to speak naturally within ${Math.max(3, block.duration - 0.2).toFixed(1)} seconds.\n- Output ONLY the Myanmar narration paragraph.\n\nIf the transcript is empty, write a very short neutral transition that does not claim a specific unseen event.`;

  const r = await groq.chat.completions.create({
    model: GROQ_RECAP_MODEL,
    temperature: 0.35,
    max_tokens: 350,
    messages: [
      { role: 'system', content: 'You are a careful Burmese movie recap writer. Never invent unsupported story facts.' },
      { role: 'user', content: prompt }
    ]
  });
  return String(r.choices?.[0]?.message?.content || '').trim();
}

async function generateTimelineNarration(jobId, blocks, language, style) {
  const groq = new Groq({ apiKey: requireEnv('GROQ_API_KEY') });
  const out = [];
  for (let i = 0; i < blocks.length; i++) {
    updateJob(jobId, {
      stage: 'Gemini',
      progress: 50 + Math.round((i / blocks.length) * 15),
      message: `Groq Recap: ${i + 1} / ${blocks.length}`
    });
    let narration = '';
    try {
      narration = await generateNarration(groq, blocks[i], language, style);
    } catch (error) {
      console.error(`[GROQ RECAP] Block ${blocks[i].index}:`, error?.message || error);
    }
    if (!narration) {
      narration = blocks[i].sourceTranscript || (language === 'en' ? 'The story continues.' : 'ဇာတ်လမ်းက ဆက်လက်ပြီး ရှေ့ဆက်သွားပါတယ်။');
    }
    out.push({ ...blocks[i], narration });
  }
  return out;
}

/* =========================================================
   OPTIONAL VISUAL CHECK — GROQ VISION, 1 FRAME PER BLOCK
   This does NOT replace the timeline. It only gives recap text
   a visual hint so the narration follows visible events better.
========================================================= */
async function extractFrame(videoPath, time, outputPath) {
  await runCommand('ffmpeg', [
    '-y', '-ss', String(Math.max(0, time.toFixed(3))), '-i', videoPath,
    '-frames:v', '1', '-vf', 'scale=768:-2', '-q:v', '4', outputPath
  ]);
}

function fileToDataUrl(filePath) {
  const mime = path.extname(filePath).toLowerCase() === '.png' ? 'image/png' : 'image/jpeg';
  return `data:${mime};base64,${fs.readFileSync(filePath).toString('base64')}`;
}

async function enrichNarrationWithVisuals(jobId, videoPath, blocks, language, style, jobFolder) {
  if (process.env.ENABLE_GROQ_VISION !== 'true') return blocks;

  const groq = new Groq({ apiKey: requireEnv('GROQ_API_KEY') });
  const frameDir = path.join(jobFolder, 'frames');
  fs.mkdirSync(frameDir, { recursive: true });

  const out = [];
  for (let i = 0; i < blocks.length; i++) {
    const b = blocks[i];
    updateJob(jobId, {
      stage: 'Gemini',
      progress: 65 + Math.round((i / blocks.length) * 5),
      message: `Visual check: ${i + 1} / ${blocks.length}`
    });

    const framePath = path.join(frameDir, `frame-${String(b.index).padStart(3, '0')}.jpg`);
    try {
      await extractFrame(videoPath, (b.start + b.end) / 2, framePath);
      const dataUrl = fileToDataUrl(framePath);
      const visualPrompt = language === 'en'
        ? `Describe only the clearly visible people, objects, setting, and action in this movie frame. Do not identify real people. Do not invent details. This frame is from ${b.start.toFixed(2)}-${b.end.toFixed(2)} seconds. Original transcript: ${b.sourceTranscript || '(none)'}`
        : `ဒီ movie frame ထဲမှာ မြင်ရတာကိုပဲ တိတိကျကျဖော်ပြပါ။ လူ၊ ပစ္စည်း၊ နေရာ၊ လုပ်ဆောင်ချက်တွေကို မြင်ရသလောက်ပဲ ပြောပါ။ မမြင်ရတာ မခန့်မှန်းပါနဲ့။ ဒီ frame က ${b.start.toFixed(2)}-${b.end.toFixed(2)} seconds အတွင်းက ဖြစ်ပါတယ်။ Original transcript: ${b.sourceTranscript || '(မရှိပါ)'}`;

      const r = await groq.chat.completions.create({
        model: GROQ_VISION_MODEL,
        temperature: 0.1,
        max_tokens: 220,
        messages: [{
          role: 'user',
          content: [
            { type: 'text', text: visualPrompt },
            { type: 'image_url', image_url: { url: dataUrl } }
          ]
        }]
      });
      const visual = String(r.choices?.[0]?.message?.content || '').trim();
      out.push({ ...b, visualHint: visual });
    } catch (error) {
      console.error(`[GROQ VISION] Block ${b.index}:`, error?.message || error);
      out.push({ ...b, visualHint: '' });
    }
  }

  return out;
}

/* =========================================================
   AZURE SPEECH TTS
========================================================= */
function resolveAzureVoice(voice) {
  const v = String(voice || '').toLowerCase();
  if (v === 'male' || v.includes('thiha')) return AZURE_TTS_MALE;
  return AZURE_TTS_FEMALE;
}

function escapeXml(text) {
  return String(text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

async function azureTTS(text, voice, outputPath) {
  if (!AZURE_SPEECH_KEY || !AZURE_SPEECH_REGION) {
    throw new Error('Azure Speech is not configured. Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in Render Environment.');
  }

  const endpoint = `https://${AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1`;
  const voiceName = resolveAzureVoice(voice);
  const language = voiceName.startsWith('my-MM') ? 'my-MM' : 'en-US';
  const safeText = escapeXml(text);
  const isMale = voiceName === AZURE_TTS_MALE;
  const rate = isMale ? '-3%' : '-1%';
  const pitch = isMale ? '-1st' : '+0st';

  const ssml = `<speak version="1.0" xml:lang="${language}"><voice xml:lang="${language}" name="${voiceName}"><prosody rate="${rate}" pitch="${pitch}">${safeText}</prosody></voice></speak>`;

  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Ocp-Apim-Subscription-Key': AZURE_SPEECH_KEY,
      'Content-Type': 'application/ssml+xml',
      'X-Microsoft-OutputFormat': 'audio-24khz-48kbitrate-mono-mp3',
      'User-Agent': 'YNT-One-Clips'
    },
    body: ssml
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Azure Speech HTTP ${response.status}: ${body.slice(0, 1000)}`);
  }

  const audio = Buffer.from(await response.arrayBuffer());
  if (audio.length < 100) throw new Error('Azure Speech returned an empty audio file.');
  fs.writeFileSync(outputPath, audio);
  return outputPath;
}

/* =========================================================
   AUDIO FIT / TIMELINE MIX
========================================================= */
function buildAtempoFilters(factor) {
  let value = factor;
  const filters = [];
  while (value > 2) { filters.push('atempo=2.0'); value /= 2; }
  while (value < 0.5) { filters.push('atempo=0.5'); value /= 0.5; }
  if (Math.abs(value - 1) > 0.001) filters.push(`atempo=${value.toFixed(6)}`);
  return filters;
}

async function getAudioDuration(audioPath) {
  const r = await runCommand('ffprobe', [
    '-v', 'error', '-show_entries', 'format=duration',
    '-of', 'default=noprint_wrappers=1:nokey=1', audioPath
  ]);
  const d = Number(r.stdout.trim());
  if (!Number.isFinite(d) || d <= 0) throw new Error('Generated TTS has invalid duration.');
  return d;
}

async function fitAudioToDuration(inputPath, outputPath, targetDuration) {
  const sourceDuration = await getAudioDuration(inputPath);
  const factor = sourceDuration / targetDuration;
  const filters = buildAtempoFilters(factor);
  if (!filters.length) filters.push('anull');
  filters.push('apad', `atrim=duration=${targetDuration.toFixed(3)}`);
  await runCommand('ffmpeg', [
    '-y', '-i', inputPath,
    '-af', filters.join(','),
    '-ar', '24000', '-ac', '1', '-c:a', 'pcm_s16le', outputPath
  ]);
  return { sourceDuration, targetDuration, factor };
}

async function renderTimelineAudio(jobId, blocks, voice, jobFolder) {
  const audioDir = path.join(jobFolder, 'timeline-audio');
  fs.mkdirSync(audioDir, { recursive: true });

  const fitted = [];
  for (let i = 0; i < blocks.length; i++) {
    const b = blocks[i];
    updateJob(jobId, {
      stage: 'Voice',
      progress: 70 + Math.round((i / blocks.length) * 15),
      message: `Azure Myanmar Voice: ${i + 1} / ${blocks.length}`
    });

    const raw = path.join(audioDir, `raw-${String(b.index).padStart(3, '0')}.mp3`);
    const fit = path.join(audioDir, `fit-${String(b.index).padStart(3, '0')}.wav`);
    await azureTTS(b.narration, voice, raw);
    const timing = await fitAudioToDuration(raw, fit, b.duration);
    fitted.push({ ...b, audioPath: fit, sourceAudioDuration: timing.sourceDuration });
  }

  const timelinePath = path.join(jobFolder, 'timeline-audio.wav');
  const inputs = [];
  const filterParts = [];

  fitted.forEach((b, i) => {
    inputs.push('-i', b.audioPath);
    filterParts.push(`[${i}:a]adelay=${Math.round(b.start * 1000)}|${Math.round(b.start * 1000)},apad=whole_dur=${Math.ceil(b.start + b.duration + 0.1)}[a${i}]`);
  });

  const mixInputs = fitted.map((_, i) => `[a${i}]`).join('');
  const filter = `${filterParts.join(';')};${mixInputs}amix=inputs=${fitted.length}:duration=longest:dropout_transition=0:normalize=0[aout]`;

  await runCommand('ffmpeg', [
    '-y', ...inputs,
    '-filter_complex', filter,
    '-map', '[aout]',
    '-ar', '24000', '-ac', '1', '-c:a', 'pcm_s16le',
    timelinePath
  ]);

  return { timelinePath, blocks: fitted };
}

/* =========================================================
   FINAL VIDEO — KEEP ORIGINAL VIDEO TIMELINE
========================================================= */
async function renderFinalVideo(moviePath, narrationAudio, outputPath, duration) {
  await runCommand('ffmpeg', [
    '-y', '-i', moviePath, '-i', narrationAudio,
    '-map', '0:v:0', '-map', '1:a:0',
    '-c:v', 'copy',
    '-c:a', 'aac', '-b:a', '160k',
    '-t', duration.toFixed(3),
    '-shortest',
    '-movflags', '+faststart',
    outputPath
  ]);
  return outputPath;
}

/* =========================================================
   PROCESS JOB
========================================================= */
async function processOneClip(jobId, moviePath, language, style, voice) {
  const jobFolder = path.dirname(moviePath);
  try {
    updateJob(jobId, { stage: 'Upload', progress: 5, message: 'Movie information စစ်နေပါတယ်...' });
    const duration = await getVideoDuration(moviePath);
    if (duration > MAX_VIDEO_SECONDS + 0.5) throw new Error(`Video က ${MAX_VIDEO_SECONDS / 60} မိနစ်ထက်မကျော်ရပါ။`);
    updateJob(jobId, { duration });

    const audioPath = path.join(jobFolder, 'movie-audio.mp3');
    updateJob(jobId, { stage: 'Whisper', progress: 10, message: 'Movie audio ကို Whisper အတွက် ပြင်နေပါတယ်...' });
    await extractAudio(moviePath, audioPath);

    const chunksDir = path.join(jobFolder, 'audio-chunks');
    const audioChunks = await splitAudio(audioPath, chunksDir);
    updateJob(jobId, { totalChunks: audioChunks.length });

    const transcript = await transcribeMovie(jobId, audioChunks);
    if (!transcript.text) throw new Error('Groq Whisper returned an empty transcript.');
    fs.writeFileSync(path.join(jobFolder, 'transcript.json'), JSON.stringify(transcript, null, 2), 'utf8');
    updateJob(jobId, {
      stage: 'Transcript', progress: 47,
      message: `Whisper timestamp ${transcript.segments.length} segments ရပါပြီ။`,
      transcript: { characters: transcript.text.length, segments: transcript.segments.length }
    });

    updateJob(jobId, { stage: 'Gemini', progress: 50, message: 'Groq Recap + movie timeline ကိုတည်ဆောက်နေပါတယ်...' });
    const boundaries = await detectSceneCuts(moviePath, duration);
    let blocks = buildTimeline(boundaries, transcript.segments);
    blocks = ensureTranscriptCoverage(blocks, transcript.segments);
    blocks = await generateTimelineNarration(jobId, blocks, language, style);
    blocks = await enrichNarrationWithVisuals(jobId, moviePath, blocks, language, style, jobFolder);

    // If visual hints are enabled, ask Groq once more per block to reconcile transcript + frame.
    if (process.env.ENABLE_GROQ_VISION === 'true') {
      const groq = new Groq({ apiKey: requireEnv('GROQ_API_KEY') });
      for (let i = 0; i < blocks.length; i++) {
        const b = blocks[i];
        if (!b.visualHint) continue;
        try {
          const prompt = language === 'en'
            ? `Rewrite this movie recap narration so it matches BOTH the visible frame and the transcript. Time window: ${b.start.toFixed(2)}-${b.end.toFixed(2)}s. Transcript: ${b.sourceTranscript || '(none)'}. Visual: ${b.visualHint}. Keep it concise for ${b.duration.toFixed(1)} seconds. Do not invent anything. Output only narration.`
            : `ဒီ movie recap narration ကို မြင်ရတဲ့ frame နဲ့ transcript နှစ်ခုလုံးနဲ့ ကိုက်ညီအောင် ပြန်ရေးပါ။ Time: ${b.start.toFixed(2)}-${b.end.toFixed(2)}s။ Transcript: ${b.sourceTranscript || '(မရှိ)'}. Visual: ${b.visualHint}. ${b.duration.toFixed(1)} စက္ကန့်အတွင်း ပြောနိုင်အောင် တိုတိုရှင်းရှင်းရေးပါ။ မမြင်ရတာ၊ transcript မထောက်ခံတာ မတီထွင်ပါနဲ့။ မြန်မာလို narration တစ်ပိုဒ်ပဲ ပြန်ပေးပါ။`;
          const r = await groq.chat.completions.create({
            model: GROQ_RECAP_MODEL,
            temperature: 0.25,
            max_tokens: 300,
            messages: [
              { role: 'system', content: 'Return only one clean narration paragraph.' },
              { role: 'user', content: prompt }
            ]
          });
          const revised = String(r.choices?.[0]?.message?.content || '').trim();
          if (revised) b.narration = revised;
        } catch (error) {
          console.error(`[GROQ VISION REWRITE] Block ${b.index}:`, error?.message || error);
        }
      }
    }

    fs.writeFileSync(path.join(jobFolder, 'timeline.json'), JSON.stringify(blocks, null, 2), 'utf8');
    updateJob(jobId, {
      stage: 'Gemini', progress: 68,
      message: `${blocks.length} visual timeline blocks ready.`,
      timeline: { blocks: blocks.length, path: path.join(jobFolder, 'timeline.json') }
    });

    const narration = await renderTimelineAudio(jobId, blocks, voice, jobFolder);

    updateJob(jobId, { stage: 'FFmpeg', progress: 90, message: 'Movie timeline + Myanmar voice ကို final MP4 ထဲထည့်နေပါတယ်...' });
    const outputPath = path.join(jobFolder, 'YNT-One-Clips-Recap.mp4');
    await renderFinalVideo(moviePath, narration.timelinePath, outputPath, duration);

    const stats = fs.statSync(outputPath);
    if (!stats.size) throw new Error('Final MP4 file is empty.');

    const filename = `YNT-One-Clips-${jobId}.mp4`;
    updateJob(jobId, {
      status: 'completed', stage: 'Ready', progress: 100,
      message: 'Your synchronized recap video is ready.',
      output: { path: outputPath, filename, size: stats.size, url: `/api/download/${jobId}` }
    });
    console.log(`[JOB ${jobId}] COMPLETED`);
  } catch (error) {
    console.error(`[JOB ${jobId}] ERROR:`, error);
    updateJob(jobId, {
      status: 'error', stage: 'Error', progress: 0,
      message: error?.message || 'Movie processing failed.',
      error: error?.message || 'Movie processing failed.'
    });
  }
}

/* =========================================================
   API
========================================================= */
app.get('/api/health', (_req, res) => {
  res.json({
    ok: true,
    name: 'YNT One Clips',
    status: 'online',
    models: {
      whisper: GROQ_WHISPER_MODEL,
      recap: GROQ_RECAP_MODEL,
      tts: 'Azure Speech'
    },
    azureVoices: { male: AZURE_TTS_MALE, female: AZURE_TTS_FEMALE },
    sceneSync: true,
    sceneDetection: 'FFmpeg',
    visualCheck: process.env.ENABLE_GROQ_VISION === 'true',
    maxScenes: MAX_SCENES,
    maxVideoSeconds: MAX_VIDEO_SECONDS,
    maxVideoSizeMB: Math.round(MAX_VIDEO_SIZE / 1024 / 1024),
    jobStorage: 'PostgreSQL'
  });
});

app.get('/health', (_req, res) => res.json({ ok: true, service: 'YNT One Clips' }));

app.get('/api/status/:id', async (req, res) => {
  try {
    const job = await getJob(req.params.id);
    if (!job) return res.status(404).json({ ok: false, error: 'Job not found' });
    return res.json({ ok: true, job });
  } catch (error) {
    return res.status(500).json({ ok: false, error: error.message });
  }
});

app.get('/api/download/:id', async (req, res) => {
  try {
    const job = await getJob(req.params.id);
    await waitForJobWrites(req.params.id);
    if (!job) return res.status(404).send('Job not found');
    if (job.status !== 'completed' || !job.output?.path) return res.status(404).send('Final Video မရသေးပါ');
    if (!fs.existsSync(job.output.path)) return res.status(404).send('Output video file မတွေ့ပါ။');
    return res.download(job.output.path, job.output.filename || 'YNT-One-Clips.mp4');
  } catch (error) {
    return res.status(500).send(error.message || 'Download failed');
  }
});

app.post('/api/one-clip', upload.single('movie'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ ok: false, error: 'Movie file is required.' });

    const job = await createJob();
    const jobFolder = path.join(JOB_DIR, job.id);
    fs.mkdirSync(jobFolder, { recursive: true });

    const originalName = req.file.originalname || 'movie.mp4';
    const ext = path.extname(originalName) || '.mp4';
    const moviePath = path.join(jobFolder, `movie${ext}`);
    fs.renameSync(req.file.path, moviePath);

    const language = req.body?.language || 'my';
    const style = req.body?.style || 'cinematic';
    const voice = req.body?.voice || 'female';

    updateJob(job.id, {
      originalFilename: originalName,
      status: 'processing', stage: 'Upload', progress: 5,
      message: 'Movie uploaded. Processing started.'
    });

    console.log('========================================');
    console.log(`[JOB ${job.id}] STARTED`);
    console.log(`[JOB ${job.id}] Language: ${language}`);
    console.log(`[JOB ${job.id}] Style: ${style}`);
    console.log(`[JOB ${job.id}] Voice: ${voice}`);
    console.log('[JOB] Gemini video analysis: DISABLED');
    console.log('[JOB] Gemini TTS: DISABLED');
    console.log('[JOB] Groq Whisper: ENABLED');
    console.log('[JOB] Groq Recap: ENABLED');
    console.log('[JOB] Azure Myanmar TTS: ENABLED');
    console.log('[JOB] FFmpeg visual timeline: ENABLED');
    console.log('========================================');

    res.status(202).json({
      success: true,
      jobId: job.id,
      status: 'processing',
      message: 'Movie processing started.',
      statusUrl: `/api/status/${job.id}`,
      download: `/api/download/${job.id}`
    });

    setImmediate(() => {
      processOneClip(job.id, moviePath, language, style, voice).catch(error => {
        console.error(`[JOB ${job.id}] UNHANDLED ERROR:`, error);
        updateJob(job.id, { status: 'error', stage: 'Error', progress: 0, message: error.message, error: error.message });
      });
    });
  } catch (error) {
    console.error('ONE CLIP START ERROR:', error);
    if (req.file?.path && fs.existsSync(req.file.path)) {
      try { fs.unlinkSync(req.file.path); } catch {}
    }
    return res.status(500).json({ ok: false, error: error.message || 'Unable to start movie processing.' });
  }
});

/* =========================================================
   FRONTEND — DO NOT CHANGE HTML
========================================================= */
app.use(express.static(PUBLIC_DIR));

app.get('/', (_req, res) => {
  const publicIndex = path.join(PUBLIC_DIR, 'index.html');
  const rootIndex = path.join(ROOT, 'index.html');
  if (fs.existsSync(publicIndex)) return res.sendFile(publicIndex);
  if (fs.existsSync(rootIndex)) return res.sendFile(rootIndex);
  return res.status(404).send('index.html not found');
});

app.use((err, _req, res, _next) => {
  console.error('SERVER ERROR:', err);
  if (err instanceof multer.MulterError) return res.status(400).json({ ok: false, error: err.message });
  return res.status(500).json({ ok: false, error: err.message || 'Server error.' });
});

/* =========================================================
   START
========================================================= */
async function startServer() {
  await initDatabase();
  app.listen(PORT, '0.0.0.0', () => {
    console.log('========================================');
    console.log('YNT One Clips — Azure TTS + Timeline Sync');
    console.log(`Server: 0.0.0.0:${PORT}`);
    console.log(`Groq Whisper: ${GROQ_WHISPER_MODEL}`);
    console.log(`Groq Recap: ${GROQ_RECAP_MODEL}`);
    console.log(`Azure TTS Male: ${AZURE_TTS_MALE}`);
    console.log(`Azure TTS Female: ${AZURE_TTS_FEMALE}`);
    console.log(`FFmpeg Scene Detection: threshold ${SCENE_THRESHOLD}`);
    console.log(`Groq Vision Check: ${process.env.ENABLE_GROQ_VISION === 'true' ? 'ON' : 'OFF'}`);
    console.log('Gemini Video Analysis: OFF');
    console.log('Gemini TTS: OFF');
    console.log('PostgreSQL Job Storage: ON');
    console.log('Frontend: /public/index.html');
    console.log('========================================');
  });
}

startServer().catch(async error => {
  console.error('[STARTUP ERROR]', error?.message || error);
  await db.end().catch(() => {});
  process.exit(1);
});
