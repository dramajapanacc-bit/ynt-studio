import 'dotenv/config';

import express from 'express';
import multer from 'multer';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import Groq from 'groq-sdk';
import { GoogleGenAI } from '@google/genai';
import pg from 'pg';

const { Pool } = pg;
const execFileAsync = promisify(execFile);
const app = express();

const PORT = Number(process.env.PORT || 3000);
const ROOT = process.cwd();
const PUBLIC_DIR = path.join(ROOT, 'public');
const UPLOAD_DIR = path.join(ROOT, 'uploads');
const JOB_DIR = path.join(ROOT, 'jobs');

const MAX_VIDEO_SIZE = 500 * 1024 * 1024;
const MAX_VIDEO_SECONDS = 5 * 60;
const WHISPER_CHUNK_SECONDS = 90;
const RECAP_BLOCK_SECONDS = 60;
const MAX_RECAP_BLOCK_SECONDS = 75;
const MIN_RECAP_BLOCK_SECONDS = 20;

const GROQ_WHISPER_MODEL = 'whisper-large-v3-turbo';
const GROQ_RECAP_MODEL = 'openai/gpt-oss-120b';
const GEMINI_TTS_MODEL = 'gemini-3.8-flash-lite-tts';
const GEMINI_TTS_RETRIES = 2;
const GEMINI_TTS_RETRY_DELAY = 2500;

const GEMINI_VOICES = {
  male: 'Puck',
  female: 'Kore',
  kore: 'Kore',
  puck: 'Puck',
  aoede: 'Aoede',
  zephyr: 'Zephyr',
  charon: 'Charon',
  fenrir: 'Fenrir'
};

for (const dir of [PUBLIC_DIR, UPLOAD_DIR, JOB_DIR]) {
  fs.mkdirSync(dir, { recursive: true });
}

app.use(express.json({ limit: '20mb' }));
app.use(express.urlencoded({ extended: true, limit: '20mb' }));

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
    if (!allowedExt.has(ext)) {
      return cb(new Error('MP4 / MOV / MKV / WEBM / AVI video ကိုသုံးပါ။'));
    }
    cb(null, true);
  }
});

/* =========================================================
   POSTGRESQL JOB STORAGE
========================================================= */

let pool = null;

async function initDatabase() {
  if (!process.env.DATABASE_URL) {
    throw new Error('DATABASE_URL မရှိပါ။ Render PostgreSQL Internal Database URL ကို Environment ထဲမှာထည့်ပါ။');
  }

  pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: process.env.DATABASE_URL.includes('localhost') || process.env.DATABASE_URL.includes('127.0.0.1')
      ? false
      : { rejectUnauthorized: false },
    max: 5,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 10000
  });

  await pool.query(`
    CREATE TABLE IF NOT EXISTS one_clip_jobs (
      id TEXT PRIMARY KEY,
      status TEXT NOT NULL,
      stage TEXT,
      progress INTEGER DEFAULT 0,
      message TEXT,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      original_filename TEXT,
      duration DOUBLE PRECISION,
      total_chunks INTEGER,
      transcript JSONB,
      recap_blocks JSONB,
      output JSONB,
      error TEXT
    )
  `);

  console.log('PostgreSQL database ready.');
}

function rowToJob(row) {
  if (!row) return null;
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
    recapBlocks: row.recap_blocks,
    output: row.output,
    error: row.error
  };
}

async function createJob(originalFilename = null) {
  const id = crypto.randomUUID();
  const result = await pool.query(
    `INSERT INTO one_clip_jobs
      (id, status, stage, progress, message, original_filename)
     VALUES ($1, 'created', 'Waiting', 0, 'Job created.', $2)
     RETURNING *`,
    [id, originalFilename]
  );
  return rowToJob(result.rows[0]);
}

async function updateJob(id, data) {
  const fields = [];
  const values = [];
  let n = 1;

  const map = {
    status: 'status',
    stage: 'stage',
    progress: 'progress',
    message: 'message',
    originalFilename: 'original_filename',
    duration: 'duration',
    totalChunks: 'total_chunks',
    transcript: 'transcript',
    recapBlocks: 'recap_blocks',
    output: 'output',
    error: 'error'
  };

  for (const [key, column] of Object.entries(map)) {
    if (!(key in data)) continue;
    fields.push(`${column} = $${n++}`);
    const value = data[key];
    values.push(['transcript', 'recapBlocks', 'output'].includes(key) ? JSON.stringify(value) : value);
  }

  if (!fields.length) return getJob(id);
  fields.push('updated_at = NOW()');
  values.push(id);

  const result = await pool.query(
    `UPDATE one_clip_jobs SET ${fields.join(', ')} WHERE id = $${n} RETURNING *`,
    values
  );

  return rowToJob(result.rows[0]);
}

async function getJob(id) {
  const result = await pool.query(
    'SELECT * FROM one_clip_jobs WHERE id = $1',
    [id]
  );
  return rowToJob(result.rows[0]);
}

/* =========================================================
   HELPERS
========================================================= */

function cleanText(value) {
  return String(value ?? '')
    .replace(/\r/g, ' ')
    .replace(/\n+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function cleanup(file) {
  if (!file) return;
  fs.rm(file, { force: true }, () => {});
}

function cleanupMany(files) {
  for (const file of files || []) cleanup(file);
}

function parseErrorMessage(error) {
  return String(error?.message || error?.error?.message || error || 'Unknown error');
}

function requireEnv(name) {
  const value = String(process.env[name] || '').trim();
  if (!value) throw new Error(`${name} မရှိပါ`);
  return value;
}

async function run(command, args, options = {}) {
  try {
    const result = await execFileAsync(command, args, {
      maxBuffer: 100 * 1024 * 1024,
      ...options
    });
    return result;
  } catch (error) {
    const details = error?.stderr || error?.message || 'Unknown command error';
    throw new Error(`${command} failed: ${details}`);
  }
}

function resolveVoice(value) {
  const key = String(value || 'female').trim().toLowerCase();
  return GEMINI_VOICES[key] || 'Kore';
}

function styleInstruction(style) {
  const s = String(style || 'cinematic').toLowerCase();
  if (s.includes('short')) return 'concise, fast-paced and easy to follow';
  if (s.includes('story')) return 'smooth storytelling with suspense and emotional flow';
  if (s.includes('detailed')) return 'detailed but natural movie recap narration';
  if (s.includes('dramatic')) return 'dramatic and emotional but still natural';
  return 'cinematic, natural and easy to understand';
}

/* =========================================================
   VIDEO / AUDIO
========================================================= */

async function probeVideo(file) {
  const { stdout } = await run('ffprobe', [
    '-v', 'error',
    '-show_entries', 'format=duration:stream=index,codec_type,width,height',
    '-of', 'json',
    file
  ]);

  const data = JSON.parse(stdout);
  const duration = Number(data?.format?.duration || 0);
  const video = (data?.streams || []).find(s => s.codec_type === 'video');

  if (!Number.isFinite(duration) || duration <= 0) {
    throw new Error('Video duration မဖတ်နိုင်ပါ');
  }

  return {
    duration,
    width: Number(video?.width || 0),
    height: Number(video?.height || 0)
  };
}

async function extractAudio(videoPath, jobFolder) {
  const audioPath = path.join(jobFolder, 'movie-audio.mp3');
  await run('ffmpeg', [
    '-y', '-i', videoPath,
    '-vn', '-map', '0:a:0?',
    '-ac', '1', '-ar', '16000',
    '-c:a', 'libmp3lame', '-b:a', '64k',
    audioPath
  ]);
  return audioPath;
}

async function splitAudio(audioPath, duration, jobFolder) {
  const chunksDir = path.join(jobFolder, 'whisper-chunks');
  fs.mkdirSync(chunksDir, { recursive: true });
  const chunks = [];

  for (let start = 0, index = 0; start < duration - 0.05; start += WHISPER_CHUNK_SECONDS, index++) {
    const seconds = Math.min(WHISPER_CHUNK_SECONDS, duration - start);
    const output = path.join(chunksDir, `chunk-${String(index).padStart(3, '0')}.mp3`);
    await run('ffmpeg', [
      '-y', '-ss', String(start), '-i', audioPath,
      '-t', String(seconds), '-ac', '1', '-ar', '16000',
      '-c:a', 'libmp3lame', '-b:a', '64k', output
    ]);
    chunks.push({ path: output, offset: start, duration: seconds });
  }

  return chunks;
}

/* =========================================================
   GROQ WHISPER TIMELINE
========================================================= */

async function transcribeGroq(chunks, jobId) {
  const groq = new Groq({ apiKey: requireEnv('GROQ_API_KEY') });
  const segments = [];

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    await updateJob(jobId, {
      stage: 'Whisper',
      progress: Math.round(8 + (i / Math.max(1, chunks.length)) * 27),
      message: `Groq Whisper ${i + 1}/${chunks.length} လုပ်နေပါတယ်...`
    });

    const result = await groq.audio.transcriptions.create({
      file: fs.createReadStream(chunk.path),
      model: GROQ_WHISPER_MODEL,
      response_format: 'verbose_json',
      timestamp_granularities: ['segment'],
      temperature: 0
    });

    const source = Array.isArray(result?.segments) ? result.segments : [];

    if (source.length) {
      for (const s of source) {
        const text = cleanText(s?.text);
        if (!text) continue;
        const localStart = Number(s?.start);
        const localEnd = Number(s?.end);
        segments.push({
          start: chunk.offset + (Number.isFinite(localStart) ? localStart : 0),
          end: chunk.offset + (Number.isFinite(localEnd) ? localEnd : Math.min(chunk.duration, 2)),
          text
        });
      }
    } else if (result?.text) {
      segments.push({
        start: chunk.offset,
        end: chunk.offset + chunk.duration,
        text: cleanText(result.text)
      });
    }
  }

  segments.sort((a, b) => a.start - b.start);

  const normalized = [];
  for (const s of segments) {
    const start = Math.max(0, Number(s.start) || 0);
    const end = Math.max(start + 0.05, Number(s.end) || start + 0.05);
    normalized.push({
      id: normalized.length + 1,
      start,
      end,
      text: s.text
    });
  }

  return {
    text: normalized.map(s => s.text).join(' ').trim(),
    segments: normalized
  };
}

/* =========================================================
   TIMELINE BLOCKS
   No Gemini video analysis here.
   Whisper timestamps are the master timeline.
========================================================= */

function buildTimelineBlocks(transcript, duration) {
  const segments = Array.isArray(transcript?.segments) ? transcript.segments : [];
  const blocks = [];

  for (let start = 0; start < duration - 0.05; start += RECAP_BLOCK_SECONDS) {
    const end = Math.min(duration, start + RECAP_BLOCK_SECONDS);
    const items = segments.filter(s => s.end > start && s.start < end);
    blocks.push({
      id: blocks.length + 1,
      start,
      end,
      duration: end - start,
      transcript: items.map(s => `[${s.start.toFixed(2)}-${s.end.toFixed(2)}] ${s.text}`).join('\n')
    });
  }

  // If the movie is short, keep one block.
  if (!blocks.length) {
    blocks.push({ id: 1, start: 0, end: duration, duration, transcript: transcript.text || '' });
  }

  return blocks;
}

/* =========================================================
   GROQ RECAP SCRIPT
   Generates narration per timeline block. Gemini is not used.
========================================================= */

async function generateRecapForBlock(groq, block, language, style) {
  const languageText = language === 'en' ? 'natural spoken English' : 'natural spoken Myanmar Burmese';
  const prompt = `
You are writing narration for ONE block of a movie recap video.

The video timeline for this block is exactly ${block.start.toFixed(2)}s to ${block.end.toFixed(2)}s.
The narration will be spoken over that exact video interval.

Write ${languageText}.
Style: ${styleInstruction(style)}.

Rules:
- Describe only story events supported by the supplied transcript.
- Keep the order of events exactly as the timestamps show.
- Do not invent characters, locations, actions or dialogue.
- Do not mention timestamps.
- Do not use headings, bullets or quotation marks.
- Make it sound like a human movie recap narrator.
- Keep the narration compact enough to fit the block naturally.
- Prefer about 2.2 to 2.8 spoken words per second for Myanmar; do not overstuff the block.
- If the transcript is silent or too short, use a brief neutral transition instead of inventing events.
- Output narration text only.

TIMELINE TRANSCRIPT:
${block.transcript || '(no speech in this interval)'}
`.trim();

  const result = await groq.chat.completions.create({
    model: GROQ_RECAP_MODEL,
    temperature: 0.2,
    max_tokens: 700,
    messages: [
      {
        role: 'system',
        content: 'You write accurate, compact movie recap narration. Never invent facts not supported by the supplied transcript.'
      },
      { role: 'user', content: prompt }
    ]
  });

  return cleanText(result?.choices?.[0]?.message?.content);
}

async function generateRecapBlocks(timelineBlocks, transcript, jobId, language, style) {
  const groq = new Groq({ apiKey: requireEnv('GROQ_API_KEY') });
  const blocks = [];

  for (let i = 0; i < timelineBlocks.length; i++) {
    const base = timelineBlocks[i];
    await updateJob(jobId, {
      stage: 'Recap Script',
      progress: Math.round(36 + (i / Math.max(1, timelineBlocks.length)) * 14),
      message: `Groq Recap ${i + 1}/${timelineBlocks.length} လုပ်နေပါတယ်...`
    });

    let narration = '';
    try {
      narration = await generateRecapForBlock(groq, base, language, style);
    } catch (error) {
      console.error(`[RECAP] Block ${base.id} failed:`, parseErrorMessage(error));
    }

    if (!narration) {
      const fallback = (transcript.segments || [])
        .filter(s => s.end > base.start && s.start < base.end)
        .map(s => s.text)
        .join(' ');
      narration = cleanText(fallback) || 'ဒီအပိုင်းမှာ ဇာတ်လမ်းက နောက်တစ်ဆင့်ကို ဆက်လက်ရွေ့လျားသွားပါတယ်။';
    }

    blocks.push({
      id: base.id,
      start: base.start,
      end: base.end,
      duration: base.duration,
      narration
    });
  }

  return blocks;
}

/* =========================================================
   GEMINI TTS
   TTS is chunked by timeline block, not by visual scene.
========================================================= */

async function generateGeminiTTS(text, apiKey, voiceName, style, language = 'my') {
  const ai = new GoogleGenAI({ apiKey });
  let lastError = null;

  for (let attempt = 0; attempt <= GEMINI_TTS_RETRIES; attempt++) {
    try {
      const response = await ai.models.generateContent({
        model: GEMINI_TTS_MODEL,
        contents: [{
          role: 'user',
          parts: [{
            text: `Speak this ${styleInstruction(style)} narration in ${language === 'en' ? 'natural English' : 'natural Myanmar Burmese'}. Do not add or remove words.\n\n${text}`
          }]
        }],
        config: {
          responseModalities: ['AUDIO'],
          responseFormat: {
            audio: {
              mimeType: 'AUDIO_L16',
              sampleRate: 24000
            }
          },
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: {
                voiceName
              }
            }
          }
        }
      });

      const part = response?.candidates?.[0]?.content?.parts?.find(p => p?.inlineData?.data);
      const base64 = part?.inlineData?.data;
      if (!base64) throw new Error('Gemini TTS audio မရပါ');
      return Buffer.from(base64, 'base64');
    } catch (error) {
      lastError = error;
      const status = Number(error?.status || error?.code || 0);
      const retryable = [429, 500, 502, 503, 504].includes(status) || /503|429|UNAVAILABLE|high demand|rate limit/i.test(parseErrorMessage(error));
      if (!retryable || attempt >= GEMINI_TTS_RETRIES) break;
      const delay = GEMINI_TTS_RETRY_DELAY * (attempt + 1);
      console.log(`[TTS] retry ${attempt + 1}/${GEMINI_TTS_RETRIES} after ${delay}ms`);
      await sleep(delay);
    }
  }

  throw new Error(`Gemini TTS failed: ${parseErrorMessage(lastError)}`);
}

async function pcmToWav(pcmBuffer, outputPath) {
  const pcmPath = `${outputPath}.pcm`;
  fs.writeFileSync(pcmPath, pcmBuffer);
  try {
    await run('ffmpeg', [
      '-y', '-f', 's16le', '-ar', '24000', '-ac', '1',
      '-i', pcmPath, '-c:a', 'pcm_s16le', outputPath
    ]);
  } finally {
    cleanup(pcmPath);
  }
}

async function getMediaDuration(file) {
  const { stdout } = await run('ffprobe', [
    '-v', 'error', '-show_entries', 'format=duration',
    '-of', 'default=noprint_wrappers=1:nokey=1', file
  ]);
  const value = Number(stdout.trim());
  if (!Number.isFinite(value) || value <= 0) throw new Error('Media duration မဖတ်နိုင်ပါ');
  return value;
}

function atempoFilters(factor) {
  let value = factor;
  const filters = [];
  while (value > 2) {
    filters.push('atempo=2');
    value /= 2;
  }
  while (value < 0.5) {
    filters.push('atempo=0.5');
    value /= 0.5;
  }
  if (Math.abs(value - 1) > 0.001) filters.push(`atempo=${value.toFixed(6)}`);
  return filters;
}

async function fitAudioToDuration(input, output, targetSeconds) {
  const source = await getMediaDuration(input);
  const target = Math.max(0.25, targetSeconds);
  const factor = source / target;
  const filters = atempoFilters(factor);
  filters.push(`apad=pad_dur=${target.toFixed(3)}`);
  filters.push(`atrim=0:${target.toFixed(3)}`);
  filters.push('asetpts=N/SR/TB');

  await run('ffmpeg', [
    '-y', '-i', input,
    '-filter:a', filters.join(','),
    '-ar', '24000', '-ac', '1',
    '-c:a', 'pcm_s16le', output
  ]);
}

/* =========================================================
   FINAL TIMELINE AUDIO
========================================================= */

async function buildNarrationAudio(jobId, blocks, voice, style, language, jobFolder) {
  const geminiKey = requireEnv('GEMINI_API_KEY');
  const voiceName = resolveVoice(voice);
  const rendered = [];

  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i];
    const rawPcm = path.join(jobFolder, `tts-${block.id}.pcm`);
    const rawWav = path.join(jobFolder, `tts-${block.id}.wav`);
    const fitWav = path.join(jobFolder, `tts-${block.id}-fit.wav`);

    await updateJob(jobId, {
      stage: 'Voice',
      progress: Math.round(50 + (i / Math.max(1, blocks.length)) * 25),
      message: `Myanmar Voice ${i + 1}/${blocks.length} ထုတ်နေပါတယ်...`
    });

    const pcm = await generateGeminiTTS(block.narration, geminiKey, voiceName, style, language);
    fs.writeFileSync(rawPcm, pcm);
    await pcmToWav(pcm, rawWav);

    // Fit exactly to this timeline block. The block start is the master clock.
    await fitAudioToDuration(rawWav, fitWav, block.duration);
    rendered.push({ ...block, audioPath: fitWav });
  }

  const narrationAudio = path.join(jobFolder, 'narration-timeline.wav');
  const inputs = [];
  const filters = [];

  for (let i = 0; i < rendered.length; i++) {
    inputs.push('-i', rendered[i].audioPath);
    filters.push(`[${i}:a]adelay=${Math.round(rendered[i].start * 1000)}|${Math.round(rendered[i].start * 1000)},apad,atrim=0:${rendered[rendered.length - 1].end.toFixed(3)}[a${i}]`);
  }

  const mixInputs = rendered.map((_, i) => `[a${i}]`).join('');
  filters.push(`${mixInputs}amix=inputs=${rendered.length}:duration=longest:dropout_transition=0,atrim=0:${rendered[rendered.length - 1].end.toFixed(3)},asetpts=N/SR/TB[out]`);

  await run('ffmpeg', [
    '-y', ...inputs,
    '-filter_complex', filters.join(';'),
    '-map', '[out]',
    '-ar', '24000', '-ac', '1',
    '-c:a', 'pcm_s16le', narrationAudio
  ]);

  return narrationAudio;
}

/* =========================================================
   FINAL VIDEO
   Keeps the original video timeline intact and replaces/overlays
   its audio with the synchronized Myanmar narration timeline.
========================================================= */

async function renderFinalVideo(moviePath, narrationAudio, outputPath) {
  await run('ffmpeg', [
    '-y',
    '-i', moviePath,
    '-i', narrationAudio,
    '-map', '0:v:0',
    '-map', '1:a:0',
    '-c:v', 'libx264',
    '-preset', 'veryfast',
    '-crf', '20',
    '-pix_fmt', 'yuv420p',
    '-c:a', 'aac',
    '-b:a', '160k',
    '-ar', '24000',
    '-ac', '1',
    '-shortest',
    '-movflags', '+faststart',
    outputPath
  ]);
}

/* =========================================================
   PROCESS ONE CLIP
========================================================= */

async function processOneClip(jobId, moviePath, originalFilename, options) {
  const jobFolder = path.join(JOB_DIR, jobId);
  const tempFiles = [];

  try {
    const language = options.language || 'my';
    const style = options.style || 'cinematic';
    const voice = options.voice || 'female';

    requireEnv('GROQ_API_KEY');
    requireEnv('GEMINI_API_KEY');

    await updateJob(jobId, {
      status: 'processing',
      stage: 'Upload',
      progress: 3,
      message: 'Movie uploaded. Processing started.'
    });

    const info = await probeVideo(moviePath);
    if (info.duration > MAX_VIDEO_SECONDS) {
      throw new Error('Video က 5 မိနစ်ထက် မကျော်ရပါ');
    }

    await updateJob(jobId, {
      originalFilename,
      duration: info.duration,
      stage: 'FFmpeg',
      progress: 5,
      message: 'Movie audio ကို ထုတ်ယူနေပါတယ်...'
    });

    const audioPath = await extractAudio(moviePath, jobFolder);
    const chunks = await splitAudio(audioPath, info.duration, jobFolder);
    await updateJob(jobId, { totalChunks: chunks.length });

    const transcript = await transcribeGroq(chunks, jobId);
    if (!transcript.segments.length && !transcript.text) {
      throw new Error('Groq Whisper က Transcript မထုတ်ပေးနိုင်ပါ');
    }

    await fsp.writeFile(
      path.join(jobFolder, 'transcript.json'),
      JSON.stringify(transcript, null, 2),
      'utf8'
    );

    await updateJob(jobId, {
      transcript: {
        text: transcript.text,
        segments: transcript.segments
      },
      stage: 'Transcript',
      progress: 35,
      message: `${transcript.segments.length} timestamp segments ရပါပြီ။`
    });

    // Timeline comes ONLY from Whisper timestamps. No Gemini video analysis.
    const timelineBlocks = buildTimelineBlocks(transcript, info.duration);

    await updateJob(jobId, {
      stage: 'Recap Script',
      progress: 37,
      message: `${timelineBlocks.length} timeline block(s) အတွက် Recap Script ပြုလုပ်နေပါတယ်...`
    });

    const recapBlocks = await generateRecapBlocks(
      timelineBlocks,
      transcript,
      jobId,
      language,
      style
    );

    await fsp.writeFile(
      path.join(jobFolder, 'recap-blocks.json'),
      JSON.stringify(recapBlocks, null, 2),
      'utf8'
    );

    await updateJob(jobId, {
      recapBlocks,
      stage: 'Voice',
      progress: 50,
      message: `${recapBlocks.length} timeline block(s) အတွက် Myanmar Voice ထုတ်နေပါတယ်...`
    });

    const narrationAudio = await buildNarrationAudio(
      jobId,
      recapBlocks,
      voice,
      style,
      language,
      jobFolder
    );

    await updateJob(jobId, {
      stage: 'FFmpeg',
      progress: 78,
      message: 'Myanmar Voice နဲ့ Movie Timeline ကို exact timing နဲ့ပေါင်းနေပါတယ်...'
    });

    const outputPath = path.join(jobFolder, 'final-recap.mp4');
    await renderFinalVideo(moviePath, narrationAudio, outputPath);

    if (!fs.existsSync(outputPath)) throw new Error('FFmpeg final MP4 မထုတ်ပေးနိုင်ပါ');
    const stats = fs.statSync(outputPath);
    if (stats.size <= 0) throw new Error('Final MP4 ဖိုင်အရွယ်အစား 0 ဖြစ်နေပါတယ်');

    const filename = `${path.parse(originalFilename || 'movie').name}-Myanmar-Recap.mp4`;

    await updateJob(jobId, {
      status: 'completed',
      stage: 'Ready',
      progress: 100,
      message: 'Final Recap Video ရပါပြီ။',
      output: {
        path: outputPath,
        filename,
        size: stats.size,
        url: `/api/download/${jobId}`
      },
      error: null
    });

    console.log(`[JOB ${jobId}] COMPLETED`);
  } catch (error) {
    const message = parseErrorMessage(error);
    console.error(`[JOB ${jobId}] FAILED:`, message);
    await updateJob(jobId, {
      status: 'error',
      stage: 'Error',
      progress: 0,
      message: 'Movie processing မအောင်မြင်ပါ။',
      error: message
    });
  } finally {
    // Keep final-recap.mp4 and job metadata. Remove only the uploaded source file.
    cleanup(moviePath);
    cleanupMany(tempFiles);
  }
}

/* =========================================================
   API
========================================================= */

app.get('/api/health', async (_req, res) => {
  res.json({
    ok: true,
    name: 'YNT One Clips',
    status: 'online',
    models: {
      whisper: GROQ_WHISPER_MODEL,
      recap: GROQ_RECAP_MODEL,
      tts: GEMINI_TTS_MODEL
    },
    timeline: 'Whisper timestamps',
    geminiSceneAnalysis: false,
    ttsBlocks: RECAP_BLOCK_SECONDS,
    maxVideoSeconds: MAX_VIDEO_SECONDS,
    maxVideoSizeMB: Math.round(MAX_VIDEO_SIZE / 1024 / 1024),
    storage: 'PostgreSQL'
  });
});

app.get('/health', (_req, res) => {
  res.json({ ok: true, name: 'YNT One Clips', status: 'online' });
});

app.get('/api/status/:jobId', async (req, res) => {
  try {
    const job = await getJob(req.params.jobId);
    if (!job) return res.status(404).json({ ok: false, error: 'Job not found', jobId: req.params.jobId });
    // Return both shapes so the existing HTML does not need to change.
    return res.json({ ok: true, ...job, job });
  } catch (error) {
    return res.status(500).json({ ok: false, error: parseErrorMessage(error) });
  }
});

app.get('/api/download/:jobId', async (req, res) => {
  try {
    const job = await getJob(req.params.jobId);
    if (!job) return res.status(404).send('Job not found');
    if (job.status !== 'completed' || !job.output?.path) return res.status(404).send('Final Video မရသေးပါ');
    if (!fs.existsSync(job.output.path)) return res.status(404).send('Final Video ဖိုင် မတွေ့ပါ။ Render filesystem ပြန်စတင်ထားနိုင်ပါတယ်။');
    return res.download(job.output.path, job.output.filename || 'YNT-One-Clips.mp4');
  } catch (error) {
    return res.status(500).send(parseErrorMessage(error));
  }
});

// IMPORTANT: the existing HTML sends multipart field name "movie".
app.post('/api/one-clip', upload.single('movie'), async (req, res) => {
  let filePath = req.file?.path;

  try {
    if (!req.file) return res.status(400).json({ ok: false, error: 'Movie file မရှိပါ' });

    const originalName = req.file.originalname || 'movie.mp4';
    const job = await createJob(originalName);
    const jobFolder = path.join(JOB_DIR, job.id);
    fs.mkdirSync(jobFolder, { recursive: true });

    const ext0 = path.extname(originalName).toLowerCase();
    const ext = allowedExt.has(ext0) ? ext0 : '.mp4';
    const moviePath = path.join(jobFolder, `movie${ext}`);
    fs.renameSync(filePath, moviePath);
    filePath = null;

    const options = {
      language: String(req.body?.language || 'my'),
      style: String(req.body?.style || 'cinematic'),
      voice: String(req.body?.voice || 'female')
    };

    await updateJob(job.id, {
      status: 'processing',
      stage: 'Upload',
      progress: 3,
      message: 'Movie uploaded. Processing started.'
    });

    console.log('============================================');
    console.log(`[JOB ${job.id}] STARTED`);
    console.log(`Language: ${options.language}`);
    console.log(`Style: ${options.style}`);
    console.log(`Voice: ${options.voice}`);
    console.log('Timeline: Whisper timestamps');
    console.log('Gemini Scene Analysis: DISABLED');
    console.log(`Groq Recap: ${GROQ_RECAP_MODEL}`);
    console.log(`Gemini TTS: ${GEMINI_TTS_MODEL}`);
    console.log(`TTS Timeline Blocks: ${RECAP_BLOCK_SECONDS}s`);
    console.log('PostgreSQL Job Storage: READY');
    console.log('============================================');

    res.status(202).json({
      ok: true,
      success: true,
      jobId: job.id,
      status: 'processing',
      statusUrl: `/api/status/${job.id}`,
      downloadUrl: `/api/download/${job.id}`
    });

    setImmediate(() => {
      processOneClip(job.id, moviePath, originalName, options).catch(async error => {
        const message = parseErrorMessage(error);
        console.error(`[JOB ${job.id}] UNHANDLED:`, message);
        try {
          await updateJob(job.id, {
            status: 'error',
            stage: 'Error',
            progress: 0,
            message: 'Background processing မအောင်မြင်ပါ။',
            error: message
          });
        } catch (dbError) {
          console.error('[JOB] Failed to write error to DB:', dbError);
        }
      });
    });
  } catch (error) {
    cleanup(filePath);
    console.error('ONE CLIP START ERROR:', error);
    return res.status(400).json({ ok: false, error: parseErrorMessage(error) });
  }
});

app.use('/api', (_req, res) => {
  res.status(404).json({ ok: false, error: 'API endpoint မတွေ့ပါ' });
});

app.use((error, _req, res, _next) => {
  console.error('SERVER ERROR:', error);
  if (res.headersSent) return;
  if (error?.code === 'LIMIT_FILE_SIZE') {
    return res.status(413).json({ ok: false, error: `Video size က ${Math.round(MAX_VIDEO_SIZE / 1024 / 1024)}MB ထက် မကျော်ရပါ` });
  }
  return res.status(500).json({ ok: false, error: parseErrorMessage(error) });
});

/* =========================================================
   START
========================================================= */

async function startServer() {
  await initDatabase();

  app.listen(PORT, '0.0.0.0', () => {
    console.log('============================================');
    console.log('YNT One Clips server started');
    console.log(`PORT: ${PORT}`);
    console.log(`Groq Whisper: ${GROQ_WHISPER_MODEL}`);
    console.log(`Groq Recap: ${GROQ_RECAP_MODEL}`);
    console.log(`Gemini TTS: ${GEMINI_TTS_MODEL}`);
    console.log('Timeline Sync: ON (Whisper timestamps)');
    console.log(`Timeline Blocks: ${RECAP_BLOCK_SECONDS}s`);
    console.log('Gemini Scene Analysis: DISABLED');
    console.log(`Gemini TTS Retries: ${GEMINI_TTS_RETRIES}`);
    console.log('PostgreSQL Job Storage: READY');
    console.log('FFmpeg Final Timeline Renderer: READY');
    console.log('============================================');
  });
}

startServer().catch(error => {
  console.error('SERVER START FAILED:', parseErrorMessage(error));
  process.exit(1);
});
