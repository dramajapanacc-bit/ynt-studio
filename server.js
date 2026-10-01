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

const execFileAsync = promisify(execFile);
const app = express();

const PORT = Number(process.env.PORT || 3000);
const ROOT = process.cwd();
const PUBLIC_DIR = path.join(ROOT, 'public');
const UPLOAD_DIR = path.join(ROOT, 'uploads');
const JOB_DIR = path.join(ROOT, 'jobs');

const MAX_VIDEO_SIZE = 500 * 1024 * 1024;
const MAX_VIDEO_SECONDS = 5 * 60;
const AUDIO_CHUNK_SECONDS = 90;
const SCENE_FPS = 2;
const MAX_SCENES = 6;
const MIN_SCENE_SECONDS = 3;

const GROQ_MODEL = 'whisper-large-v3-turbo';
const GEMINI_MODEL = 'gemini-3.8-flash';
const GEMINI_TTS_MODEL = 'gemini-3.8-flash-tts';
const GEMINI_MAX_RETRIES = 0;
const GEMINI_INITIAL_RETRY_DELAY = 2000;

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
app.use(express.static(PUBLIC_DIR));

/* =========================================================
   FILE UPLOAD
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
    if (!allowedExt.has(ext)) {
      return cb(new Error('MP4 / MOV / MKV / WEBM / AVI video ကိုသုံးပါ။'));
    }
    cb(null, true);
  }
});

/* =========================================================
   JOB STORAGE

   This fixes the immediate "Job not found" problem caused by
   an in-memory Map. Job metadata is also written to jobs/<id>/job.json.

   Important: Render Free still has an ephemeral filesystem, so this
   cannot make local files survive a Render redeploy/spin-down/restart.
========================================================= */

const jobs = new Map();

function getJobFolder(jobId) {
  return path.join(JOB_DIR, jobId);
}

function getJobFile(jobId) {
  return path.join(getJobFolder(jobId), 'job.json');
}

function saveJob(job) {
  try {
    const folder = getJobFolder(job.id);
    fs.mkdirSync(folder, { recursive: true });

    const tempFile = path.join(folder, `job-${process.pid}.tmp`);
    fs.writeFileSync(tempFile, JSON.stringify(job, null, 2), 'utf8');
    fs.renameSync(tempFile, getJobFile(job.id));
  } catch (error) {
    console.error(`[JOB STORAGE] Failed to save ${job?.id}:`, error?.message || error);
  }
}

function loadJob(jobId) {
  const file = getJobFile(jobId);
  if (!fs.existsSync(file)) return null;

  try {
    const job = JSON.parse(fs.readFileSync(file, 'utf8'));
    jobs.set(job.id, job);
    return job;
  } catch (error) {
    console.error(`[JOB STORAGE] Failed to load ${jobId}:`, error?.message || error);
    return null;
  }
}

function loadAllJobs() {
  if (!fs.existsSync(JOB_DIR)) return;

  const entries = fs.readdirSync(JOB_DIR, { withFileTypes: true });
  let restored = 0;

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const job = loadJob(entry.name);
    if (job) restored++;
  }

  console.log(`[JOB STORAGE] Restored ${restored} job(s) from disk.`);
}

function createJob() {
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
    scenePlan: null,
    output: null,
    error: null
  };

  jobs.set(id, job);
  saveJob(job);
  return job;
}

function updateJob(id, data) {
  let job = jobs.get(id);
  if (!job) job = loadJob(id);

  if (!job) {
    console.error(`[JOB STORAGE] Job ${id} not found while updating.`);
    return null;
  }

  Object.assign(job, data, {
    updatedAt: new Date().toISOString()
  });

  jobs.set(id, job);
  saveJob(job);
  return job;
}

function getJob(id) {
  return jobs.get(id) || loadJob(id);
}

loadAllJobs();

/* =========================================================
   BASIC HELPERS
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

function safeJsonParse(text) {
  if (typeof text !== 'string') return text;

  const trimmed = text.trim();
  if (!trimmed) return null;

  try {
    return JSON.parse(trimmed);
  } catch {}

  const fenced = trimmed
    .replace(/^```json\s*/i, '')
    .replace(/^```\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();

  try {
    return JSON.parse(fenced);
  } catch {}

  const first = Math.min(
    ...['[', '{'].map(ch => {
      const i = fenced.indexOf(ch);
      return i >= 0 ? i : Number.POSITIVE_INFINITY;
    })
  );

  if (Number.isFinite(first)) {
    const lastArray = fenced.lastIndexOf(']');
    const lastObject = fenced.lastIndexOf('}');
    const last = Math.max(lastArray, lastObject);

    if (last > first) {
      try {
        return JSON.parse(fenced.slice(first, last + 1));
      } catch {}
    }
  }

  return null;
}

function parseErrorMessage(error) {
  return String(
    error?.message ||
    error?.error?.message ||
    error ||
    'Unknown error'
  );
}

function isRateLimitError(error) {
  const message = parseErrorMessage(error).toLowerCase();
  return (
    message.includes('429') ||
    message.includes('rate limit') ||
    message.includes('resource exhausted') ||
    message.includes('quota')
  );
}

function isRetryableGeminiError(error) {
  const message = parseErrorMessage(error).toLowerCase();
  return (
    message.includes('429') ||
    message.includes('500') ||
    message.includes('502') ||
    message.includes('503') ||
    message.includes('504') ||
    message.includes('temporarily unavailable') ||
    message.includes('internal error')
  );
}

async function withGeminiRetry(fn, label) {
  let lastError = null;

  for (let attempt = 0; attempt <= GEMINI_MAX_RETRIES; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (attempt >= GEMINI_MAX_RETRIES || !isRetryableGeminiError(error)) {
        const message = parseErrorMessage(error);
        if (isRateLimitError(error)) {
          throw new Error(`${label} failed: Gemini quota/rate limit. ${message}`);
        }
        throw new Error(`${label} failed: ${message}`);
      }

      const delay = GEMINI_INITIAL_RETRY_DELAY * (attempt + 1);
      console.log(`[GEMINI] ${label} retry ${attempt + 1} after ${delay}ms`);
      await sleep(delay);
    }
  }

  throw lastError || new Error(`${label} failed.`);
}

function getApiKey(req, headerName, bodyName, envName, label) {
  const value = String(
    req.get(headerName) ||
    req.body?.[bodyName] ||
    process.env[envName] ||
    ''
  ).trim();

  if (!value) throw new Error(`${label} မရှိပါ`);
  return value;
}

function run(command, args, options = {}) {
  return execFileAsync(command, args, {
    maxBuffer: 50 * 1024 * 1024,
    ...options
  });
}

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

async function extractAudio(videoPath, jobId) {
  const audioPath = path.join(UPLOAD_DIR, `${jobId}-audio.mp3`);

  await run('ffmpeg', [
    '-y',
    '-i', videoPath,
    '-vn',
    '-ac', '1',
    '-ar', '16000',
    '-codec:a', 'libmp3lame',
    '-b:a', '64k',
    audioPath
  ]);

  return audioPath;
}

async function splitAudio(audioPath, duration, jobId) {
  const chunks = [];

  for (let start = 0, index = 0; start < duration - 0.05; start += AUDIO_CHUNK_SECONDS, index++) {
    const seconds = Math.min(AUDIO_CHUNK_SECONDS, duration - start);
    const chunkPath = path.join(UPLOAD_DIR, `${jobId}-chunk-${index}.mp3`);

    await run('ffmpeg', [
      '-y',
      '-ss', String(start),
      '-i', audioPath,
      '-t', String(seconds),
      '-ac', '1',
      '-ar', '16000',
      '-codec:a', 'libmp3lame',
      '-b:a', '64k',
      chunkPath
    ]);

    chunks.push({ path: chunkPath, offset: start, duration: seconds });
  }

  return chunks;
}

/* =========================================================
   GROQ WHISPER
========================================================= */

async function transcribeGroqChunks(chunks, apiKey, jobId) {
  const groq = new Groq({ apiKey });
  const all = [];

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];

    updateJob(jobId, {
      status: 'processing',
      stage: 'Groq Whisper',
      progress: Math.round(8 + (i / Math.max(1, chunks.length)) * 22),
      message: `Groq Whisper ${i + 1}/${chunks.length} လုပ်နေပါတယ်...`
    });

    const result = await groq.audio.transcriptions.create({
      file: fs.createReadStream(chunk.path),
      model: GROQ_MODEL,
      response_format: 'verbose_json',
      timestamp_granularities: ['segment'],
      language: 'my'
    });

    const segments = Array.isArray(result?.segments) ? result.segments : [];

    if (segments.length) {
      for (const segment of segments) {
        const start = Number(segment?.start);
        const end = Number(segment?.end);
        const text = cleanText(segment?.text);

        if (!text) continue;

        all.push({
          id: all.length + 1,
          start: Number.isFinite(start) ? start + chunk.offset : chunk.offset,
          end: Number.isFinite(end) ? end + chunk.offset : chunk.offset + 1,
          text
        });
      }
    } else if (result?.text) {
      all.push({
        id: all.length + 1,
        start: chunk.offset,
        end: chunk.offset + chunk.duration,
        text: cleanText(result.text)
      });
    }
  }

  all.sort((a, b) => a.start - b.start);

  return all.map((item, index) => ({
    id: index + 1,
    start: Math.max(0, item.start),
    end: Math.max(item.start + 0.05, item.end),
    text: item.text
  }));
}

/* =========================================================
   TIMELINE + GROQ RECAP

   Gemini is intentionally NOT used for movie/scene analysis.
   Whisper supplies the real speech timeline. We group that timeline
   into a small number of contiguous blocks and ask Groq to write
   Myanmar recap narration for each block. The same selected Gemini
   voice is then used for every narration block.
========================================================= */

function buildTimeline(transcript, duration) {
  const source = Array.isArray(transcript) ? transcript : [];
  const usable = source
    .map(item => ({
      start: Math.max(0, Number(item?.start) || 0),
      end: Math.min(duration, Math.max(0, Number(item?.end) || 0)),
      text: cleanText(item?.text)
    }))
    .filter(item => item.text && item.end > item.start);

  if (!usable.length) {
    return [{ start: 0, end: duration, transcript: '' }];
  }

  const blocks = [];
  const target = Math.max(45, Math.min(90, duration / 5));
  let current = { start: 0, end: 0, items: [] };

  for (const item of usable) {
    if (!current.items.length) {
      current.start = Math.max(0, Math.min(duration, item.start));
      current.end = item.end;
      current.items.push(item);
      continue;
    }

    const nextEnd = Math.max(current.end, item.end);
    const currentLength = nextEnd - current.start;

    if (currentLength <= target || current.items.length < 2) {
      current.end = nextEnd;
      current.items.push(item);
    } else {
      blocks.push(current);
      current = { start: item.start, end: item.end, items: [item] };
    }
  }

  if (current.items.length) blocks.push(current);

  // Force complete 0 -> duration coverage without gaps.
  const timeline = [];
  let cursor = 0;

  for (const block of blocks) {
    const start = cursor;
    const end = Math.min(duration, Math.max(start, block.end));
    if (end <= start) continue;

    timeline.push({
      start,
      end,
      transcript: block.items.map(x => `[${x.start.toFixed(2)}-${x.end.toFixed(2)}] ${x.text}`).join(' ')
    });
    cursor = end;
  }

  if (!timeline.length) {
    return [{ start: 0, end: duration, transcript: usable.map(x => x.text).join(' ') }];
  }

  if (timeline[0].start > 0) timeline[0].start = 0;
  timeline[timeline.length - 1].end = duration;

  for (let i = 1; i < timeline.length; i++) {
    timeline[i].start = timeline[i - 1].end;
  }

  return timeline.map((item, index) => ({
    index: index + 1,
    start: item.start,
    end: item.end,
    duration: Math.max(0.1, item.end - item.start),
    transcript: item.transcript
  }));
}

function buildGroqRecapPrompt(timeline, language, style) {
  const languageInstruction = language === 'my'
    ? 'Write natural Myanmar Burmese only.'
    : `Write naturally in ${language}.`;

  const styleInstruction = narrationStyle(style);

  const blocks = timeline.map(block =>
    `BLOCK ${block.index} | ${block.start.toFixed(2)}s - ${block.end.toFixed(2)}s\n${block.transcript}`
  ).join('\n\n');

  return `You are a professional movie recap narrator and timeline editor.

${languageInstruction}
Style: ${styleInstruction}

Create a concise recap narration for EVERY timeline block below.
The narration must describe only events supported by that block's transcript/context.
Keep the story flowing naturally from one block to the next.
Do not mention timestamps, blocks, AI, prompts, or analysis.
Do not use markdown.
Return ONLY valid JSON.

JSON format:
{
  "blocks": [
    {
      "index": 1,
      "narration": "..."
    }
  ]
}

TIMELINE BLOCKS:
${blocks}`;
}

async function generateGroqRecap(timeline, apiKey, language, style) {
  const groq = new Groq({ apiKey });
  const model = process.env.GROQ_RECAP_MODEL || 'openai/gpt-oss-120b';
  const prompt = buildGroqRecapPrompt(timeline, language, style);

  let response;
  try {
    response = await groq.chat.completions.create({
      model,
      temperature: 0.35,
      response_format: { type: 'json_object' },
      messages: [
        { role: 'system', content: 'Return JSON only.' },
        { role: 'user', content: prompt }
      ]
    });
  } catch (error) {
    console.log('[GROQ RECAP] JSON mode retry:', parseErrorMessage(error));
    response = await groq.chat.completions.create({
      model,
      temperature: 0.35,
      messages: [
        { role: 'system', content: 'Return valid JSON only.' },
        { role: 'user', content: prompt }
      ]
    });
  }

  const content = response?.choices?.[0]?.message?.content || '';
  const parsed = safeJsonParse(content);
  const blocks = Array.isArray(parsed) ? parsed : parsed?.blocks;
  const byIndex = new Map(
    (Array.isArray(blocks) ? blocks : [])
      .map(item => [Number(item?.index), cleanText(item?.narration)])
      .filter(([index, narration]) => Number.isFinite(index) && narration)
  );

  return timeline.map(block => ({
    ...block,
    narration: byIndex.get(block.index) || ''
  }));
}

/* =========================================================
   GEMINI TTS
========================================================= */



function resolveVoice(value) {
  const key = String(value || 'female').trim().toLowerCase();
  return GEMINI_VOICES[key] || 'Kore';
}

function narrationStyle(style) {
  const value = String(style || 'cinematic').toLowerCase();

  if (value.includes('dramatic')) return 'dramatic, emotional Myanmar movie recap narration';
  if (value.includes('fun')) return 'energetic but natural Myanmar movie recap narration';
  if (value.includes('serious')) return 'serious, calm Myanmar movie recap narration';
  return 'cinematic, natural Myanmar movie recap narration';
}

async function generateGeminiTTS(text, apiKey, voiceName, style) {
  const ai = new GoogleGenAI({ apiKey });

  const response = await withGeminiRetry(
    () => ai.models.generateContent({
      model: GEMINI_TTS_MODEL,
      contents: [{
        role: 'user',
        parts: [{
          text: `${narrationStyle(style)}. Speak clearly in natural Myanmar Burmese.\n\n${text}`,
          speechMetadata: {
            style: narrationStyle(style)
          }
        }]
      }],
      config: {
        responseModalities: ['AUDIO'],
        speechConfig: {
          voiceConfig: {
            voice: voiceName
          }
        }
      }
    }),
    'Gemini TTS'
  );

  const part = response?.candidates?.[0]?.content?.parts?.find(
    p => p?.inlineData?.data
  );

  const base64 = part?.inlineData?.data;
  if (!base64) throw new Error('Gemini TTS audio မရပါ');

  return Buffer.from(base64, 'base64');
}

async function pcmToWav(pcmBuffer, outputPath) {
  const pcmPath = `${outputPath}.pcm`;
  fs.writeFileSync(pcmPath, pcmBuffer);

  try {
    await run('ffmpeg', [
      '-y',
      '-f', 's16le',
      '-ar', '24000',
      '-ac', '1',
      '-i', pcmPath,
      '-c:a', 'pcm_s16le',
      outputPath
    ]);
  } finally {
    cleanup(pcmPath);
  }
}

async function mediaDuration(file) {
  const { stdout } = await run('ffprobe', [
    '-v', 'error',
    '-show_entries', 'format=duration',
    '-of', 'default=noprint_wrappers=1:nokey=1',
    file
  ]);

  const duration = Number(stdout.trim());
  if (!Number.isFinite(duration) || duration <= 0) {
    throw new Error(`Audio duration မဖတ်နိုင်ပါ: ${path.basename(file)}`);
  }
  return duration;
}

function atempoChain(factor) {
  let value = factor;
  const filters = [];

  while (value > 2.0) {
    filters.push('atempo=2.0');
    value /= 2.0;
  }

  while (value < 0.5) {
    filters.push('atempo=0.5');
    value /= 0.5;
  }

  if (Math.abs(value - 1) > 0.001) {
    filters.push(`atempo=${value.toFixed(6)}`);
  }

  return filters;
}

async function fitAudioToScene(inputAudio, outputAudio, targetDuration) {
  const sourceDuration = await mediaDuration(inputAudio);
  const target = Math.max(0.25, targetDuration);

  if (sourceDuration <= 0) throw new Error('TTS audio duration မမှန်ပါ');

  const factor = sourceDuration / target;
  const filters = atempoChain(factor);

  if (filters.length) {
    filters.push(`apad=pad_dur=${target.toFixed(3)}`);
  } else {
    filters.push(`apad=pad_dur=${target.toFixed(3)}`);
  }

  filters.push(`atrim=0:${target.toFixed(3)}`);
  filters.push('asetpts=N/SR/TB');

  await run('ffmpeg', [
    '-y',
    '-i', inputAudio,
    '-filter:a', filters.join(','),
    '-ar', '24000',
    '-ac', '1',
    '-c:a', 'pcm_s16le',
    outputAudio
  ]);
}

/* =========================================================
   SCENE VIDEO RENDERING
========================================================= */

async function renderSceneVideo(moviePath, scene, outputPath) {
  const duration = Math.max(0.1, scene.end - scene.start);

  await run('ffmpeg', [
    '-y',
    '-i', moviePath,
    '-ss', scene.start.toFixed(3),
    '-t', duration.toFixed(3),
    '-an',
    '-vf', `fps=${SCENE_FPS},format=yuv420p`,
    '-c:v', 'libx264',
    '-preset', 'veryfast',
    '-crf', '20',
    '-pix_fmt', 'yuv420p',
    '-r', String(SCENE_FPS),
    '-movflags', '+faststart',
    outputPath
  ]);
}

async function muxScene(sceneVideo, sceneAudio, outputPath) {
  await run('ffmpeg', [
    '-y',
    '-i', sceneVideo,
    '-i', sceneAudio,
    '-map', '0:v:0',
    '-map', '1:a:0',
    '-c:v', 'copy',
    '-c:a', 'aac',
    '-b:a', '160k',
    '-ar', '24000',
    '-ac', '1',
    '-shortest',
    '-movflags', '+faststart',
    outputPath
  ]);
}

async function concatScenes(sceneFiles, outputPath, jobId) {
  const listPath = path.join(getJobFolder(jobId), 'concat.txt');
  const lines = sceneFiles.map(file => `file '${file.replace(/'/g, "'\\''")}'`);
  await fsp.writeFile(listPath, `${lines.join('\n')}\n`, 'utf8');

  try {
    await run('ffmpeg', [
      '-y',
      '-f', 'concat',
      '-safe', '0',
      '-i', listPath,
      '-c', 'copy',
      '-movflags', '+faststart',
      outputPath
    ]);
  } finally {
    cleanup(listPath);
  }
}

/* =========================================================
   ONE-CLIP PIPELINE
========================================================= */

async function processOneClip(jobId, videoPath, originalFilename, options) {
  const jobFolder = getJobFolder(jobId);
  fs.mkdirSync(jobFolder, { recursive: true });

  const tempFiles = [];

  try {
    const language = options.language || 'my';
    const style = options.style || 'cinematic';
    const voice = resolveVoice(options.voice || 'female');
    const geminiKey = process.env.GEMINI_API_KEY;
    const groqKey = process.env.GROQ_API_KEY;

    if (!groqKey) throw new Error('GROQ_API_KEY မရှိပါ');
    if (!geminiKey) throw new Error('GEMINI_API_KEY မရှိပါ');

    updateJob(jobId, {
      status: 'processing',
      stage: 'Video Probe',
      progress: 3,
      message: 'Video ကို စစ်ဆေးနေပါတယ်...'
    });

    const info = await probeVideo(videoPath);

    if (info.duration > MAX_VIDEO_SECONDS) {
      throw new Error('Video က 5 မိနစ်ထက် မကျော်ရပါ');
    }

    updateJob(jobId, {
      originalFilename,
      duration: info.duration,
      stage: 'Audio Extract',
      progress: 5,
      message: 'Video အသံကို ထုတ်ယူနေပါတယ်...'
    });

    const audioPath = await extractAudio(videoPath, jobId);
    tempFiles.push(audioPath);

    const chunks = await splitAudio(audioPath, info.duration, jobId);
    tempFiles.push(...chunks.map(x => x.path));

    updateJob(jobId, {
      totalChunks: chunks.length,
      stage: 'Groq Whisper',
      progress: 8,
      message: `Groq Whisper ${chunks.length} chunk(s) လုပ်နေပါတယ်...`
    });

    const transcript = await transcribeGroqChunks(chunks, groqKey, jobId);

    if (!transcript.length) {
      throw new Error('Groq က စကားပြော Transcript မထုတ်ပေးနိုင်ပါ');
    }

    // Whisper timestamps become the master timeline.
    updateJob(jobId, {
      transcript,
      stage: 'Timeline',
      progress: 34,
      message: 'Video အဖြစ်အပျက် Timeline ကို ချိန်ညှိနေပါတယ်...'
    });

    const timeline = buildTimeline(transcript, info.duration);
    fs.writeFileSync(
      path.join(jobFolder, 'timeline.json'),
      JSON.stringify(timeline, null, 2),
      'utf8'
    );

    updateJob(jobId, {
      scenePlan: timeline,
      stage: 'Groq Recap',
      progress: 40,
      message: `Timeline ${timeline.length} ပိုင်းအတွက် Myanmar Recap စာရေးနေပါတယ်...`
    });

    const recap = await generateGroqRecap(timeline, groqKey, language, style);

    // If Groq misses a block, use the source transcript instead of stopping the job.
    for (const block of recap) {
      if (!block.narration) {
        block.narration = cleanText(block.transcript) || 'ဒီအပိုင်းမှာ ဇာတ်လမ်းက ဆက်လက်ဖြစ်ပျက်နေပါတယ်။';
      }
    }

    fs.writeFileSync(
      path.join(jobFolder, 'recap-timeline.json'),
      JSON.stringify(recap, null, 2),
      'utf8'
    );

    updateJob(jobId, {
      scenePlan: recap,
      stage: 'Gemini TTS',
      progress: 48,
      message: `Myanmar Voice တစ်သံတည်းနဲ့ Timeline ${recap.length} ပိုင်းကို အသံထုတ်နေပါတယ်...`
    });

    const finalAudioFiles = [];

    for (let i = 0; i < recap.length; i++) {
      const block = recap[i];
      const number = i + 1;
      const rawPcmPath = path.join(jobFolder, `voice-${number}.pcm`);
      const rawWavPath = path.join(jobFolder, `voice-${number}.wav`);
      const fittedWavPath = path.join(jobFolder, `voice-${number}-fitted.wav`);

      tempFiles.push(rawPcmPath, rawWavPath, fittedWavPath);

      updateJob(jobId, {
        stage: 'Gemini TTS',
        progress: Math.round(48 + (i / Math.max(1, recap.length)) * 22),
        message: `Myanmar Voice ${number}/${recap.length} ထုတ်နေပါတယ်...`
      });

      const pcm = await generateGeminiTTS(
        block.narration,
        geminiKey,
        voice,
        style
      );

      fs.writeFileSync(rawPcmPath, pcm);
      await pcmToWav(pcm, rawWavPath);
      await fitAudioToScene(rawWavPath, fittedWavPath, block.duration);

      finalAudioFiles.push(fittedWavPath);
    }

    // Join the already time-fitted narration blocks into one continuous narrator track.
    updateJob(jobId, {
      stage: 'FFmpeg Audio Timeline',
      progress: 74,
      message: 'မြန်မာအသံ Timeline တစ်ကြောင်းတည်းအဖြစ် ပေါင်းနေပါတယ်...'
    });

    const narrationPath = path.join(jobFolder, 'myanmar-narration.wav');
    await concatAudioFiles(finalAudioFiles, narrationPath, jobId);
    tempFiles.push(narrationPath);

    updateJob(jobId, {
      stage: 'FFmpeg Final Render',
      progress: 84,
      message: 'Video နဲ့ မြန်မာအသံကို Timeline အတိုင်း ကွက်တိ Sync လုပ်နေပါတယ်...'
    });

    const outputPath = path.join(jobFolder, 'final-recap.mp4');

    await run('ffmpeg', [
      '-y',
      '-i', videoPath,
      '-i', narrationPath,
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
      '-t', info.duration.toFixed(3),
      '-movflags', '+faststart',
      outputPath
    ]);

    const stats = fs.statSync(outputPath);
    const filename = `${path.parse(originalFilename || 'movie').name}-Myanmar-Recap.mp4`;

    updateJob(jobId, {
      status: 'completed',
      stage: 'Completed',
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

  } catch (error) {
    const message = parseErrorMessage(error);
    console.error(`[JOB ${jobId}] FAILED:`, message);

    updateJob(jobId, {
      status: 'failed',
      stage: 'Failed',
      progress: 0,
      message: isRateLimitError(error)
        ? 'Gemini quota/rate limit ရောက်နေပါတယ်။'
        : 'Generate မအောင်မြင်ပါ။',
      error: message
    });
  } finally {
    cleanup(videoPath);
    cleanupMany(tempFiles);
  }
}

async function concatAudioFiles(audioFiles, outputPath, jobId) {
  const listPath = path.join(getJobFolder(jobId), 'audio-concat.txt');
  const lines = audioFiles.map(file => `file '${file.replace(/'/g, "'\\''")}'`);
  await fsp.writeFile(listPath, `${lines.join('\n')}\n`, 'utf8');

  try {
    await run('ffmpeg', [
      '-y',
      '-f', 'concat',
      '-safe', '0',
      '-i', listPath,
      '-c:a', 'pcm_s16le',
      '-ar', '24000',
      '-ac', '1',
      outputPath
    ]);
  } finally {
    cleanup(listPath);
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
      whisper: GROQ_MODEL,
      recap: process.env.GROQ_RECAP_MODEL || 'openai/gpt-oss-120b',
      tts: GEMINI_TTS_MODEL
    },
    sceneSync: true,
    sceneFPS: SCENE_FPS,
    maxScenes: MAX_SCENES,
    maxGeminiRequestsPerMovie: null,
    geminiRetries: GEMINI_MAX_RETRIES,
    timelineSync: true,
    maxVideoSeconds: MAX_VIDEO_SECONDS,
    maxVideoSizeMB: Math.round(MAX_VIDEO_SIZE / 1024 / 1024)
  });
});

app.get('/api/status/:jobId', (req, res) => {
  const job = getJob(req.params.jobId);

  if (!job) {
    return res.status(404).json({
      ok: false,
      error: 'Job not found',
      jobId: req.params.jobId
    });
  }

  res.json({ ok: true, job });
});

app.get('/api/download/:jobId', (req, res) => {
  const job = getJob(req.params.jobId);

  if (!job) {
    return res.status(404).send('Job not found');
  }

  if (job.status !== 'completed' || !job.output?.path) {
    return res.status(404).send('Final Video မရသေးပါ');
  }

  if (!fs.existsSync(job.output.path)) {
    return res.status(404).send('Final Video ဖိုင် မတွေ့ပါ။ Server filesystem ပြန်စတင်ထားနိုင်ပါတယ်။');
  }

  return res.download(
    job.output.path,
    job.output.filename || 'YNT-One-Clips.mp4'
  );
});

app.post('/api/one-clip', upload.single('video'), async (req, res) => {
  const filePath = req.file?.path;

  try {
    if (!filePath || !req.file) {
      return res.status(400).json({ ok: false, error: 'Video file မရှိပါ' });
    }

    const job = createJob();

    updateJob(job.id, {
      originalFilename: req.file.originalname,
      status: 'queued',
      stage: 'Queued',
      progress: 1,
      message: 'Movie ကို queue ထဲထည့်ပြီးပါပြီ။'
    });

    const options = {
      language: String(req.body?.language || 'my'),
      style: String(req.body?.style || 'cinematic'),
      voice: String(req.body?.voice || 'female'),
      mimeType: String(req.file.mimetype || 'video/mp4')
    };

    /* Start after the HTTP response is returned. */
    setImmediate(() => {
      processOneClip(
        job.id,
        filePath,
        req.file.originalname,
        options
      ).catch(error => {
        console.error(`[JOB ${job.id}] Background error:`, error);
        updateJob(job.id, {
          status: 'failed',
          stage: 'Failed',
          error: parseErrorMessage(error),
          message: 'Background processing မအောင်မြင်ပါ။'
        });
      });
    });

    return res.status(202).json({
      ok: true,
      jobId: job.id,
      statusUrl: `/api/status/${job.id}`,
      downloadUrl: `/api/download/${job.id}`
    });

  } catch (error) {
    cleanup(filePath);
    console.error('ONE CLIP START ERROR:', error);
    return res.status(400).json({
      ok: false,
      error: parseErrorMessage(error)
    });
  }
});

app.use('/api', (_req, res) => {
  res.status(404).json({ ok: false, error: 'API endpoint မတွေ့ပါ' });
});

app.use((error, _req, res, _next) => {
  console.error('SERVER ERROR:', error);

  if (res.headersSent) return;

  if (error?.code === 'LIMIT_FILE_SIZE') {
    return res.status(413).json({
      ok: false,
      error: `Video size က ${Math.round(MAX_VIDEO_SIZE / 1024 / 1024)}MB ထက် မကျော်ရပါ`
    });
  }

  return res.status(500).json({
    ok: false,
    error: parseErrorMessage(error)
  });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log('============================================');
  console.log('YNT One Clips server started');
  console.log(`PORT: ${PORT}`);
  console.log(`Groq: ${GROQ_MODEL}`);
  console.log(`Groq Recap: ${process.env.GROQ_RECAP_MODEL || 'openai/gpt-oss-120b'}`);
  console.log(`Gemini TTS: ${GEMINI_TTS_MODEL}`);
  console.log('Timeline Sync: ON (Whisper timestamps)');
  console.log(`Timeline Blocks: auto (45-90s)`);
  console.log(`Gemini Retries: ${GEMINI_MAX_RETRIES}`);
  console.log('Job Storage: JSON on disk');
  console.log('============================================');
});
