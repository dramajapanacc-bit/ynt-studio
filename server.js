import express from "express";
import multer from "multer";
import dotenv from "dotenv";
import Groq from "groq-sdk";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import { spawn, execFile } from "node:child_process";
import { promisify } from "node:util";
import { createReadStream } from "node:fs";
import { pipeline } from "node:stream/promises";

dotenv.config();
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();
const PORT = Number(process.env.PORT || 3000);
const ROOT = __dirname;
const UPLOAD_DIR = path.join(ROOT, "uploads");
const OUTPUT_DIR = path.join(ROOT, "outputs");
const TEMP_DIR = path.join(ROOT, "temp");
for (const dir of [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]) fs.mkdirSync(dir, { recursive: true });
const upload = multer({
  dest: UPLOAD_DIR,
  limits: { fileSize: Number(process.env.MAX_UPLOAD_MB || 2048) * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (/^video\/|application\/octet-stream/.test(file.mimetype) || /\.(mp4|mov|mkv|avi|webm)$/i.test(file.originalname)) cb(null, true);
    else cb(new Error("Video ဖိုင်သာ တင်ပေးပါ (MP4, MOV, MKV, AVI, WEBM)."));
  }
});
const jobs = new Map();
const execFileAsync = promisify(execFile);
const PYTHON = process.env.PYTHON_BIN || (process.platform === "win32" ? "python" : "python3");
const FFMPEG = process.env.FFMPEG_PATH || "ffmpeg";
const FFPROBE = process.env.FFPROBE_PATH || "ffprobe";
const groq = process.env.GROQ_API_KEY ? new Groq({ apiKey: process.env.GROQ_API_KEY }) : null;

app.use(express.json({ limit: "2mb" }));
app.use(express.static(path.join(ROOT, "public")));
app.get("/api/health", async (_req, res) => {
  let ffmpeg = false;
  try { await execFileAsync(FFMPEG, ["-version"]); ffmpeg = true; } catch {}
  res.json({ ok: true, configured: !!groq, ffmpeg, python: PYTHON, message: !groq ? "GROQ_API_KEY မထည့်ရသေးပါ။" : !ffmpeg ? "FFmpeg မတွေ့ပါ။" : "ready" });
});
app.get("/api/history", (_req, res) => {
  res.json([...jobs.values()].filter(j => j.status === "completed").map(j => ({
    jobId: j.id, name: j.originalName, createdAt: j.createdAt, downloadUrl: j.downloadUrl
  })).reverse().slice(0, 30));
});
app.post("/api/voice-preview", async (req, res) => {
  try {
    const text = "မင်္ဂလာပါ။ YanNaing Recap Tool မှ ကြိုဆိုပါတယ်။";
    const out = path.join(TEMP_DIR, `preview-${crypto.randomUUID()}.mp3`);
    await synthesize(text, req.body?.voice || "my-MM-Naing (Male)", Number(req.body?.speed) || 1, Number(req.body?.volume ?? 1), out);
    res.setHeader("Content-Type", "audio/mpeg");
    res.setHeader("Cache-Control", "no-store");
    res.sendFile(out, err => { fs.promises.rm(out, { force: true }).catch(() => {}); if (err && !res.headersSent) res.status(500).end(); });
  } catch (e) { res.status(500).json({ error: friendlyError(e) }); }
});
app.post("/api/generate", upload.single("video"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "Video ဖိုင်တင်ပါ။" });
  if (!groq) { await fs.promises.rm(req.file.path, { force: true }); return res.status(503).json({ error: "GROQ_API_KEY မရှိသေးပါ။ .env.example ကို .env အဖြစ်ကူးပြီး API key ထည့်ပါ။" }); }
  const id = crypto.randomUUID();
  let settings = {};
  try { settings = JSON.parse(req.body.settings || "{}"); } catch {}
  const job = {
    id, status: "queued", progress: 3, message: "တန်းစီနေပါတယ်…",
    originalName: req.file.originalname, inputPath: req.file.path,
    settings, createdAt: new Date().toISOString(), error: null
  };
  jobs.set(id, job);
  res.status(202).json({ jobId: id });
  processJob(job).catch(async e => {
    job.status = "failed"; job.error = friendlyError(e); job.message = job.error; job.progress = 100;
    await fs.promises.rm(job.inputPath, { force: true }).catch(() => {});
    console.error(`[${id}]`, e);
  });
});
app.get("/api/jobs/:id", (req, res) => {
  const j = jobs.get(req.params.id);
  if (!j) return res.status(404).json({ error: "Job မတွေ့ပါ။" });
  res.json({
    jobId: j.id, status: j.status, progress: j.progress, message: j.message,
    error: j.error, downloadUrl: j.downloadUrl, script: j.script
  });
});
app.get("/api/download/:id", (req, res) => {
  const j = jobs.get(req.params.id);
  if (!j || j.status !== "completed" || !j.outputPath || !fs.existsSync(j.outputPath)) return res.status(404).send("Output file မတွေ့ပါ။");
  res.download(j.outputPath, `${path.parse(j.originalName).name}-recap.mp4`);
});
app.use((err, _req, res, _next) => {
  const status = err instanceof multer.MulterError ? 413 : 400;
  res.status(status).json({ error: err.message || "Request failed" });
});

async function processJob(job) {
  const work = path.join(TEMP_DIR, job.id);
  fs.mkdirSync(work, { recursive: true });
  const audioPath = path.join(work, "source-audio.mp3");
  const voicePath = path.join(work, "recap-voice.mp3");
  const srtPath = path.join(work, "captions.srt");
  const outputPath = path.join(OUTPUT_DIR, `${job.id}.mp4`);
  try {
    job.status = "processing";
    let script = String(job.settings?.story?.customScript || "").trim();
    if (!script) {
      job.progress = 8; job.message = "Video အသံကို ထုတ်ယူနေပါတယ်…";
      await execFileAsync(FFMPEG, ["-y", "-i", job.inputPath, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", audioPath], { maxBuffer: 10 * 1024 * 1024 });
      job.progress = 18; job.message = "အသံကို စာသားအဖြစ် ပြောင်းနေပါတယ်…";
      const transcript = await transcribeLongAudio(audioPath, work, job);
      job.progress = 47; job.message = "မြန်မာ Recap Script ရေးနေပါတယ်…";
      script = await makeRecap(transcript, job.settings);
    }
    if (!script.trim()) throw new Error("Recap Script မထုတ်နိုင်ပါ။ Custom Script ထည့်ပြီး ပြန်စမ်းပါ။");
    job.script = script;
    job.progress = 60; job.message = "Edge TTS နဲ့ မြန်မာအသံထုတ်နေပါတယ်…";
    await synthesize(script, job.settings?.voice?.name, Number(job.settings?.voice?.speed) || 1, Number(job.settings?.voice?.volume ?? 1), voicePath);
    job.progress = 74; job.message = "Subtitle အချိန်ညှိနေပါတယ်…";
    const voiceDuration = await mediaDuration(voicePath);
    await fs.promises.writeFile(srtPath, makeSrt(script, voiceDuration), "utf8");
    job.progress = 82; job.message = "ဗီဒီယို Render လုပ်နေပါတယ်…";
    await renderVideo(job.inputPath, voicePath, srtPath, outputPath, job.settings, voiceDuration);
    job.outputPath = outputPath;
    job.downloadUrl = `/api/download/${job.id}`;
    job.status = "completed"; job.progress = 100; job.message = "Recap ပြီးပါပြီ။ Download လုပ်နိုင်ပါပြီ။";
  } finally {
    await fs.promises.rm(job.inputPath, { force: true }).catch(() => {});
    await fs.promises.rm(work, { recursive: true, force: true }).catch(() => {});
  }
}
async function transcribeLongAudio(audioPath, work, job) {
  const duration = await mediaDuration(audioPath);
  const segments = [];
  if (duration > 540) {
    const pattern = path.join(work, "audio-%03d.mp3");
    await execFileAsync(FFMPEG, ["-y", "-i", audioPath, "-f", "segment", "-segment_time", "480", "-reset_timestamps", "1", "-c", "copy", pattern]);
    for (const f of (await fs.promises.readdir(work)).filter(n => /^audio-\d+\.mp3$/.test(n)).sort()) segments.push(path.join(work, f));
  } else segments.push(audioPath);
  const texts = [];
  for (let i = 0; i < segments.length; i++) {
    job.progress = 20 + Math.round((i / segments.length) * 22);
    job.message = `အသံစာသားထုတ်ယူနေပါတယ်… (${i + 1}/${segments.length})`;
    const result = await groq.audio.transcriptions.create({
      file: createReadStream(segments[i]),
      model: process.env.GROQ_TRANSCRIBE_MODEL || "whisper-large-v3-turbo",
      response_format: "json",
      language: process.env.TRANSCRIBE_LANGUAGE || "my"
    });
    texts.push(result.text || "");
  }
  const transcript = texts.join("\n").trim();
  if (!transcript) throw new Error("Video ထဲက အသံစာသား မရပါ။ Video အသံပါ/မပါ စစ်ပါ၊ သို့မဟုတ် Custom Script ထည့်ပါ။");
  return transcript;
}
async function makeRecap(transcript, settings) {
  const length = settings?.story?.length || "Standard (2–4 min)";
  const language = settings?.story?.language || "မြန်မာ (Myanmar)";
  const template = settings?.story?.template || "Cinematic Recap";
  const target = length.includes("Short") ? "under 1 minute" : length.includes("Long") ? "5–10 minutes" : "2–4 minutes";
  const prompt = `You are a professional movie recap script writer. Write a spoken narration script based ONLY on the transcript below. Language: ${language}. Style: ${template}. Target spoken duration: ${target}. Explain the events in coherent order, engaging but accurate. Do not invent visual events not supported by the transcript. No headings, no markdown, no bullet points, no speaker labels; output only the narration text. If transcript is incomplete, avoid unsupported claims.\n\nTRANSCRIPT:\n${transcript.slice(0, 85000)}`;
  const result = await groq.chat.completions.create({
    model: process.env.GROQ_CHAT_MODEL || "llama-3.3-70b-versatile",
    temperature: 0.55,
    messages: [
      { role: "system", content: "You write clear, natural spoken Myanmar movie recap scripts. Return only the script." },
      { role: "user", content: prompt }
    ]
  });
  return result.choices?.[0]?.message?.content?.trim() || "";
}
function resolveVoice(value) {
  const v = String(value || "");
  if (/Nilar|Female/i.test(v)) return "my-MM-NilarNeural";
  return process.env.EDGE_TTS_VOICE || "my-MM-ThihaNeural";
}
async function synthesize(text, voice, speed, volume, output) {
  const txtPath = output + ".txt";
  await fs.promises.writeFile(txtPath, text, "utf8");
  const rate = Math.round((Math.max(0.5, Math.min(1.5, speed || 1)) - 1) * 100);
  const vol = Math.round((Math.max(0, Math.min(1, volume ?? 1)) - 1) * 100);
  const args = ["-m", "edge_tts", "--voice", resolveVoice(voice), "--rate", `${rate >= 0 ? "+" : ""}${rate}%`, "--volume", `${vol >= 0 ? "+" : ""}${vol}%`, "--file", txtPath, "--write-media", output];
  try { await execFileAsync(PYTHON, args, { maxBuffer: 10 * 1024 * 1024, timeout: 10 * 60 * 1000 }); }
  catch (e) { throw new Error(`Edge TTS မအောင်မြင်ပါ။ Python dependency များ install လုပ်ထားပါ။ ${e.stderr || e.message}`); }
  finally { await fs.promises.rm(txtPath, { force: true }).catch(() => {}); }
}
async function mediaDuration(file) {
  const { stdout } = await execFileAsync(FFPROBE, ["-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file]);
  const n = Number(stdout.trim());
  if (!Number.isFinite(n) || n <= 0) throw new Error("Media duration ကို မဖတ်နိုင်ပါ။ FFmpeg/FFprobe ကို စစ်ပါ။");
  return n;
}
function splitCaptionText(text, maxChars = 34) {
  const clean = text.replace(/\s+/g, " ").trim();
  const sentences = clean.split(/(?<=[။.!?])\s*/u).filter(Boolean);
  const out = [];
  for (const sentence of (sentences.length ? sentences : [clean])) {
    let rest = sentence.trim();
    while (rest.length > maxChars) {
      let cut = rest.lastIndexOf(" ", maxChars);
      if (cut < Math.floor(maxChars * 0.55)) cut = maxChars;
      out.push(rest.slice(0, cut).trim()); rest = rest.slice(cut).trim();
    }
    if (rest) out.push(rest);
  }
  return out;
}
function srtTime(seconds) {
  const ms = Math.max(0, Math.floor(seconds * 1000));
  const h = String(Math.floor(ms / 3600000)).padStart(2, "0");
  const m = String(Math.floor(ms / 60000) % 60).padStart(2, "0");
  const s = String(Math.floor(ms / 1000) % 60).padStart(2, "0");
  const x = String(ms % 1000).padStart(3, "0");
  return `${h}:${m}:${s},${x}`;
}
function makeSrt(text, duration) {
  const chunks = splitCaptionText(text);
  const weights = chunks.map(t => Math.max(1, t.length));
  const total = weights.reduce((a, b) => a + b, 0);
  let cursor = 0;
  return chunks.map((line, i) => {
    const start = cursor;
    cursor += duration * weights[i] / total;
    return `${i + 1}\n${srtTime(start)} --> ${srtTime(Math.min(duration, cursor))}\n${line}\n`;
  }).join("\n");
}
async function renderVideo(input, voice, srt, output, settings, voiceDuration) {
  const ratio = settings?.ratio || "16:9";
  const size = String(settings?.resolution || "").includes("3840") ? (ratio === "9:16" ? "2160:3840" : "3840:2160")
    : String(settings?.resolution || "").includes("1280") ? (ratio === "9:16" ? "720:1280" : "1280:720")
    : (ratio === "9:16" ? "1080:1920" : "1920:1080");
  const [w, h] = size.split(":");
  const fps = Number.parseInt(settings?.fps || "30", 10) || 30;
  const fontSize = Math.max(18, Math.min(72, Number(settings?.subtitle?.fontSize) || 36));
  const position = String(settings?.subtitle?.position || "Bottom").toLowerCase();
  const marginV = position.includes("top") ? 40 : position.includes("center") ? Math.round(Number(h) / 2) : 45;
  const color = (settings?.subtitle?.fontColor || "#ffffff").replace("#", "");
  const bg = settings?.subtitle?.background || "Semi-Transparent";
  const outline = settings?.subtitle?.outline === false ? "0" : "2";
  const alpha = bg === "None" ? "&HFF" : bg === "Solid Black" ? "&H00" : "&H66";
  const style = `FontName=Noto Sans Myanmar,FontSize=${fontSize},PrimaryColour=&H00${color.slice(4,6)}${color.slice(2,4)}${color.slice(0,2)},OutlineColour=&H000000,BorderStyle=${bg === "None" ? 1 : 3},BackColour=${alpha}000000,Outline=${outline},Shadow=1,Alignment=${position.includes("top") ? 8 : position.includes("center") ? 5 : 2},MarginV=${marginV}`;
  const safeSrt = srt.replace(/\\/g, "/").replace(/:/g, "\\:").replace(/'/g, "\\'");
  const videoFilter = `scale=${w}:${h}:force_original_aspect_ratio=increase,crop=${w}:${h},fps=${fps},subtitles='${safeSrt}':force_style='${style}'`;
  const args = ["-y", "-i", input, "-i", voice, "-vf", videoFilter, "-map", "0:v:0", "-map", "1:a:0", "-t", String(voiceDuration), "-c:v", "libx264", "-preset", process.env.FFMPEG_PRESET || "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", output];
  try { await execFileAsync(FFMPEG, args, { maxBuffer: 20 * 1024 * 1024, timeout: 0 }); }
  catch (e) {
    if (/No such filter: 'subtitles'|not found|Unable to open/i.test(e.stderr || "")) {
      throw new Error("FFmpeg တွင် subtitles/libass filter မပါပါ။ libass ပါသော FFmpeg build ကို install လုပ်ပါ။");
    }
    throw new Error(`Video Render မအောင်မြင်ပါ။ ${String(e.stderr || e.message).slice(-1200)}`);
  }
}
function friendlyError(e) {
  const m = String(e?.message || e || "Unknown error");
  if (/401|invalid_api_key|authentication/i.test(m)) return "Groq API Key မမှန်ပါ။ .env ထဲက GROQ_API_KEY ကို စစ်ပါ။";
  if (/429|rate_limit/i.test(m)) return "Groq API rate limit ပြည့်နေပါတယ်။ ခဏစောင့်ပြီး ပြန်စမ်းပါ။";
  if (/ENOSPC|no space/i.test(m)) return "Disk space မလုံလောက်ပါ။ uploads, temp, outputs ဖိုင်များကိုရှင်းပါ။";
  if (/ENOENT/i.test(m) && /ffmpeg|ffprobe/i.test(m)) return "FFmpeg/FFprobe မတွေ့ပါ။ Install လုပ်ပြီး PATH ထဲထည့်ပါ။";
  return m.slice(0, 1800);
}
const server = app.listen(PORT, () => console.log(`YanNaing Recap Tool running at http://localhost:${PORT}`));
function shutdown() { server.close(() => process.exit(0)); }
process.on("SIGINT", shutdown); process.on("SIGTERM", shutdown);
