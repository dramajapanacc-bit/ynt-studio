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

const GROQ_WHISPER_MODEL = 'whisper-large-v3-turbo';
const GROQ_RECAP_MODEL = 'openai/gpt-oss-120b';
const GEMINI_TTS_MODEL = 'gemini-3.8-flash-lite-tts';
const GEMINI_TTS_RETRIES = 2;
const GEMINI_TTS_RETRY_DELAY = 2500;

const GEMINI_VOICES = {
  male: 'Puck', female: 'Kore', kore: 'Kore', puck: 'Puck',
  aoede: 'Aoede', zephyr: 'Zephyr', charon: 'Charon', fenrir: 'Fenrir'
};

for (const dir of [PUBLIC_DIR, UPLOAD_DIR, JOB_DIR]) {
  fs.mkdirSync(dir, { recursive: true });
}

app.use(express.json({ limit: '20mb' }));
app.use(express.urlencoded({ extended: true, limit: '20mb' }));

// FRONTEND: serve public/index.html first, with root index.html as fallback.
app.use(express.static(PUBLIC_DIR));
app.get('/', (_req, res) => {
  const publicIndex = path.join(PUBLIC_DIR, 'index.html');
  const rootIndex = path.join(ROOT, 'index.html');
  if (fs.existsSync(publicIndex)) return res.sendFile(publicIndex);
  if (fs.existsSync(rootIndex)) return res.sendFile(rootIndex);
  return res.status(404).send('index.html မတွေ့ပါ');
});

const allowedExt = new Set([
  '.mp4','.mov','.mkv','.webm','.avi','.m4v','.flv','.wmv','.mpeg','.mpg'
]);

const upload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, UPLOAD_DIR),
    filename: (_req, file, cb) => {
      const e = path.extname(file.originalname || '').toLowerCase();
      cb(null, `${Date.now()}-${crypto.randomBytes(8).toString('hex')}${allowedExt.has(e) ? e : '.mp4'}`);
    }
  }),
  limits: { fileSize: MAX_VIDEO_SIZE, files: 1 },
  fileFilter: (_req, file, cb) => {
    const e = path.extname(file.originalname || '').toLowerCase();
    if (!allowedExt.has(e)) return cb(new Error('MP4 / MOV / MKV / WEBM / AVI video ကိုသုံးပါ။'));
    cb(null, true);
  }
});

// PostgreSQL job storage
let pool;
async function initDatabase() {
  const url = String(process.env.DATABASE_URL || '').trim();
  if (!url) throw new Error('DATABASE_URL မရှိပါ။ Render PostgreSQL Internal Database URL ကို Environment ထဲမှာထည့်ပါ။');
  pool = new Pool({
    connectionString: url,
    ssl: /localhost|127\.0\.0\.1/.test(url) ? false : { rejectUnauthorized: false },
    max: 5, idleTimeoutMillis: 30000, connectionTimeoutMillis: 10000
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

function rowToJob(r) {
  if (!r) return null;
  return {
    id:r.id,status:r.status,stage:r.stage,progress:r.progress,message:r.message,
    createdAt:r.created_at,updatedAt:r.updated_at,originalFilename:r.original_filename,
    duration:r.duration,totalChunks:r.total_chunks,transcript:r.transcript,
    recapBlocks:r.recap_blocks,output:r.output,error:r.error
  };
}
async function createJob(originalFilename) {
  const id = crypto.randomUUID();
  const r = await pool.query(
    `INSERT INTO one_clip_jobs(id,status,stage,progress,message,original_filename)
     VALUES($1,'created','Waiting',0,'Job created.',$2) RETURNING *`, [id, originalFilename || null]
  );
  return rowToJob(r.rows[0]);
}
async function getJob(id) {
  const r = await pool.query('SELECT * FROM one_clip_jobs WHERE id=$1',[id]);
  return rowToJob(r.rows[0]);
}
async function updateJob(id, data) {
  const map = {
    status:'status',stage:'stage',progress:'progress',message:'message',
    originalFilename:'original_filename',duration:'duration',totalChunks:'total_chunks',
    transcript:'transcript',recapBlocks:'recap_blocks',output:'output',error:'error'
  };
  const fields=[], vals=[]; let n=1;
  for (const [k,col] of Object.entries(map)) {
    if (!(k in data)) continue;
    fields.push(`${col}=$${n++}`);
    vals.push(['transcript','recapBlocks','output'].includes(k) ? JSON.stringify(data[k]) : data[k]);
  }
  if (!fields.length) return getJob(id);
  fields.push('updated_at=NOW()'); vals.push(id);
  const r = await pool.query(`UPDATE one_clip_jobs SET ${fields.join(',')} WHERE id=$${n} RETURNING *`, vals);
  return rowToJob(r.rows[0]);
}

const cleanText = v => String(v ?? '').replace(/\r/g,' ').replace(/\n+/g,' ').replace(/\s+/g,' ').trim();
const sleep = ms => new Promise(r => setTimeout(r,ms));
const cleanup = f => { if (f) fs.rm(f,{force:true},()=>{}); };
const parseErrorMessage = e => String(e?.message || e?.error?.message || e || 'Unknown error');
function requireEnv(n) { const v=String(process.env[n]||'').trim(); if(!v) throw new Error(`${n} မရှိပါ`); return v; }
async function run(cmd,args,opts={}) {
  try { return await execFileAsync(cmd,args,{maxBuffer:100*1024*1024,...opts}); }
  catch(e) { throw new Error(`${cmd} failed: ${e?.stderr || e?.message || e}`); }
}
function resolveVoice(v) { return GEMINI_VOICES[String(v||'female').toLowerCase()] || 'Kore'; }
function styleInstruction(s) {
  s=String(s||'cinematic').toLowerCase();
  if(s.includes('short')) return 'concise, fast-paced and easy to follow';
  if(s.includes('story')) return 'smooth storytelling with suspense and emotional flow';
  if(s.includes('detailed')) return 'detailed but natural movie recap narration';
  if(s.includes('dramatic')) return 'dramatic and emotional but still natural';
  return 'cinematic, natural and easy to understand';
}

async function probeVideo(file) {
  const {stdout}=await run('ffprobe',['-v','error','-show_entries','format=duration:stream=index,codec_type,width,height','-of','json',file]);
  const d=JSON.parse(stdout), duration=Number(d?.format?.duration||0);
  if(!Number.isFinite(duration)||duration<=0) throw new Error('Video duration မဖတ်နိုင်ပါ');
  const video=(d.streams||[]).find(s=>s.codec_type==='video');
  return {duration,width:Number(video?.width||0),height:Number(video?.height||0)};
}
async function extractAudio(video,folder) {
  const out=path.join(folder,'movie-audio.mp3');
  await run('ffmpeg',['-y','-i',video,'-vn','-map','0:a:0?','-ac','1','-ar','16000','-c:a','libmp3lame','-b:a','64k',out]);
  return out;
}
async function splitAudio(audio,duration,folder) {
  const dir=path.join(folder,'whisper-chunks'); fs.mkdirSync(dir,{recursive:true}); const chunks=[];
  for(let start=0,i=0;start<duration-.05;start+=WHISPER_CHUNK_SECONDS,i++){
    const seconds=Math.min(WHISPER_CHUNK_SECONDS,duration-start), out=path.join(dir,`chunk-${String(i).padStart(3,'0')}.mp3`);
    await run('ffmpeg',['-y','-ss',String(start),'-i',audio,'-t',String(seconds),'-ac','1','-ar','16000','-c:a','libmp3lame','-b:a','64k',out]);
    chunks.push({path:out,offset:start,duration:seconds});
  }
  return chunks;
}

async function transcribeGroq(chunks,jobId) {
  const groq=new Groq({apiKey:requireEnv('GROQ_API_KEY')}), segments=[];
  for(let i=0;i<chunks.length;i++){
    const c=chunks[i];
    await updateJob(jobId,{stage:'Whisper',progress:Math.round(8+(i/Math.max(1,chunks.length))*27),message:`Groq Whisper ${i+1}/${chunks.length} လုပ်နေပါတယ်...`});
    const r=await groq.audio.transcriptions.create({
      file:fs.createReadStream(c.path),model:GROQ_WHISPER_MODEL,response_format:'verbose_json',
      timestamp_granularities:['segment'],temperature:0
    });
    if(Array.isArray(r?.segments)&&r.segments.length){
      for(const s of r.segments){
        const t=cleanText(s?.text); if(!t) continue;
        const ls=Number(s?.start), le=Number(s?.end);
        segments.push({start:c.offset+(Number.isFinite(ls)?ls:0),end:c.offset+(Number.isFinite(le)?le:Math.min(c.duration,2)),text:t});
      }
    } else if(r?.text) segments.push({start:c.offset,end:c.offset+c.duration,text:cleanText(r.text)});
  }
  segments.sort((a,b)=>a.start-b.start);
  const normalized=segments.map((s,i)=>({id:i+1,start:Math.max(0,Number(s.start)||0),end:Math.max((Number(s.start)||0)+.05,Number(s.end)||((Number(s.start)||0)+.05)),text:s.text}));
  return {text:normalized.map(s=>s.text).join(' ').trim(),segments:normalized};
}

function buildTimelineBlocks(transcript,duration) {
  const blocks=[];
  for(let start=0;start<duration-.05;start+=RECAP_BLOCK_SECONDS){
    const end=Math.min(duration,start+RECAP_BLOCK_SECONDS);
    const items=(transcript.segments||[]).filter(s=>s.end>start&&s.start<end);
    blocks.push({id:blocks.length+1,start,end,duration:end-start,
      transcript:items.map(s=>`[${s.start.toFixed(2)}-${s.end.toFixed(2)}] ${s.text}`).join('\n')});
  }
  if(!blocks.length) blocks.push({id:1,start:0,end:duration,duration,transcript:transcript.text||''});
  return blocks;
}

async function generateRecapForBlock(groq,block,language,style) {
  const languageText=language==='en'?'natural spoken English':'natural spoken Myanmar Burmese';
  const prompt=`You are writing narration for ONE block of a movie recap video.
The video timeline is exactly ${block.start.toFixed(2)}s to ${block.end.toFixed(2)}s.
Write ${languageText}. Style: ${styleInstruction(style)}.
Rules:
- Describe only story events supported by the supplied transcript.
- Keep event order exactly as timestamps show.
- Do not invent characters, locations, actions or dialogue.
- Do not mention timestamps.
- No headings, bullets or quotation marks.
- Sound like a human movie recap narrator.
- Keep narration compact enough to fit this block.
- Prefer about 2.2 to 2.8 spoken words per second for Myanmar.
- If silent/too short, use a brief neutral transition.
- Output narration text only.
TIMELINE TRANSCRIPT:
${block.transcript||'(no speech in this interval)'}`;
  const r=await groq.chat.completions.create({
    model:GROQ_RECAP_MODEL,temperature:.2,max_tokens:700,
    messages:[{role:'system',content:'Write accurate compact movie recap narration. Never invent unsupported facts.'},{role:'user',content:prompt}]
  });
  return cleanText(r?.choices?.[0]?.message?.content);
}
async function generateRecapBlocks(timeline,transcript,jobId,language,style) {
  const groq=new Groq({apiKey:requireEnv('GROQ_API_KEY')}), out=[];
  for(let i=0;i<timeline.length;i++){
    const b=timeline[i];
    await updateJob(jobId,{stage:'Recap Script',progress:Math.round(36+(i/Math.max(1,timeline.length))*14),message:`Groq Recap ${i+1}/${timeline.length} လုပ်နေပါတယ်...`});
    let narration='';
    try{ narration=await generateRecapForBlock(groq,b,language,style); }catch(e){ console.error(`[RECAP] Block ${b.id}:`,parseErrorMessage(e)); }
    if(!narration) narration=cleanText((transcript.segments||[]).filter(s=>s.end>b.start&&s.start<b.end).map(s=>s.text).join(' '))||'ဒီအပိုင်းမှာ ဇာတ်လမ်းက နောက်တစ်ဆင့်ကို ဆက်လက်ရွေ့လျားသွားပါတယ်။';
    out.push({...b,narration});
  }
  return out;
}

async function generateGeminiTTS(text,apiKey,voiceName,style,language) {
  const ai=new GoogleGenAI({apiKey}); let lastError;
  for(let attempt=0;attempt<=GEMINI_TTS_RETRIES;attempt++){
    try{
      const r=await ai.models.generateContent({
        model:GEMINI_TTS_MODEL,
        contents:[{role:'user',parts:[{text:`Speak this ${styleInstruction(style)} narration in ${language==='en'?'natural English':'natural Myanmar Burmese'}. Do not add or remove words.\n\n${text}`}]}],
        config:{responseModalities:['AUDIO'],responseFormat:{audio:{mimeType:'AUDIO_L16',sampleRate:24000}},
          speechConfig:{voiceConfig:{prebuiltVoiceConfig:{voiceName}}}}
      });
      const base64=r?.candidates?.[0]?.content?.parts?.find(p=>p?.inlineData?.data)?.inlineData?.data;
      if(!base64) throw new Error('Gemini TTS audio မရပါ');
      return Buffer.from(base64,'base64');
    }catch(e){
      lastError=e; const m=parseErrorMessage(e);
      const code=Number(e?.status||e?.code||0);
      const retryable=[429,500,502,503,504].includes(code)||/429|500|502|503|504|UNAVAILABLE|high demand|rate limit/i.test(m);
      if(!retryable||attempt>=GEMINI_TTS_RETRIES) break;
      await sleep(GEMINI_TTS_RETRY_DELAY*(attempt+1));
    }
  }
  throw new Error(`Gemini TTS failed: ${parseErrorMessage(lastError)}`);
}
async function pcmToWav(pcm,out) {
  const p=`${out}.pcm`; fs.writeFileSync(p,pcm);
  try{await run('ffmpeg',['-y','-f','s16le','-ar','24000','-ac','1','-i',p,'-c:a','pcm_s16le',out]);}
  finally{cleanup(p);}
}
async function mediaDuration(file){
  const {stdout}=await run('ffprobe',['-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',file]);
  const d=Number(stdout.trim()); if(!Number.isFinite(d)||d<=0) throw new Error('Media duration မဖတ်နိုင်ပါ'); return d;
}
function atempoFilters(factor){
  let v=factor, a=[];
  while(v>2){a.push('atempo=2');v/=2;} while(v<.5){a.push('atempo=.5');v/=.5;}
  if(Math.abs(v-1)>.001)a.push(`atempo=${v.toFixed(6)}`); return a;
}
async function fitAudioToDuration(input,output,targetSeconds){
  const source=await mediaDuration(input), target=Math.max(.25,targetSeconds), filters=atempoFilters(source/target);
  filters.push(`apad=pad_dur=${target.toFixed(3)}`,`atrim=0:${target.toFixed(3)}`,'asetpts=N/SR/TB');
  await run('ffmpeg',['-y','-i',input,'-filter:a',filters.join(','),'-ar','24000','-ac','1','-c:a','pcm_s16le',output]);
}
async function buildNarrationAudio(jobId,blocks,voice,style,language,folder){
  const key=requireEnv('GEMINI_API_KEY'), rendered=[];
  for(let i=0;i<blocks.length;i++){
    const b=blocks[i], raw=path.join(folder,`tts-${b.id}.pcm`), wav=path.join(folder,`tts-${b.id}.wav`), fit=path.join(folder,`tts-${b.id}-fit.wav`);
    await updateJob(jobId,{stage:'Voice',progress:Math.round(50+(i/Math.max(1,blocks.length))*25),message:`Myanmar Voice ${i+1}/${blocks.length} ထုတ်နေပါတယ်...`});
    const pcm=await generateGeminiTTS(b.narration,key,resolveVoice(voice),style,language);
    fs.writeFileSync(raw,pcm); await pcmToWav(pcm,wav); await fitAudioToDuration(wav,fit,b.duration);
    rendered.push({...b,audioPath:fit});
  }
  const out=path.join(folder,'narration-timeline.wav'), inputs=[], filters=[];
  const end=rendered[rendered.length-1].end;
  for(let i=0;i<rendered.length;i++){
    inputs.push('-i',rendered[i].audioPath);
    const d=Math.round(rendered[i].start*1000);
    filters.push(`[${i}:a]adelay=${d}|${d},apad,atrim=0:${end.toFixed(3)}[a${i}]`);
  }
  const mix=rendered.map((_,i)=>`[a${i}]`).join('');
  filters.push(`${mix}amix=inputs=${rendered.length}:duration=longest:dropout_transition=0,atrim=0:${end.toFixed(3)},asetpts=N/SR/TB[out]`);
  await run('ffmpeg',['-y',...inputs,'-filter_complex',filters.join(';'),'-map','[out]','-ar','24000','-ac','1','-c:a','pcm_s16le',out]);
  return out;
}
async function renderFinalVideo(movie,audio,out){
  await run('ffmpeg',['-y','-i',movie,'-i',audio,'-map','0:v:0','-map','1:a:0','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-ar','24000','-ac','1','-shortest','-movflags','+faststart',out]);
}

async function processOneClip(jobId,moviePath,originalFilename,options){
  const folder=path.join(JOB_DIR,jobId); fs.mkdirSync(folder,{recursive:true});
  try{
    const language=options.language||'my', style=options.style||'cinematic', voice=options.voice||'female';
    requireEnv('GROQ_API_KEY'); requireEnv('GEMINI_API_KEY');
    await updateJob(jobId,{status:'processing',stage:'Upload',progress:3,message:'Movie uploaded. Processing started.'});
    const info=await probeVideo(moviePath);
    if(info.duration>MAX_VIDEO_SECONDS) throw new Error('Video က 5 မိနစ်ထက် မကျော်ရပါ');
    await updateJob(jobId,{originalFilename,duration:info.duration,stage:'FFmpeg',progress:5,message:'Movie audio ကို ထုတ်ယူနေပါတယ်...'});
    const audio=await extractAudio(moviePath,folder), chunks=await splitAudio(audio,info.duration,folder);
    await updateJob(jobId,{totalChunks:chunks.length});
    const transcript=await transcribeGroq(chunks,jobId);
    if(!transcript.segments.length&&!transcript.text) throw new Error('Groq Whisper က Transcript မထုတ်ပေးနိုင်ပါ');
    await fsp.writeFile(path.join(folder,'transcript.json'),JSON.stringify(transcript,null,2),'utf8');
    await updateJob(jobId,{transcript,stage:'Transcript',progress:35,message:`${transcript.segments.length} timestamp segments ရပါပြီ။`});
    const timeline=buildTimelineBlocks(transcript,info.duration);
    await updateJob(jobId,{stage:'Recap Script',progress:37,message:`${timeline.length} timeline block(s) အတွက် Recap Script ပြုလုပ်နေပါတယ်...`});
    const recap=await generateRecapBlocks(timeline,transcript,jobId,language,style);
    await fsp.writeFile(path.join(folder,'recap-blocks.json'),JSON.stringify(recap,null,2),'utf8');
    await updateJob(jobId,{recapBlocks:recap,stage:'Voice',progress:50,message:`${recap.length} timeline block(s) အတွက် Myanmar Voice ထုတ်နေပါတယ်...`});
    const narration=await buildNarrationAudio(jobId,recap,voice,style,language,folder);
    await updateJob(jobId,{stage:'FFmpeg',progress:78,message:'Myanmar Voice နဲ့ Movie Timeline ကို exact timing နဲ့ပေါင်းနေပါတယ်...'});
    const outputPath=path.join(folder,'final-recap.mp4'); await renderFinalVideo(moviePath,narration,outputPath);
    if(!fs.existsSync(outputPath)||fs.statSync(outputPath).size<=0) throw new Error('FFmpeg final MP4 မထုတ်ပေးနိုင်ပါ');
    const stats=fs.statSync(outputPath), filename=`${path.parse(originalFilename||'movie').name}-Myanmar-Recap.mp4`;
    await updateJob(jobId,{status:'completed',stage:'Ready',progress:100,message:'Final Recap Video ရပါပြီ။',output:{path:outputPath,filename,size:stats.size,url:`/api/download/${jobId}`},error:null});
    console.log(`[JOB ${jobId}] COMPLETED`);
  }catch(e){
    const message=parseErrorMessage(e); console.error(`[JOB ${jobId}] FAILED:`,message);
    await updateJob(jobId,{status:'error',stage:'Error',progress:0,message:'Movie processing မအောင်မြင်ပါ။',error:message});
  }finally{ cleanup(moviePath); }
}

app.get('/api/health',async(_req,res)=>res.json({ok:true,name:'YNT One Clips',status:'online',models:{whisper:GROQ_WHISPER_MODEL,recap:GROQ_RECAP_MODEL,tts:GEMINI_TTS_MODEL},timeline:'Whisper timestamps',geminiSceneAnalysis:false,ttsBlocks:RECAP_BLOCK_SECONDS,maxVideoSeconds:MAX_VIDEO_SECONDS,maxVideoSizeMB:Math.round(MAX_VIDEO_SIZE/1024/1024),storage:'PostgreSQL'}));
app.get('/health',(_req,res)=>res.json({ok:true,name:'YNT One Clips',status:'online'}));

app.get('/api/status/:jobId',async(req,res)=>{
  try{const job=await getJob(req.params.jobId); if(!job)return res.status(404).json({ok:false,error:'Job not found',jobId:req.params.jobId}); return res.json({ok:true,...job,job});}
  catch(e){return res.status(500).json({ok:false,error:parseErrorMessage(e)});}
});
app.get('/api/download/:jobId',async(req,res)=>{
  try{
    const job=await getJob(req.params.jobId);
    if(!job)return res.status(404).send('Job not found');
    if(job.status!=='completed'||!job.output?.path)return res.status(404).send('Final Video မရသေးပါ');
    if(!fs.existsSync(job.output.path))return res.status(404).send('Final Video ဖိုင် မတွေ့ပါ။ Render filesystem ပြန်စတင်ထားနိုင်ပါတယ်။');
    return res.download(job.output.path,job.output.filename||'YNT-One-Clips.mp4');
  }catch(e){return res.status(500).send(parseErrorMessage(e));}
});

app.post('/api/one-clip',upload.single('movie'),async(req,res)=>{
  let filePath=req.file?.path;
  try{
    if(!req.file)return res.status(400).json({ok:false,error:'Movie file မရှိပါ'});
    const originalName=req.file.originalname||'movie.mp4', job=await createJob(originalName), folder=path.join(JOB_DIR,job.id);
    fs.mkdirSync(folder,{recursive:true});
    const ext0=path.extname(originalName).toLowerCase(), ext=allowedExt.has(ext0)?ext0:'.mp4', moviePath=path.join(folder,`movie${ext}`);
    fs.renameSync(filePath,moviePath); filePath=null;
    const options={language:String(req.body?.language||'my'),style:String(req.body?.style||'cinematic'),voice:String(req.body?.voice||'female')};
    await updateJob(job.id,{status:'processing',stage:'Upload',progress:3,message:'Movie uploaded. Processing started.'});
    console.log(`[JOB ${job.id}] STARTED | Timeline: Whisper timestamps | Gemini Scene Analysis: DISABLED`);
    res.status(202).json({ok:true,success:true,jobId:job.id,status:'processing',statusUrl:`/api/status/${job.id}`,downloadUrl:`/api/download/${job.id}`});
    setImmediate(()=>processOneClip(job.id,moviePath,originalName,options).catch(async e=>{
      try{await updateJob(job.id,{status:'error',stage:'Error',progress:0,message:'Background processing မအောင်မြင်ပါ။',error:parseErrorMessage(e)});}catch{}
    }));
  }catch(e){cleanup(filePath);return res.status(400).json({ok:false,error:parseErrorMessage(e)});}
});

app.use('/api',(_req,res)=>res.status(404).json({ok:false,error:'API endpoint မတွေ့ပါ'}));
app.use((error,_req,res,_next)=>{
  console.error('SERVER ERROR:',error);
  if(res.headersSent)return;
  if(error?.code==='LIMIT_FILE_SIZE')return res.status(413).json({ok:false,error:`Video size က ${Math.round(MAX_VIDEO_SIZE/1024/1024)}MB ထက် မကျော်ရပါ`});
  return res.status(500).json({ok:false,error:parseErrorMessage(error)});
});

async function startServer(){
  await initDatabase();
  app.listen(PORT,'0.0.0.0',()=> {
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
    console.log('Frontend: public/index.html -> /');
    console.log('============================================');
  });
}
startServer().catch(e=>{console.error('SERVER START FAILED:',parseErrorMessage(e));process.exit(1);});
