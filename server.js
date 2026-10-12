import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import { spawn } from 'node:child_process';
import Groq from 'groq-sdk';
import { GoogleGenAI } from '@google/genai';
import ffmpegStatic from 'ffmpeg-static';
import ffprobeStatic from 'ffprobe-static';
import { ensureFont, parseSrt, buildAss, buildFilterGraph, buildFfmpegArgs, FONT_FAMILY } from './burn.js';

const app = express();
const PORT = Number(process.env.PORT || 10000);
const MAX_BYTES = 300 * 1024 * 1024;
const MAX_SECONDS = 5 * 60;
const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
const ROOT = process.cwd();
const PUBLIC = path.join(ROOT, 'public');
const TMP = path.join(os.tmpdir(), 'ynt-thumbnail-srt');
await fs.mkdir(TMP, { recursive: true });
const FONTS=path.join(ROOT,'fonts');
let fontState=await ensureFont(FONTS);
if(!fontState.ok)console.warn('Myanmar font download failed - put a Myanmar .ttf in ./fonts');

const GROQ_MODEL = 'whisper-large-v3';
const GEMINI_TEXT_MODEL = 'gemini-3.5-flash';
const GEMINI_IMAGE_MODEL = 'gemini-3.1-flash-image';
const APP_VERSION = '4.1.0-burn-in-subtitles';

const videoExt = new Set(['.mp4','.mov','.mkv','.webm','.avi','.m4v','.flv','.wmv','.mpeg','.mpg']);
const imageExt = new Set(['.jpg','.jpeg','.png','.webp']);

const videoStorage = multer.diskStorage({
  destination: (_req,_file,cb)=>cb(null,TMP),
  filename: (_req,file,cb)=>cb(null,`${Date.now()}-${crypto.randomBytes(8).toString('hex')}${path.extname(file.originalname||'').toLowerCase()||'.mp4'}`)
});
const imageStorage = multer.memoryStorage();
const uploadVideo = multer({
  storage: videoStorage,
  limits:{fileSize:MAX_BYTES,files:1},
  fileFilter:(_req,file,cb)=>{
    const ext=path.extname(file.originalname||'').toLowerCase();
    if(!videoExt.has(ext)) return cb(new Error('MP4 / MOV / MKV / WEBM video ကိုသုံးပါ။'));
    cb(null,true);
  }
});
const uploadImages = multer({
  storage:imageStorage,
  limits:{fileSize:MAX_IMAGE_BYTES,files:5},
  fileFilter:(_req,file,cb)=>{
    const ext=path.extname(file.originalname||'').toLowerCase();
    if(!imageExt.has(ext)) return cb(new Error('JPG / PNG / WEBP image ကိုသုံးပါ။'));
    cb(null,true);
  }
});

app.use(express.json({limit:'20mb'}));
app.use(express.urlencoded({extended:true,limit:'20mb'}));
app.use(express.static(PUBLIC));

function getKey(req, headerName, bodyName, envName, label){
  const value=String(req.get(headerName)||req.body?.[bodyName]||process.env[envName]||'').trim();
  if(!value) throw new Error(`${label} မရှိပါ`);
  return value;
}
function sleep(ms){return new Promise(r=>setTimeout(r,ms));}
function cleanText(v){return String(v??'').replace(/\r/g,' ').replace(/\n+/g,' ').replace(/\s+/g,' ').trim();}
function cleanup(file){if(file)fs.rm(file,{force:true}).catch(()=>{});}
function runProcess(command,args){
  return new Promise((resolve,reject)=>{
    const child=spawn(command,args); let stdout='',stderr='';
    child.stdout.on('data',d=>stdout+=d.toString());
    child.stderr.on('data',d=>stderr+=d.toString());
    child.on('error',reject);
    child.on('close',code=>code===0?resolve({stdout,stderr}):reject(new Error(stderr||`${command} exited with code ${code}`)));
  });
}
async function probeVideo(file){
  const r=await runProcess(ffprobeStatic.path,['-v','error','-show_entries','format=duration:stream=codec_type,width,height','-of','json',file]);
  const d=JSON.parse(r.stdout); const duration=Number(d?.format?.duration||0);
  const video=(d?.streams||[]).find(s=>s.codec_type==='video');
  if(!duration||!video)throw new Error('Video ကိုဖတ်မရပါ');
  return {duration,width:Number(video.width||0),height:Number(video.height||0)};
}

function extractJson(text){
  const raw=String(text||'').trim().replace(/^```json\s*/i,'').replace(/^```\s*/i,'').replace(/```\s*$/i,'').trim();
  try{return JSON.parse(raw);}catch{}
  const a=raw.indexOf('{'),b=raw.lastIndexOf('}');
  if(a>=0&&b>a){try{return JSON.parse(raw.slice(a,b+1));}catch{}}
  return null;
}
function keepMyanmarOnly(text){
  return String(text||'').replace(/[A-Za-z\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]/g,'').replace(/[^\u1000-\u109F\uAA60-\uAA7F\uA9E0-\uA9FF\u104A\u104B\u104C\u104D\u104E\u104F0-9၀-၉၊။!?,.؟…'"“”‘’()\-\s]/g,'').replace(/\s{2,}/g,' ').trim();
}

async function retry(fn){
  let last;
  for(let i=0;i<4;i++){
    try{return await fn();}catch(e){last=e;const m=String(e?.message||e).toLowerCase();if(!/429|resource_exhausted|503|unavailable|500|502|504|timeout/.test(m)||i===3)throw e;await sleep(1000*(i+1));}
  }
  throw last;
}

async function uploadGeminiFile(ai,filePath,mimeType){
  return retry(()=>ai.files.upload({file:filePath,config:{mimeType,displayName:path.basename(filePath)}}));
}
async function waitActive(ai,file){
  for(let i=0;i<90;i++){
    const current=await ai.files.get({name:file.name});
    const state=String(current?.state||'').toUpperCase();
    if(state==='ACTIVE'||!state)return current;
    if(state==='FAILED')throw new Error('Gemini video processing failed');
    await sleep(2000);
  }
  throw new Error('Gemini video processing အချိန်ကြာလွန်းပါတယ်');
}
function mimeFromName(name){
  const ext=path.extname(name||'').toLowerCase();
  return ext==='.png'?'image/png':ext==='.webp'?'image/webp':'image/jpeg';
}
async function imageBlock(file){
  const data=file.buffer?file.buffer:await fs.readFile(file.path);
  return {type:'image',mime_type:file.mimetype||mimeFromName(file.originalname),data:data.toString('base64')};
}

async function generateTitleFromGemFile(ai, active, videoMime='video/mp4'){
  const response=await retry(()=>ai.models.generateContent({
    model:GEMINI_TEXT_MODEL,
    contents:[
      {fileData:{fileUri:active.uri,mimeType:active.mimeType||videoMime}},
      {text:`Analyze this entire video carefully. Understand the actual story, main characters, relationships, conflict, turning point and most important event. Then write ONE strong Myanmar Burmese drama/recap title based ONLY on what is actually shown or said in the video. Make it catchy and emotional for a social-media recap thumbnail, but do not invent events or characters. Keep it short enough for a thumbnail. Return ONLY JSON: {"title":"..."}. No emojis, no quotes, no explanation.`}
    ],
    config:{temperature:0.2,maxOutputTokens:300}
  }));
  const parsed=extractJson(response?.text||'');
  const title=cleanText(parsed?.title||response?.text||'').replace(/^['"“”]+|['"“”]+$/g,'');
  if(!title)throw new Error('Gemini က Title မထုတ်ပေးနိုင်ပါ');
  return title;
}
async function generateTitle(ai, videoFile){
  const gemFile=await uploadGeminiFile(ai,videoFile.path,videoFile.mimetype||'video/mp4');
  const active=await waitActive(ai,gemFile);
  const title=await generateTitleFromGemFile(ai,active,videoFile.mimetype||'video/mp4');
  return {title,gemFile:active};
}

function collectImageOutput(interaction){
  const parts=[];
  if(interaction?.output_image?.data)parts.push({data:interaction.output_image.data,mimeType:interaction.output_image.mime_type||'image/jpeg'});
  for(const step of (interaction?.steps||[])){
    for(const block of (step?.content||[])){
      if(block?.type==='image'&&block?.data)parts.push({data:block.data,mimeType:block.mime_type||'image/jpeg'});
    }
  }
  return parts[0]||null;
}

async function generateThumbnail(ai, videoGemFile, characterFiles, styleFile, title, ratio, epEnabled, epText){
  const input=[
    {type:'video',uri:videoGemFile.uri,mime_type:videoGemFile.mimeType||'video/mp4'},
    ...(await Promise.all(characterFiles.map(imageBlock))),
    ...(styleFile?[await imageBlock(styleFile)]:[]),
    {type:'text',text:`Create a professional, high-impact Korean-drama / social-media recap thumbnail based on the entire video.

STORY: Analyze the video itself and choose the most important story moment and the most important characters. Do not invent unrelated events.

CHARACTER REFERENCES: The uploaded character reference images are identity references. Preserve the recognizable facial identity, hairstyle and key visual traits of each referenced person as closely as possible. Use them for the corresponding characters when appropriate. Do not replace them with random faces.

STYLE REFERENCE: The final design should have the same general visual energy as the provided style reference: premium drama-recap thumbnail composition, cinematic lighting, strong facial expressions, layered character arrangement, attractive color contrast, depth, clean professional framing. Do NOT copy logos, watermarks, usernames, or exact text from the style reference.

TEXT: Put ONLY this Burmese title as the main headline: "${title.replace(/"/g,'\\"')}"${epEnabled?`\nPut ONLY this episode label: "${cleanText(epText||'EP. 01').replace(/"/g,'\\"')}"`:''}
Do not add any other words, labels, captions, hashtags, logos, watermarks, channel names or decorative text.

TYPOGRAPHY: Use a bold, premium, highly readable Burmese display font style suitable for a viral drama recap thumbnail. Make the headline visually striking with strong contrast, tasteful outline/shadow and polished placement. Keep Burmese letters correctly shaped and readable.

COMPOSITION: Main characters large and clearly visible, faces unobstructed, emotionally expressive. Use cinematic depth, foreground/background separation, dramatic lighting and polished poster-like composition. The result should feel like a professionally designed drama recap cover, not a generic AI collage.

OUTPUT: One finished thumbnail image only. Aspect ratio: ${ratio}.`}
  ];
  const interaction=await retry(()=>ai.interactions.create({
    model:GEMINI_IMAGE_MODEL,
    input,
    response_format:{type:'image',mime_type:'image/jpeg',aspect_ratio:ratio,image_size:'1K'}
  }));
  const image=collectImageOutput(interaction);
  if(!image)throw new Error('Gemini က Final Thumbnail ပုံ မပြန်ပါ');
  return image;
}

async function transcribeGroq(audioPath,key){
  const groq=new Groq({apiKey:key});
  return groq.audio.transcriptions.create({file:createReadStream(audioPath),model:GROQ_MODEL,response_format:'verbose_json',timestamp_granularities:['word','segment'],temperature:0});
}
async function extractAudio(video){
  const out=path.join(TMP,`${crypto.randomUUID()}.wav`);
  await runProcess(ffmpegStatic,['-y','-i',video,'-map','0:a:0','-vn','-ar','16000','-ac','1','-c:a','pcm_s16le',out]);
  return out;
}
function normalizeWord(x){const word=cleanText(x?.word),start=Number(x?.start),end=Number(x?.end);return word&&Number.isFinite(start)&&Number.isFinite(end)&&end>start?{word,start,end}:null;}
function makePreciseSegments(result,duration){
  const words=(Array.isArray(result?.words)?result.words:[]).map(normalizeWord).filter(Boolean);
  if(!words.length)return(Array.isArray(result?.segments)?result.segments:[]).map((s,i)=>({id:i+1,start:Number(s.start)||0,end:Number(s.end)||0,text:cleanText(s.text)})).filter(s=>s.text&&s.end>s.start);
  const out=[];let cur=[],start=0,end=0;
  const flush=()=>{if(!cur.length)return;const text=cur.map(x=>x.word).join(' ').replace(/\s+([,.!?;:，。！？；：])/g,'$1').trim();if(text&&end>start)out.push({id:out.length+1,start,end:Math.min(duration,end),text});cur=[];};
  for(const w of words){if(!cur.length){cur=[w];start=w.start;end=w.end;continue;}const gap=w.start-end;const text=cur.map(x=>x.word).join(' ');if(gap>.55||w.end-start>5||text.length>55||cur.length>=11){flush();cur=[w];start=w.start;end=w.end;}else{cur.push(w);end=w.end;}}flush();return out;
}
function srtTime(v){const ms=Math.max(0,Math.round(Number(v||0)*1000)),h=Math.floor(ms/3600000),m=Math.floor(ms%3600000/60000),s=Math.floor(ms%60000/1000),x=ms%1000;return`${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')},${String(x).padStart(3,'0')}`;}
function makeSrt(list){return list.map((s,i)=>`${i+1}\n${srtTime(s.start)} --> ${srtTime(s.end)}\n${s.text}\n`).join('\n');}

const downloads=new Map();
function registerDownload(file,filename){const token=crypto.randomUUID();downloads.set(token,{file,filename,expires:Date.now()+30*60*1000});return token;}
setInterval(()=>{const now=Date.now();for(const [t,v] of downloads){if(v.expires<=now){downloads.delete(t);cleanup(v.file);}}},60000).unref();

app.get('/api/health',(_req,res)=>res.json({ok:true,version:APP_VERSION,geminiTextModel:GEMINI_TEXT_MODEL,geminiImageModel:GEMINI_IMAGE_MODEL,thumbnail:'VIDEO+CHARACTER+STYLE->AI_IMAGE',ffmpegThumbnail:false,burnIn:true,fontReady:fontState.ok}));
app.get('/api/download/:token',(req,res)=>{const v=downloads.get(req.params.token);if(!v)return res.status(404).send('Download expired');downloads.delete(req.params.token);res.download(v.file,v.filename,()=>cleanup(v.file));});

app.post('/api/thumbnail/title',uploadVideo.single('video'),async(req,res)=>{
  const video=req.file?.path;
  try{
    if(!video)throw new Error('Video file မရှိပါ');
    const info=await probeVideo(video);if(info.duration>MAX_SECONDS)throw new Error('Video က 5 မိနစ်ထက်မကျော်ရပါ');
    const key=getKey(req,'x-gemini-api-key','geminiApiKey','GEMINI_API_KEY','Gemini API Key');
    const ai=new GoogleGenAI({apiKey:key});const result=await generateTitle(ai,req.file);res.json({ok:true,title:result.title});
  }catch(e){console.error('TITLE ERROR',e);res.status(400).json({ok:false,error:e?.message||'AI Title Error'});}finally{cleanup(video);}
});

// Combined route: video + up to 4 character refs + 1 style ref in one multipart request.
const thumbnailUpload=multer({storage:videoStorage,limits:{fileSize:MAX_BYTES,files:6},fileFilter:(_req,file,cb)=>{
  const ext=path.extname(file.originalname||'').toLowerCase();
  if(file.fieldname==='video'){if(!videoExt.has(ext))return cb(new Error('Video format မမှန်ပါ'));}
  else {if(!imageExt.has(ext))return cb(new Error('Reference image format မမှန်ပါ'));}
  cb(null,true);
}});
app.post('/api/thumbnail/generate-ai',thumbnailUpload.fields([{name:'video',maxCount:1},{name:'character',maxCount:4},{name:'style',maxCount:1}]),async(req,res)=>{
  const video=req.files?.video?.[0];const chars=req.files?.character||[];const style=req.files?.style?.[0];
  try{
    if(!video)throw new Error('Video file မရှိပါ');
    const info=await probeVideo(video.path);if(info.duration>MAX_SECONDS)throw new Error('Video က 5 မိနစ်ထက်မကျော်ရပါ');
    const key=getKey(req,'x-gemini-api-key','geminiApiKey','GEMINI_API_KEY','Gemini API Key');
    const ai=new GoogleGenAI({apiKey:key});
    let title=cleanText(req.body?.title);
    const gemVideo=await uploadGeminiFile(ai,video.path,video.mimetype||'video/mp4');await waitActive(ai,gemVideo);
    if(!title) title=await generateTitleFromGemFile(ai,gemVideo,video.mimetype||'video/mp4');
    const ratio=['16:9','9:16','1:1'].includes(String(req.body?.ratio))?String(req.body.ratio):'16:9';
    const epEnabled=String(req.body?.epEnabled)!=='false';
    const epText=cleanText(req.body?.epText||'EP. 01');
    const image=await generateThumbnail(ai,gemVideo,chars,style,title,ratio,epEnabled,epText);
    const output=path.join(TMP,`${crypto.randomUUID()}-YNT-AI-Thumbnail.jpg`);await fs.writeFile(output,Buffer.from(image.data,'base64'));
    const token=registerDownload(output,'YNT-AI-Thumbnail.jpg');
    res.json({ok:true,title,ratio,download:`/api/download/${token}`,mimeType:image.mimeType});
  }catch(e){console.error('AI THUMBNAIL ERROR',e);res.status(400).json({ok:false,error:e?.message||'AI Thumbnail Error'});}finally{
    cleanup(video?.path);for(const f of chars)cleanup(f.path);cleanup(style?.path);
  }
});

app.post('/api/transcribe',uploadVideo.single('video'),async(req,res)=>{
  const video=req.file?.path;let audio=null;
  try{
    if(!video)throw new Error('Video file မရှိပါ');const info=await probeVideo(video);if(info.duration>MAX_SECONDS)throw new Error('Video က 5 မိနစ်ထက်မကျော်ရပါ');
    const key=getKey(req,'x-groq-api-key','groqApiKey','GROQ_API_KEY','Groq API Key');audio=await extractAudio(video);const result=await transcribeGroq(audio,key);const transcript=makePreciseSegments(result,info.duration);res.json({ok:true,transcript,text:result?.text||''});
  }catch(e){console.error('TRANSCRIBE',e);res.status(400).json({ok:false,error:e?.message||'Groq Transcript Error'});}finally{cleanup(video);cleanup(audio);}
});

// Safely extract a JSON array from Gemini output. Handles markdown fences and
// ignores square brackets that appear inside quoted strings.
function parseTranslationArray(text){
  const raw=String(text||'').trim().replace(/^```(?:json)?\s*/i,'').replace(/\s*```$/,'').trim();
  try{const value=JSON.parse(raw);if(Array.isArray(value))return value;}catch{}
  let start=-1,depth=0,inString=false,escaped=false;
  for(let i=0;i<raw.length;i++){
    const ch=raw[i];
    if(inString){if(escaped)escaped=false;else if(ch==='\\')escaped=true;else if(ch==='"')inString=false;continue;}
    if(ch==='"'){inString=true;continue;}
    if(ch==='['){if(start<0)start=i;depth++;}
    else if(ch===']'&&start>=0){depth--;if(depth===0){try{const value=JSON.parse(raw.slice(start,i+1));if(Array.isArray(value))return value;}catch{}start=-1;}}
  }
  return null;
}

app.post('/api/translate',async(req,res)=>{
  try{
    const key=getKey(req,'x-gemini-api-key','geminiApiKey','GEMINI_API_KEY','Gemini API Key');
    const list=Array.isArray(req.body?.transcript)?req.body.transcript:[];
    if(!list.length)throw new Error('Transcript မရှိပါ');
    const ai=new GoogleGenAI({apiKey:key});
    const source=list.map((x,i)=>({id:Number(x.id)||i+1,text:cleanText(x.text)}));
    const prompt=`You are a subtitle translator. Translate each input line into natural Myanmar Burmese Unicode. Preserve the meaning and keep each line separate. Return ONLY a valid JSON array, no markdown or commentary. Every item must have exactly two fields: id (number) and translation (string). Escape all quotation marks and special characters correctly. Do not put literal line breaks inside JSON strings. Do not omit, merge, reorder, duplicate, or invent ids. Translation text must be Myanmar only.\nINPUT JSON:\n${JSON.stringify(source)}`;
    let response=await retry(()=>ai.models.generateContent({model:GEMINI_TEXT_MODEL,contents:prompt,config:{temperature:0,maxOutputTokens:12000,responseMimeType:'application/json'}}));
    let parsed=parseTranslationArray(response?.text||'');
    // If Gemini returns malformed JSON, ask it once to repair its own output.
    if(!parsed){
      const repairPrompt=`Convert the following content into valid JSON only. Return a JSON array of objects with numeric id and string translation fields. Keep all translation wording and ids, escape quotes and control characters correctly, and do not add commentary.\nEXPECTED IDS: ${JSON.stringify(source.map(x=>x.id))}\nBROKEN OUTPUT:\n${String(response?.text||'').slice(0,24000)}`;
      response=await retry(()=>ai.models.generateContent({model:GEMINI_TEXT_MODEL,contents:repairPrompt,config:{temperature:0,maxOutputTokens:12000,responseMimeType:'application/json'}}));
      parsed=parseTranslationArray(response?.text||'');
    }
    if(!parsed)throw new Error('Gemini ဘာသာပြန်အဖြေ JSON ပုံစံမမှန်ပါ။ ထပ်စမ်းကြည့်ပါ။');
    const translatedMap=new Map();
    for(const item of parsed){
      const id=Number(item?.id);const value=keepMyanmarOnly(item?.translation);
      if(Number.isFinite(id)&&value)translatedMap.set(id,value);
    }
    const out=list.map((x,i)=>({...x,text:translatedMap.get(Number(x.id)||i+1)||''})).filter(x=>x.text);
    if(!out.length)throw new Error('မြန်မာစာ ဘာသာပြန်ရလဒ် မရပါ။ ထပ်စမ်းကြည့်ပါ။');
    res.json({ok:true,transcript:out,srt:makeSrt(out)});
  }catch(e){console.error('TRANSLATE ERROR:',e);res.status(400).json({ok:false,error:e?.message||'Gemini Translation Error'});}
});


// ---------- Burn-in: blur original text + Myanmar subtitles -> final MP4 ----------
const burnJobs=new Map();
const uploadBurn=multer({storage:videoStorage,limits:{fileSize:MAX_BYTES,files:1},fileFilter:(_req,file,cb)=>{
  const ext=path.extname(file.originalname||'').toLowerCase();
  if(!videoExt.has(ext))return cb(new Error('Video format မမှန်ပါ'));cb(null,true);
}});
setInterval(()=>{const now=Date.now();for(const [id,j] of burnJobs)if(now-j.created>60*60*1000)burnJobs.delete(id);},60000).unref();

async function runBurnJob(job,{video,srtText,settings}){
  let assPath=null,output=null;
  try{
    const info=await probeVideo(video);
    if(info.duration>MAX_SECONDS)throw new Error('Video က 5 မိနစ်ထက်မကျော်ရပါ');
    const W=info.width-info.width%2,H=info.height-info.height%2;
    const probe=await runProcess(ffprobeStatic.path,['-v','error','-select_streams','a','-show_entries','stream=index','-of','csv=p=0',video]).catch(()=>({stdout:''}));
    const hasAudio=Boolean(String(probe.stdout||'').trim());
    const sub=settings?.sub||{};
    const cues=sub.enabled===false?[]:parseSrt(srtText);
    if(cues.length){
      if(!fontState.ok)fontState=await ensureFont(FONTS);
      if(!fontState.ok)throw new Error('Myanmar font မရှိပါ။ server ရဲ့ ./fonts ထဲမှာ Myanmar .ttf ထည့်ပါ (ဥပမာ NotoSansMyanmar-Bold.ttf)');
      assPath=path.join(TMP,`${crypto.randomUUID()}.ass`);
      await fs.writeFile(assPath,buildAss(cues,W,H,sub,FONT_FAMILY),'utf8');
    }
    const boxes=Array.isArray(settings?.blurs)?settings.blurs.slice(0,12):[];
    const {graph}=buildFilterGraph({W,H,boxes,strength:settings?.blurStrength,assPath,fontsDir:FONTS});
    output=path.join(TMP,`${crypto.randomUUID()}-YNT-Final.mp4`);
    const args=buildFfmpegArgs({input:video,output,graph,hasAudio});
    await new Promise((resolve,reject)=>{
      const child=spawn(ffmpegStatic,args);let err='',buf='';
      child.stdout.on('data',d=>{buf+=d.toString();const lines=buf.split('\n');buf=lines.pop();for(const l of lines){const m=l.match(/^out_time_(?:us|ms)=(\d+)/);if(m){const sec=Number(m[1])/1e6;job.progress=Math.min(99,Math.round(sec/info.duration*100));}}});
      child.stderr.on('data',d=>{err=(err+d.toString()).slice(-4000);});
      child.on('error',reject);
      child.on('close',code=>code===0?resolve():reject(new Error('ffmpeg error: '+err.split('\n').slice(-6).join(' ').trim())));
    });
    const token=registerDownload(output,'YNT-Final-Myanmar-Subtitle.mp4');
    job.status='done';job.progress=100;job.download=`/api/download/${token}`;output=null;
  }catch(e){console.error('BURN ERROR',e);job.status='error';job.error=e?.message||'Burn-in Error';}
  finally{cleanup(video);cleanup(assPath);cleanup(output);}
}

app.post('/api/burn',uploadBurn.single('video'),(req,res)=>{
  const video=req.file?.path;
  try{
    if(!video)throw new Error('Video file မရှိပါ');
    let settings={};try{settings=JSON.parse(String(req.body?.settings||'{}'));}catch{}
    const id=crypto.randomUUID();
    const job={id,status:'running',progress:0,created:Date.now()};
    burnJobs.set(id,job);
    runBurnJob(job,{video,srtText:String(req.body?.srt||''),settings});
    res.json({ok:true,jobId:id});
  }catch(e){cleanup(video);res.status(400).json({ok:false,error:e?.message||'Burn-in Error'});}
});
app.get('/api/burn/:id',(req,res)=>{
  const j=burnJobs.get(req.params.id);
  if(!j)return res.status(404).json({ok:false,error:'Job မတွေ့ပါ (သက်တမ်းကုန်သွားနိုင်ပါတယ်)'});
  res.json({ok:true,status:j.status,progress:j.progress,error:j.error,download:j.download});
});

app.use((err,_req,res,_next)=>{console.error('REQUEST ERROR',err);res.status(400).json({ok:false,error:err?.message||'Request Error'});});
app.get('/{*splat}',(_req,res)=>res.sendFile(path.join(PUBLIC,'index.html')));
app.listen(PORT,()=>console.log(`YNT Thumbnail & SRT ${APP_VERSION} listening on ${PORT}`));
