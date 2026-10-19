// Burn-in helpers: blur regions + Myanmar subtitles -> ffmpeg arguments.
import fs from 'node:fs/promises';
import path from 'node:path';

export const FONT_FILE = 'NotoSansMyanmar-Bold.ttf';
export const FONT_FAMILY = 'Noto Sans Myanmar';
const FONT_URLS = [
  'https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSansMyanmar/full/ttf/NotoSansMyanmar-Bold.ttf',
  'https://raw.githubusercontent.com/notofonts/notofonts.github.io/main/fonts/NotoSansMyanmar/full/ttf/NotoSansMyanmar-Bold.ttf',
  'https://github.com/google/fonts/raw/main/ofl/notosansmyanmar/NotoSansMyanmar%5Bwdth%2Cwght%5D.ttf'
];

// Make sure a Myanmar-capable font exists in fontsDir. Users can also drop their own
// .ttf/.otf file into the fonts folder (the first Myanmar-capable font found is used).
export async function ensureFont(fontsDir) {
  await fs.mkdir(fontsDir, { recursive: true });
  const target = path.join(fontsDir, FONT_FILE);
  try { const s = await fs.stat(target); if (s.size > 50_000) return { ok: true, family: FONT_FAMILY }; } catch {}
  for (const url of FONT_URLS) {
    try {
      const r = await fetch(url, { redirect: 'follow' });
      if (!r.ok) continue;
      const buf = Buffer.from(await r.arrayBuffer());
      if (buf.length < 50_000) continue;
      await fs.writeFile(target, buf);
      return { ok: true, family: FONT_FAMILY };
    } catch {}
  }
  return { ok: false, family: FONT_FAMILY };
}

export function parseSrt(text) {
  const blocks = String(text || '').replace(/\r/g, '').replace(/^\ufeff/, '').trim().split(/\n{2,}/);
  const out = [];
  for (const block of blocks) {
    const lines = block.split('\n').map(l => l.trimEnd());
    const ti = lines.findIndex(l => l.includes('-->'));
    if (ti < 0) continue;
    const m = lines[ti].match(/(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)/);
    if (!m) continue;
    const t = (h, mi, s, ms) => Number(h) * 3600 + Number(mi) * 60 + Number(s) + Number(String(ms).padEnd(3, '0').slice(0, 3)) / 1000;
    const start = t(m[1], m[2], m[3], m[4]);
    const end = t(m[5], m[6], m[7], m[8]);
    const txt = lines.slice(ti + 1).join(' ').replace(/\s+/g, ' ').trim();
    if (txt && end > start) out.push({ start, end, text: txt });
  }
  return out;
}

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const num = (v, d) => (Number.isFinite(Number(v)) ? Number(v) : d);

function assTime(sec) {
  const cs = Math.max(0, Math.round(sec * 100));
  const h = Math.floor(cs / 360000), m = Math.floor(cs % 360000 / 6000), s = Math.floor(cs % 6000 / 100), c = cs % 100;
  return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(c).padStart(2, '0')}`;
}
// ASS colour is &HAABBGGRR
function assColor(hex, fallback = '#ffffff') {
  const h = /^#?[0-9a-f]{6}$/i.test(String(hex || '')) ? String(hex).replace('#', '') : fallback.replace('#', '');
  const r = h.slice(0, 2), g = h.slice(2, 4), b = h.slice(4, 6);
  return `&H00${b}${g}${r}`.toUpperCase();
}
function escapeAss(t) {
  return String(t).replace(/\\/g, '').replace(/[{}]/g, '').replace(/\n/g, '\\N');
}

export function buildAss(cues, W, H, sub, fontFamily = FONT_FAMILY) {
  const sizePct = clamp(num(sub?.sizePct, 3.6), 1.5, 12);
  const outlinePct = clamp(num(sub?.outlinePct, 0.28), 0, 2);
  const fontSize = Math.round(H * sizePct / 100);
  const outline = Math.max(0, +(H * outlinePct / 100).toFixed(2));
  const px = Math.round(clamp(num(sub?.x, 0.5), 0, 1) * W);
  const py = Math.round(clamp(num(sub?.y, 0.8), 0, 1) * H);
  const bold = sub?.bold === false ? 0 : -1;
  const marginLR = Math.round(W * 0.06);
  const primary = assColor(sub?.color, '#ffe600');
  const outlineC = assColor(sub?.outlineColor, '#000000');
  const header = `[Script Info]
ScriptType: v4.00+
PlayResX: ${W}
PlayResY: ${H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,${fontFamily},${fontSize},${primary},${primary},${outlineC},&H80000000,${bold},0,0,0,100,100,0,0,1,${outline},0,5,${marginLR},${marginLR},10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
`;
  const lines = cues.map(c => `Dialogue: 0,${assTime(c.start)},${assTime(c.end)},Default,,0,0,0,,{\\an5\\pos(${px},${py})}${escapeAss(c.text)}`);
  return header + lines.join('\n') + '\n';
}

function escapeFilterPath(p) {
  return String(p).replace(/\\/g, '/').replace(/:/g, '\\:').replace(/'/g, "\\'").replace(/,/g, '\\,').replace(/\[/g, '\\[').replace(/\]/g, '\\]');
}

// boxes: [{x,y,w,h}] normalised 0..1 relative to the frame.
export function buildFilterGraph({ W, H, boxes, strength, assPath, fontsDir }) {
  const sigma = clamp(num(strength, 14), 2, 60);
  const steps = [];
  const valid = [];
  for (const b of Array.isArray(boxes) ? boxes : []) {
    let x = Math.round(clamp(num(b.x, 0), 0, 1) * W), y = Math.round(clamp(num(b.y, 0), 0, 1) * H);
    let w = Math.round(clamp(num(b.w, 0), 0, 1) * W), h = Math.round(clamp(num(b.h, 0), 0, 1) * H);
    x = clamp(x, 0, W - 2); y = clamp(y, 0, H - 2);
    w = clamp(w, 2, W - x); h = clamp(h, 2, H - y);
    w -= w % 2; h -= h % 2; x -= x % 2; y -= y % 2;
    if (w >= 2 && h >= 2) valid.push({ x, y, w, h });
  }
  let cur = 'base';
  steps.push(`[0:v]scale=${W}:${H},setsar=1[${cur}]`);
  valid.forEach((b, i) => {
    const a = `a${i}`, s = `s${i}`, bl = `b${i}`, out = `o${i}`;
    steps.push(`[${cur}]split=2[${a}][${s}]`);
    steps.push(`[${s}]crop=${b.w}:${b.h}:${b.x}:${b.y},gblur=sigma=${sigma}:steps=2[${bl}]`);
    steps.push(`[${a}][${bl}]overlay=${b.x}:${b.y}[${out}]`);
    cur = out;
  });
  if (assPath) {
    steps.push(`[${cur}]ass='${escapeFilterPath(assPath)}':fontsdir='${escapeFilterPath(fontsDir)}'[vsub]`);
    cur = 'vsub';
  }
  steps.push(`[${cur}]format=yuv420p[vout]`);
  return { graph: steps.join(';'), blurCount: valid.length };
}

export function buildFfmpegArgs({ input, output, graph, hasAudio }) {
  return [
    '-y', '-i', input,
    '-filter_complex', graph,
    '-map', '[vout]',
    ...(hasAudio ? ['-map', '0:a:0?', '-c:a', 'aac', '-b:a', '160k'] : []),
    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
    '-movflags', '+faststart',
    '-progress', 'pipe:1', '-nostats',
    output
  ];
}
