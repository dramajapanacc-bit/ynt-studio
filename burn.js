// Burn-in helpers: Burmese subtitles (ASS / libass) + blur boxes (ffmpeg filter_complex).
import fs from "fs";
import { execFile } from "child_process";
import { promisify } from "util";
import path from "path";
import { fileURLToPath } from "url";

const execFileAsync = promisify(execFile);

const SUBTITLE_FONT = process.env.SUBTITLE_FONT || "Padauk";
const SUBTITLE_FONT_SCALE = Number(process.env.SUBTITLE_FONT_SCALE) || 1;
const HERE = path.dirname(fileURLToPath(import.meta.url));
// Project "fonts" folder first (drop NotoSansMyanmar-Regular/Bold.ttf in there),
// then common system locations.
const FONT_DIRS = [
  path.join(HERE, "fonts"),
  path.join(HERE, "fonts-runtime"),
  "/usr/share/fonts/truetype/noto",
  "/usr/share/fonts/opentype/noto",
  "/usr/share/fonts/truetype/padauk",
  "/usr/share/fonts/truetype",
  "/usr/share/fonts"
];

// Fallback for hosts without system fonts (e.g. Render "Node" runtime): the npm
// package @expo-google-fonts/noto-sans-myanmar ships the .ttf files. They are
// copied flat into ./fonts-runtime so libass can load them.
const RUNTIME_FONT_DIR = path.join(HERE, "fonts-runtime");
function prepareNpmFonts() {
  try {
    if (hasFontFiles(RUNTIME_FONT_DIR)) return true;
    const root = path.join(HERE, "node_modules", "@expo-google-fonts", "noto-sans-myanmar");
    if (!fs.existsSync(root)) return false;
    const found = [];
    const walk = dir => {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) walk(full);
        else if (/NotoSansMyanmar.*\.ttf$/i.test(entry.name) && !/Condensed|Expanded|Semi|Extra|Light|Thin|Black/i.test(entry.name)) found.push(full);
      }
    };
    walk(root);
    if (!found.length) return false;
    fs.mkdirSync(RUNTIME_FONT_DIR, { recursive: true });
    for (const file of found) {
      // Expo names: NotoSansMyanmar_400Regular.ttf / _700Bold.ttf -> keep as is
      fs.copyFileSync(file, path.join(RUNTIME_FONT_DIR, path.basename(file)));
    }
    return true;
  } catch (error) {
    console.warn("[FONT] npm font prepare failed:", String(error?.message || error).slice(0, 160));
    return false;
  }
}

const hasFontFiles = dir => {
  try { return fs.readdirSync(dir).some(name => /\.(ttf|otf)$/i.test(name)); } catch { return false; }
};

// Warn loudly at startup if no Myanmar font is installed (subtitles would be □□□). Padauk is the default because its Myanmar glyph shaping is reliable in FFmpeg/libass.
export async function checkSubtitleFont() {
  if (prepareNpmFonts()) { console.log("[FONT] Myanmar font ready from npm package (fonts-runtime)"); return true; }
  try {
    const { stdout } = await execFileAsync("fc-list", [":lang=my", "family"]);
    if (stdout.trim()) { console.log(`[FONT] Myanmar font OK: ${stdout.split("\n")[0].trim()}`); return true; }
  } catch {}
  if (hasFontFiles(FONT_DIRS[0])) { console.log("[FONT] using fonts from ./fonts"); return true; }
  console.warn("[FONT] ⚠ No Myanmar font found — burned subtitles will show □□□. Install `fonts-sil-padauk` (apt) or put Padauk-Regular.ttf and Padauk-Bold.ttf into the ./fonts folder.");
  return false;
}
const PRESET_Y = { top: 14, middle: 50, bottom: 84 };

const num = (v, d, lo, hi) => {
  const n = Number(v);
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : d;
};

/* ---------- sanitize the payload coming from the browser ---------- */

export function normalizeBurn(raw) {
  let b = raw;
  if (typeof b === "string") {
    try { b = JSON.parse(b); } catch { return null; }
  }
  if (!b || typeof b !== "object") return null;

  const sub = b.subtitle || {};
  const blur = b.blur || {};

  const boxes = (Array.isArray(blur.boxes) ? blur.boxes : [])
    .slice(0, 6)
    .map(item => {
      const x = num(item?.x, 0, 0, 1);
      const y = num(item?.y, 0, 0, 1);
      return {
        x,
        y,
        w: Math.min(num(item?.w, 0, 0, 1), 1 - x),
        h: Math.min(num(item?.h, 0, 0, 1), 1 - y)
      };
    })
    .filter(item => item.w > 0.01 && item.h > 0.01);

  return {
    subtitle: {
      enabled: sub.enabled === true,
      sizePct: num(sub.sizePct, 3.6, 1, 12),
      color: /^#[0-9a-f]{6}$/i.test(String(sub.color)) ? String(sub.color) : "#ffe600",
      x: num(sub.xPct, 50, 0, 100),
      y: num(sub.yPct, PRESET_Y[sub.position] ?? 84, 0, 100)
    },
    blur: {
      enabled: blur.enabled === true && boxes.length > 0,
      boxes,
      strength: num(blur.strength, 20, 1, 60)
    }
  };
}

export function burnActive(burn) {
  return !!burn && (burn.subtitle.enabled || burn.blur.enabled);
}

/* ---------- video size (as displayed, rotation aware) ---------- */

export async function probeVideoSize(videoPath) {
  const { stdout } = await execFileAsync("ffprobe", [
    "-v", "error",
    "-select_streams", "v:0",
    "-show_entries", "stream=width,height:stream_side_data=rotation",
    "-of", "json",
    videoPath
  ]);
  const stream = JSON.parse(stdout).streams?.[0];
  if (!stream?.width || !stream?.height) throw new Error("Cannot read video size.");
  const rotation = Math.abs(Number(stream.side_data_list?.[0]?.rotation || 0)) % 180;
  return rotation === 90
    ? { width: stream.height, height: stream.width }
    : { width: stream.width, height: stream.height };
}

/* ---------- subtitle text splitting ---------- */

// Burmese combining marks / virama: never cut right before or after these.
const MY_JOINER = /[\u102B-\u103E\u1056-\u1059\u105E-\u1060\u1062-\u1064\u1067-\u106D\u1071-\u1074\u1082-\u108D\u108F\u109A-\u109D]/;

function hardCut(text, max) {
  let cut = max;
  while (cut > 1 && (MY_JOINER.test(text[cut]) || text[cut - 1] === "\u1039")) cut--;
  return cut;
}

export function splitNarration(text, maxChars) {
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  if (!clean) return [];

  const sentences = clean.split(/(?<=[။၊!?.])\s*/).filter(Boolean);
  const chunks = [];

  for (const sentence of sentences) {
    let rest = sentence.trim();
    while (rest.length > maxChars) {
      let cut = rest.lastIndexOf(" ", maxChars);
      if (cut < maxChars * 0.4) cut = hardCut(rest, maxChars);
      chunks.push(rest.slice(0, cut).trim());
      rest = rest.slice(cut).trim();
    }
    if (rest) chunks.push(rest);
  }
  return chunks;
}

/* ---------- ASS file ---------- */

function assTime(seconds) {
  const cs = Math.max(0, Math.round(seconds * 100));
  const h = Math.floor(cs / 360000);
  const m = Math.floor((cs % 360000) / 6000);
  const s = Math.floor((cs % 6000) / 100);
  const c = cs % 100;
  return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(c).padStart(2, "0")}`;
}

function assColor(hex) {
  const r = hex.slice(1, 3), g = hex.slice(3, 5), b = hex.slice(5, 7);
  return `&H00${b}${g}${r}`.toUpperCase();
}

const assEscape = text => text.replace(/\\/g, "").replace(/[{}]/g, "").replace(/\n/g, "\\N");

export function writeAss(assPath, size, sub, narration, duration) {
  const { width, height } = size;
  const fontSize = Math.max(8, Math.round(height * sub.sizePct / 100 * SUBTITLE_FONT_SCALE));
  const perLine = Math.max(8, Math.floor((width * 0.9) / (fontSize * 0.5)));
  const chunks = splitNarration(narration, Math.min(90, Math.max(16, perLine * 2)));
  if (!chunks.length || !(duration > 0.2)) return false;

  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const margin = Math.round(width * 0.04);
  const outline = Math.max(1, Math.round(fontSize / 14));
  const px = Math.round(width * sub.x / 100);
  const py = Math.round(height * sub.y / 100);

  let t = 0;
  const events = chunks.map(chunk => {
    const start = t;
    t += duration * chunk.length / total;
    return `Dialogue: 0,${assTime(start)},${assTime(t)},Default,,0,0,0,,{\\an5\\pos(${px},${py})}${assEscape(chunk)}`;
  });

  const ass = [
    "[Script Info]",
    "ScriptType: v4.00+",
    `PlayResX: ${width}`,
    `PlayResY: ${height}`,
    "WrapStyle: 0",
    "ScaledBorderAndShadow: yes",
    "",
    "[V4+ Styles]",
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
    `Style: Default,${SUBTITLE_FONT},${fontSize},${assColor(sub.color)},&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,${outline},1,5,${margin},${margin},0,1`,
    "",
    "[Events]",
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ...events,
    ""
  ].join("\n");

  fs.writeFileSync(assPath, ass, "utf8");
  return true;
}

/* ---------- ffmpeg filter_complex ---------- */

const quote = value => `'${String(value).replace(/\\/g, "/").replace(/'/g, "")}'`;

// Returns a filter_complex string ending in [vout], or null when nothing to do.
export function buildBurnFilter(burn, size, assPath) {
  const { width: W, height: H } = size;
  const parts = [];
  let current = "0:v";

  if (burn.blur.enabled) {
    const boxes = burn.blur.boxes;
    const sigma = Math.max(1.5, burn.blur.strength * W / 1280).toFixed(2);
    parts.push(`[0:v]split=${boxes.length + 1}[base]${boxes.map((_, i) => `[s${i}]`).join("")}`);
    current = "base";
    boxes.forEach((box, i) => {
      const x = Math.min(W - 4, Math.round(box.x * W));
      const y = Math.min(H - 4, Math.round(box.y * H));
      const w = Math.max(4, Math.min(W - x, Math.round(box.w * W)));
      const h = Math.max(4, Math.min(H - y, Math.round(box.h * H)));
      parts.push(`[s${i}]crop=${w}:${h}:${x}:${y},gblur=sigma=${sigma}[g${i}]`);
      parts.push(`[${current}][g${i}]overlay=${x}:${y}[o${i}]`);
      current = `o${i}`;
    });
  }

  if (assPath) {
    const fontsDir = FONT_DIRS.find(hasFontFiles);
    parts.push(
      `[${current}]subtitles=filename=${quote(assPath)}${fontsDir ? `:fontsdir=${quote(fontsDir)}` : ""}[vout]`
    );
  } else if (parts.length) {
    parts.push(`[${current}]null[vout]`);
  } else {
    return null;
  }
  return parts.join(";");
}
