// Scene timing + SRT helpers.
//
// Why this file exists: narration is now synthesized ONE SCENE AT A TIME, so the
// real speech length of every scene is known exactly. These helpers decide how
// to fit that speech to the scene's video slot, and build an SRT whose cue times
// come from the measured speech (not from guesses).
import { splitNarration } from "./burn.js";

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export const TIMING_DEFAULTS = {
  minTempo: 0.9,       // never slow the voice below 90% (of the style speed)
  maxTempo: 1.25,      // never speed the voice above 125% (of the style speed)
  maxTotalTempo: 1.5,  // absolute ceiling for the final voice speed
  maxVideoSlow: 1.2,   // slow the video at most 1.2x before freezing the last frame
  tail: 0.2            // breathing gap kept after the last word of a scene
};

/**
 * Decide how to fit one scene's speech into its video slot.
 *
 *  slot   - seconds of original footage this scene covers
 *  speech - seconds of real speech at normal speed (silence already trimmed)
 *  base   - style speed (1.0 natural, 1.1 recap, 1.25 short ...)
 *
 * Voice speed is the only thing changed first (within limits). If the speech
 * still does not fit, the video is slowed a little, and any remaining gap is a
 * frozen last frame. Footage is never taken from the NEXT scene, so what you
 * see always belongs to what is being said.
 */
export function planSceneTiming({ slot, speech, base = 1, ...overrides }) {
  const o = { ...TIMING_DEFAULTS, ...overrides };
  slot = Math.max(0.3, Number(slot) || 0.3);
  speech = Math.max(0.05, Number(speech) || 0.05);
  base = Math.max(0.5, Number(base) || 1);

  const room = Math.max(0.2, slot - o.tail);       // time available for speech
  const need = speech / room;                      // tempo that exactly fills it
  const lo = base * o.minTempo;
  const hi = Math.min(base * o.maxTempo, o.maxTotalTempo);
  const tempo = clamp(need, Math.min(lo, hi), hi);

  const speechDur = speech / tempo;
  const outDur = Math.max(slot, speechDur + o.tail);
  const overflow = outDur - slot;

  const videoStretch = overflow > 0.02 ? Math.min(o.maxVideoSlow, outDur / slot) : 1;
  const padSec = Math.max(0, outDur - slot * videoStretch);

  return { tempo, speechDur, outDur, videoStretch, padSec, overflow };
}

/**
 * Append "slow video + freeze last frame" to a filter_complex that ends in
 * [vout] (or create one when there is no filter yet).
 */
export function appendVideoTiming(filterComplex, stretch, padSec) {
  const steps = [];
  if (stretch > 1.001) steps.push(`setpts=${stretch.toFixed(5)}*PTS`);
  if (padSec > 0.02) steps.push(`tpad=stop_mode=clone:stop_duration=${padSec.toFixed(3)}`);
  if (!steps.length) return filterComplex;
  const chain = steps.join(",");
  if (!filterComplex) return `[0:v]${chain}[vout]`;
  if (!/\[vout\]\s*$/.test(filterComplex)) throw new Error("filter_complex must end with [vout]");
  return `${filterComplex.replace(/\[vout\]\s*$/, "[vpre]")};[vpre]${chain}[vout]`;
}

/** Cut silence from the start and end of a TTS clip (keeps pauses inside). */
export const TRIM_SILENCE_FILTER = [
  "silenceremove=start_periods=1:start_duration=0.02:start_threshold=-45dB:detection=peak",
  "areverse",
  "silenceremove=start_periods=1:start_duration=0.02:start_threshold=-45dB:detection=peak",
  "areverse"
].join(",");

/* ---------------- SRT ---------------- */

export function srtTime(seconds) {
  const ms = Math.max(0, Math.round(Number(seconds) * 1000));
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const r = ms % 1000;
  const p = (n, w = 2) => String(n).padStart(w, "0");
  return `${p(h)}:${p(m)}:${p(s)},${p(r, 3)}`;
}

/**
 * Split one scene's narration into subtitle cues spread over the time the
 * voice is really speaking: [offset, offset + speechDur].
 */
export function makeSceneCues(narration, offset, speechDur, maxChars = 42) {
  const chunks = splitNarration(narration, maxChars);
  if (!chunks.length || !(speechDur > 0.05)) return [];

  // Very short cues flash by too fast to read: pack neighbouring sentences
  // together while they still fit on one cue.
  const merged = [];
  for (const chunk of chunks) {
    const last = merged.length - 1;
    if (last >= 0 && (chunk.length < 6 || merged[last].length + 1 + chunk.length <= maxChars)) {
      merged[last] = `${merged[last]} ${chunk}`;
    } else {
      merged.push(chunk);
    }
  }

  const total = merged.reduce((sum, c) => sum + c.length, 0);
  const cues = [];
  let t = offset;
  merged.forEach((text, i) => {
    const dur = i === merged.length - 1
      ? offset + speechDur - t
      : speechDur * text.length / total;
    cues.push({ start: t, end: t + Math.max(0.2, dur), text });
    t += dur;
  });
  return cues;
}

/** cues: [{start,end,text}] in seconds -> SRT text. */
export function buildSrt(cues) {
  let lastEnd = 0;
  return cues
    .map((cue, i) => {
      const start = Math.max(cue.start, lastEnd);
      const end = Math.max(cue.end, start + 0.2);
      lastEnd = end;
      return `${i + 1}\n${srtTime(start)} --> ${srtTime(end)}\n${cue.text}\n`;
    })
    .join("\n");
}
