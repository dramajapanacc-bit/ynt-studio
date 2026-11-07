// Voice cloning through ElevenLabs "Instant Voice Cloning".
// The cloned voice lives in the user's ElevenLabs account; we only keep a small
// local list (id + name) so the UI can show it next to the built-in voices.
import fs from "fs";
import path from "path";

export const CLONE_PREFIX = "clone:";
const ID_RE = /^[A-Za-z0-9]{8,64}$/;

export const isCloneVoice = voice => typeof voice === "string" && voice.startsWith(CLONE_PREFIX) && ID_RE.test(voice.slice(CLONE_PREFIX.length));
export const cloneIdOf = voice => String(voice).slice(CLONE_PREFIX.length);

export function createCloneStore(file) {
  const read = () => {
    try {
      const data = JSON.parse(fs.readFileSync(file, "utf8"));
      return Array.isArray(data) ? data.filter(item => item && ID_RE.test(String(item.id))) : [];
    } catch {
      return [];
    }
  };
  const write = list => {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, JSON.stringify(list, null, 2), "utf8");
  };
  return {
    list: () => read().map(item => ({ id: item.id, name: item.name, createdAt: item.createdAt })),
    has: id => read().some(item => item.id === id),
    add(item) {
      const list = read().filter(x => x.id !== item.id);
      list.push({ id: item.id, name: String(item.name).slice(0, 40), createdAt: new Date().toISOString() });
      write(list);
    },
    remove(id) {
      const list = read();
      const next = list.filter(item => item.id !== id);
      write(next);
      return next.length !== list.length;
    }
  };
}

async function describeError(res) {
  const text = (await res.text().catch(() => "")).slice(0, 400);
  try {
    const data = JSON.parse(text);
    const detail = data?.detail;
    const message = typeof detail === "string" ? detail : detail?.message || detail?.status;
    if (message) return `ElevenLabs HTTP ${res.status}: ${message}`;
  } catch {}
  return `ElevenLabs HTTP ${res.status} ${text.slice(0, 160)}`.trim();
}

/**
 * files: [{ buffer, filename, mimetype }]
 * Returns the new voice id.
 */
export async function elevenLabsAddVoice({ key, name, files, description = "", fetchImpl = fetch }) {
  if (!key) throw new Error("ELEVENLABS_API_KEY မထည့်ရသေးပါ");
  if (!Array.isArray(files) || !files.length) throw new Error("အသံနမူနာ ဖိုင် မပါပါ");

  const form = new FormData();
  form.append("name", name);
  if (description) form.append("description", description);
  for (const file of files) {
    form.append("files", new Blob([file.buffer], { type: file.mimetype || "audio/mpeg" }), file.filename || "sample.mp3");
  }

  const res = await fetchImpl("https://api.elevenlabs.io/v1/voices/add", {
    method: "POST",
    headers: { "xi-api-key": key },
    body: form
  });
  if (!res.ok) throw new Error(await describeError(res));
  const data = await res.json().catch(() => ({}));
  if (!data?.voice_id || !ID_RE.test(String(data.voice_id))) throw new Error("ElevenLabs က voice id မပြန်ပါ");
  return String(data.voice_id);
}

export async function elevenLabsDeleteVoice({ key, id, fetchImpl = fetch }) {
  if (!key) throw new Error("ELEVENLABS_API_KEY မထည့်ရသေးပါ");
  if (!ID_RE.test(String(id))) throw new Error("voice id မမှန်ပါ");
  const res = await fetchImpl(`https://api.elevenlabs.io/v1/voices/${encodeURIComponent(id)}`, {
    method: "DELETE",
    headers: { "xi-api-key": key }
  });
  // 404 = already gone on their side; treat as deleted.
  if (!res.ok && res.status !== 404) throw new Error(await describeError(res));
  return true;
}
