import express from "express";
import multer from "multer";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import crypto from "crypto";
import { execFile } from "child_process";
import { promisify } from "util";
import Groq from "groq-sdk";
import { GoogleGenAI } from "@google/genai";

dotenv.config();

const execFileAsync = promisify(execFile);

const app = express();

const PORT = process.env.PORT || 3000;

const ROOT = process.cwd();

const PUBLIC_DIR = path.join(ROOT, "public");
const UPLOAD_DIR = path.join(ROOT, "uploads");
const JOB_DIR = path.join(ROOT, "jobs");

/* =========================================================
   MODELS
========================================================= */

const GROQ_MODEL =
  "whisper-large-v3-turbo";

const GEMINI_MODEL =
  "gemini-3.8-flash";

const GEMINI_TTS_MODEL =
  "gemini-3.8-flash-lite-tts";

/* =========================================================
   SETTINGS
========================================================= */

const AUDIO_CHUNK_SECONDS = 90;

/*
 * Gemini video analysis frame rate.
 *
 * 2 FPS gives Gemini more visual information
 * than the default 1 FPS and helps with scene changes.
 */
const SCENE_FPS = 2;

/*
 * Maximum number of scene blocks Gemini should create.
 */
const MAX_SCENES = 18;

/*
 * Minimum useful scene duration.
 */
const MIN_SCENE_SECONDS = 3;

/*
 * TTS chunk size.
 */
const TTS_CHUNK_CHARS = 3000;

const MAX_VIDEO_SIZE =
  500 * 1024 * 1024;

/*
 * Gemini retry system.
 */
const GEMINI_MAX_RETRIES = 5;
const GEMINI_INITIAL_RETRY_DELAY = 2000;

/* =========================================================
   DIRECTORIES
========================================================= */

for (
  const dir of [
    PUBLIC_DIR,
    UPLOAD_DIR,
    JOB_DIR
  ]
) {
  fs.mkdirSync(
    dir,
    {
      recursive: true
    }
  );
}

/* =========================================================
   EXPRESS
========================================================= */

app.use(
  express.json({
    limit: "10mb"
  })
);

app.use(
  express.urlencoded({
    extended: true
  })
);

/* =========================================================
   MULTER
========================================================= */

const upload =
  multer({
    dest: UPLOAD_DIR,

    limits: {
      fileSize:
        MAX_VIDEO_SIZE
    },

    fileFilter:
      (req, file, cb) => {

        const allowed = [
          "video/mp4",
          "video/webm",
          "video/quicktime",
          "video/x-matroska",
          "video/x-msvideo"
        ];

        if (
          allowed.includes(
            file.mimetype
          )
        ) {
          return cb(
            null,
            true
          );
        }

        return cb(
          new Error(
            "Only MP4, MKV, MOV, WEBM or AVI video files are allowed."
          )
        );
      }
  });

/* =========================================================
   JOB STORAGE
========================================================= */

const jobs = new Map();

function createJob() {

  const id =
    crypto.randomUUID();

  const job = {

    id,

    status:
      "created",

    stage:
      "Waiting",

    progress:
      0,

    message:
      "Job created.",

    createdAt:
      new Date().toISOString(),

    duration:
      null,

    totalChunks:
      null,

    transcript:
      null,

    scenePlan:
      null,

    recap:
      null,

    output:
      null,

    error:
      null
  };

  jobs.set(
    id,
    job
  );

  return job;
}

function updateJob(
  id,
  data
) {

  const job =
    jobs.get(
      id
    );

  if (!job) {
    return;
  }

  Object.assign(
    job,
    data
  );
}

/* =========================================================
   ENVIRONMENT
========================================================= */

function requireEnv(
  name
) {

  const value =
    process.env[name];

  if (!value) {
    throw new Error(
      `${name} is not configured.`
    );
  }

  return value;
}

/* =========================================================
   COMMAND RUNNER
========================================================= */

async function runCommand(
  command,
  args
) {

  console.log(
    `[CMD] ${command} ${args.join(" ")}`
  );

  try {

    const result =
      await execFileAsync(
        command,
        args,
        {
          maxBuffer:
            100 *
            1024 *
            1024
        }
      );

    return result;

  } catch (error) {

    console.error(
      `[CMD ERROR] ${command}`,
      error
    );

    throw new Error(
      `${command} failed: ${
        error.stderr ||
        error.message ||
        "Unknown error"
      }`
    );
  }
}

/* =========================================================
   SLEEP
========================================================= */

function sleep(
  ms
) {

  return new Promise(
    resolve =>
      setTimeout(
        resolve,
        ms
      )
  );
}

/* =========================================================
   GEMINI ERROR / RETRY
========================================================= */

function getGeminiErrorStatus(
  error
) {

  const candidates = [

    error?.status,

    error?.code,

    error?.response?.status,

    error?.error?.status,

    error?.error?.code,

    error?.cause?.status,

    error?.cause?.code

  ];

  for (
    const value of candidates
  ) {

    const number =
      Number(
        value
      );

    if (
      Number.isFinite(
        number
      )
    ) {
      return number;
    }
  }

  const message =
    String(
      error?.message ||
      error ||
      ""
    );

  const match =
    message.match(
      /\b(429|500|502|503|504)\b/
    );

  if (match) {
    return Number(
      match[1]
    );
  }

  return null;
}

function isRetryableGeminiError(
  error
) {

  const status =
    getGeminiErrorStatus(
      error
    );

  return [
    429,
    500,
    502,
    503,
    504
  ].includes(
    status
  );
}

async function callGeminiWithRetry(
  operationName,
  operation
) {

  let lastError =
    null;

  for (
    let attempt = 0;
    attempt <=
      GEMINI_MAX_RETRIES;
    attempt++
  ) {

    try {

      if (
        attempt > 0
      ) {

        const baseDelay =
          GEMINI_INITIAL_RETRY_DELAY *
          Math.pow(
            2,
            attempt - 1
          );

        const jitter =
          Math.floor(
            Math.random() *
            1000
          );

        const wait =
          Math.min(
            baseDelay +
              jitter,
            30000
          );

        console.log(
          `[GEMINI RETRY] ${operationName} - retry ${attempt}/${GEMINI_MAX_RETRIES} after ${wait}ms`
        );

        await sleep(
          wait
        );
      }

      console.log(
        `[GEMINI] ${operationName} - attempt ${
          attempt + 1
        }/${GEMINI_MAX_RETRIES + 1}`
      );

      return await operation();

    } catch (error) {

      lastError =
        error;

      const status =
        getGeminiErrorStatus(
          error
        );

      console.error(
        `[GEMINI ERROR] ${operationName}`,
        status || "",
        error?.message ||
        error
      );

      if (
        !isRetryableGeminiError(
          error
        )
      ) {
        throw error;
      }

      if (
        attempt >=
        GEMINI_MAX_RETRIES
      ) {
        break;
      }
    }
  }

  const status =
    getGeminiErrorStatus(
      lastError
    );

  throw new Error(
    `Gemini ${operationName} failed after ${
      GEMINI_MAX_RETRIES + 1
    } attempts${
      status
        ? ` (HTTP ${status})`
        : ""
    }. ${
      lastError?.message ||
      "Temporary Gemini API error."
    }`
  );
}

/* =========================================================
   VIDEO DURATION
========================================================= */

async function getVideoDuration(
  videoPath
) {

  const result =
    await runCommand(
      "ffprobe",
      [

        "-v",
        "error",

        "-show_entries",
        "format=duration",

        "-of",
        "default=noprint_wrappers=1:nokey=1",

        videoPath

      ]
    );

  const duration =
    Number(
      result.stdout.trim()
    );

  if (
    !Number.isFinite(
      duration
    )
  ) {
    throw new Error(
      "Unable to read video duration."
    );
  }

  return duration;
}

/* =========================================================
   EXTRACT AUDIO
========================================================= */

async function extractAudio(
  videoPath,
  outputPath
) {

  await runCommand(
    "ffmpeg",
    [

      "-y",

      "-i",
      videoPath,

      "-vn",

      "-map",
      "0:a:0",

      "-ac",
      "1",

      "-ar",
      "16000",

      "-b:a",
      "64k",

      "-c:a",
      "libmp3lame",

      outputPath

    ]
  );
}

/* =========================================================
   SPLIT AUDIO
========================================================= */

async function splitAudio(
  audioPath,
  outputDir
) {

  fs.mkdirSync(
    outputDir,
    {
      recursive: true
    }
  );

  await runCommand(
    "ffmpeg",
    [

      "-y",

      "-i",
      audioPath,

      "-f",
      "segment",

      "-segment_time",
      String(
        AUDIO_CHUNK_SECONDS
      ),

      "-reset_timestamps",
      "1",

      "-c",
      "copy",

      path.join(
        outputDir,
        "chunk-%04d.mp3"
      )

    ]
  );

  const files =
    fs
      .readdirSync(
        outputDir
      )
      .filter(
        file =>
          file.startsWith(
            "chunk-"
          ) &&
          file.endsWith(
            ".mp3"
          )
      )
      .sort();

  if (
    !files.length
  ) {
    throw new Error(
      "FFmpeg could not create audio chunks."
    );
  }

  return files.map(
    file =>
      path.join(
        outputDir,
        file
      )
  );
}

/* =========================================================
   GROQ WHISPER
========================================================= */

async function transcribeChunk(
  groq,
  audioPath,
  chunkIndex
) {

  const file =
    fs.createReadStream(
      audioPath
    );

  const response =
    await groq.audio.transcriptions.create({

      file,

      model:
        GROQ_MODEL,

      response_format:
        "verbose_json",

      timestamp_granularities:
        [
          "segment"
        ],

      temperature:
        0
    });

  const offset =
    chunkIndex *
    AUDIO_CHUNK_SECONDS;

  const segments =
    Array.isArray(
      response.segments
    )
      ? response.segments
      : [];

  return {

    text:
      response.text ||
      "",

    segments:
      segments.map(
        segment => ({

          start:
            Number(
              segment.start ||
              0
            ) +
            offset,

          end:
            Number(
              segment.end ||
              0
            ) +
            offset,

          text:
            String(
              segment.text ||
              ""
            ).trim()

        })
      )
  };
}

async function transcribeMovie(
  jobId,
  audioChunks
) {

  const groq =
    new Groq({
      apiKey:
        requireEnv(
          "GROQ_API_KEY"
        )
    });

  const allSegments =
    [];

  const allTexts =
    [];

  for (
    let i = 0;
    i <
      audioChunks.length;
    i++
  ) {

    const percent =
      15 +
      Math.round(
        (
          i /
          audioChunks.length
        ) *
        35
      );

    updateJob(
      jobId,
      {

        stage:
          "Whisper",

        progress:
          percent,

        message:
          `Groq Whisper: ${
            i + 1
          } / ${
            audioChunks.length
          }`

      }
    );

    const result =
      await transcribeChunk(
        groq,
        audioChunks[i],
        i
      );

    if (
      result.text
    ) {
      allTexts.push(
        result.text
      );
    }

    allSegments.push(
      ...result.segments
    );
  }

  allSegments.sort(
    (a, b) =>
      a.start -
      b.start
  );

  return {

    text:
      allTexts
        .join(" ")
        .trim(),

    segments:
      allSegments

  };
}

/* =========================================================
   TRANSCRIPT TIMELINE FOR GEMINI
========================================================= */

function buildTranscriptTimeline(
  segments
) {

  if (
    !Array.isArray(
      segments
    )
  ) {
    return "";
  }

  return segments
    .map(
      segment =>
        `[${segment.start.toFixed(
          2
        )}s - ${segment.end.toFixed(
          2
        )}s] ${segment.text}`
    )
    .join("\n");
}

/* =========================================================
   GEMINI VIDEO FILE UPLOAD
========================================================= */

async function uploadVideoToGemini(
  ai,
  videoPath
) {

  const extension =
    path
      .extname(
        videoPath
      )
      .toLowerCase();

  let mimeType =
    "video/mp4";

  if (
    extension ===
    ".webm"
  ) {
    mimeType =
      "video/webm";
  } else if (
    extension ===
    ".mov"
  ) {
    mimeType =
      "video/mov";
  } else if (
    extension ===
    ".avi"
  ) {
    mimeType =
      "video/avi";
  } else if (
    extension ===
    ".mkv"
  ) {
    mimeType =
      "video/x-matroska";
  }

  console.log(
    `[GEMINI VIDEO] Uploading ${videoPath}`
  );

  const file =
    await ai.files.upload({
      file:
        videoPath,
      config: {
        mimeType
      }
    });

  let current =
    file;

  while (
    current?.state ===
    "PROCESSING"
  ) {

    console.log(
      "[GEMINI VIDEO] Processing..."
    );

    await sleep(
      3000
    );

    current =
      await ai.files.get({
        name:
          file.name
      });
  }

  if (
    current?.state ===
    "FAILED"
  ) {
    throw new Error(
      "Gemini failed to process the movie video."
    );
  }

  if (
    current?.state !==
    "ACTIVE"
  ) {

    throw new Error(
      `Gemini video is not ready. State: ${
        current?.state ||
        "unknown"
      }`
    );
  }

  console.log(
    "[GEMINI VIDEO] ACTIVE"
  );

  return current;
}

/* =========================================================
   SCENE PLAN SCHEMA
========================================================= */

const scenePlanSchema = {

  type:
    "object",

  properties: {

    scenes: {

      type:
        "array",

      items: {

        type:
          "object",

        properties: {

          start_sec: {
            type:
              "number"
          },

          end_sec: {
            type:
              "number"
          },

          visual_summary: {
            type:
              "string"
          },

          event_summary: {
            type:
              "string"
          },

          narration: {
            type:
              "string"
          }

        },

        required: [

          "start_sec",

          "end_sec",

          "visual_summary",

          "event_summary",

          "narration"

        ]

      }

    }

  },

  required: [
    "scenes"
  ]
};

/* =========================================================
   CLEAN JSON
========================================================= */

function parseGeminiJSON(
  text
) {

  let clean =
    String(
      text ||
      ""
    ).trim();

  clean =
    clean.replace(
      /^```json\s*/i,
      ""
    );

  clean =
    clean.replace(
      /^```\s*/i,
      ""
    );

  clean =
    clean.replace(
      /\s*```$/i,
      ""
    );

  try {

    return JSON.parse(
      clean
    );

  } catch {

    const first =
      clean.indexOf(
        "{"
      );

    const last =
      clean.lastIndexOf(
        "}"
      );

    if (
      first >= 0 &&
      last > first
    ) {

      return JSON.parse(
        clean.slice(
          first,
          last + 1
        )
      );
    }

    throw new Error(
      "Gemini returned invalid scene JSON."
    );
  }
}

/* =========================================================
   NORMALIZE SCENES
========================================================= */

function normalizeScenes(
  rawScenes,
  duration
) {

  if (
    !Array.isArray(
      rawScenes
    )
  ) {
    throw new Error(
      "Gemini did not return a scene list."
    );
  }

  let scenes =
    rawScenes
      .map(
        scene => ({

          start:
            Number(
              scene.start_sec
            ),

          end:
            Number(
              scene.end_sec
            ),

          visual:
            String(
              scene.visual_summary ||
              ""
            ).trim(),

          event:
            String(
              scene.event_summary ||
              ""
            ).trim(),

          narration:
            String(
              scene.narration ||
              ""
            ).trim()

        })
      )
      .filter(
        scene =>
          Number.isFinite(
            scene.start
          ) &&
          Number.isFinite(
            scene.end
          ) &&
          scene.end >
            scene.start
        )
      )
      .sort(
        (a, b) =>
          a.start -
          b.start
      );

  if (
    !scenes.length
  ) {
    throw new Error(
      "No usable scenes were returned by Gemini."
    );
  }

  /*
   * Clamp timestamps.
   */

  scenes =
    scenes.map(
      scene => ({

        ...scene,

        start:
          Math.max(
            0,
            Math.min(
              duration,
              scene.start
            )
          ),

        end:
          Math.max(
            0,
            Math.min(
              duration,
              scene.end
            )
          )

      })
    );

  /*
   * Remove invalid scenes again.
   */

  scenes =
    scenes.filter(
      scene =>
        scene.end >
        scene.start
    );

  /*
   * Make scenes cover the complete video.
   *
   * Any gap is assigned to the
   * previous scene.
   */

  const normalized =
    [];

  for (
    const scene of scenes
  ) {

    if (
      !normalized.length
    ) {

      normalized.push({
        ...scene,
        start: 0
      });

      continue;
    }

    const previous =
      normalized[
        normalized.length - 1
      ];

    if (
      scene.start >
      previous.end
    ) {

      previous.end =
        scene.start;
    }

    if (
      scene.start <
      previous.end
    ) {

      scene.start =
        previous.end;
    }

    if (
      scene.end >
      scene.start
    ) {

      normalized.push(
        scene
      );
    }
  }

  /*
   * Make the last scene reach
   * the exact video duration.
   */

  if (
    normalized.length
  ) {

    normalized[
      normalized.length - 1
    ].end =
      duration;
  }

  /*
   * Merge very short scenes.
   */

  let merged =
    [];

  for (
    const scene of normalized
  ) {

    const sceneDuration =
      scene.end -
      scene.start;

    if (
      sceneDuration <
      MIN_SCENE_SECONDS &&
      merged.length
    ) {

      const previous =
        merged[
          merged.length - 1
        ];

      previous.end =
        scene.end;

      previous.visual =
        `${previous.visual} ${scene.visual}`.trim();

      previous.event =
        `${previous.event} ${scene.event}`.trim();

      previous.narration =
        `${previous.narration} ${scene.narration}`.trim();

    } else {

      merged.push({
        ...scene
      });
    }
  }

  /*
   * If first scene is too short,
   * merge it into second scene.
   */

  if (
    merged.length >= 2 &&
    (
      merged[0].end -
      merged[0].start
    ) <
      MIN_SCENE_SECONDS
  ) {

    const first =
      merged.shift();

    const second =
      merged[0];

    second.start =
      first.start;

    second.visual =
      `${first.visual} ${second.visual}`.trim();

    second.event =
      `${first.event} ${second.event}`.trim();

    second.narration =
      `${first.narration} ${second.narration}`.trim();
  }

  /*
   * Hard limit.
   *
   * If Gemini somehow returns too many
   * scenes, merge the smallest ones.
   */

  while (
    merged.length >
    MAX_SCENES
  ) {

    let smallestIndex =
      0;

    let smallestDuration =
      Infinity;

    for (
      let i = 0;
      i <
        merged.length;
      i++
    ) {

      const d =
        merged[i].end -
        merged[i].start;

      if (
        d <
        smallestDuration
      ) {

        smallestDuration =
          d;

        smallestIndex =
          i;
      }
    }

    if (
      smallestIndex <
      merged.length - 1
    ) {

      const a =
        merged[
          smallestIndex
        ];

      const b =
        merged[
          smallestIndex + 1
        ];

      b.start =
        a.start;

      b.visual =
        `${a.visual} ${b.visual}`.trim();

      b.event =
        `${a.event} ${b.event}`.trim();

      b.narration =
        `${a.narration} ${b.narration}`.trim();

      merged.splice(
        smallestIndex,
        1
      );

    } else {

      const a =
        merged[
          smallestIndex - 1
        ];

      const b =
        merged[
          smallestIndex
        ];

      a.end =
        b.end;

      a.visual =
        `${a.visual} ${b.visual}`.trim();

      a.event =
        `${a.event} ${b.event}`.trim();

      a.narration =
        `${a.narration} ${b.narration}`.trim();

      merged.pop();
    }
  }

  /*
   * Final exact correction.
   */

  if (
    merged.length
  ) {

    merged[0].start =
      0;

    merged[
      merged.length - 1
    ].end =
      duration;

  }

  return merged.map(
    (scene, index) => ({

      index:
        index + 1,

      start:
        Number(
          scene.start.toFixed(
            3
          )
        ),

      end:
        Number(
          scene.end.toFixed(
            3
          )
        ),

      duration:
        Number(
          (
            scene.end -
            scene.start
          ).toFixed(
            3
          )
        ),

      visual:
        scene.visual,

      event:
        scene.event,

      narration:
        scene.narration

    })
  );
}

/* =========================================================
   GEMINI SCENE-SYNC RECAP
========================================================= */

async function generateScenePlan(
  jobId,
  moviePath,
  duration,
  transcript,
  language = "my",
  style = "cinematic"
) {

  const ai =
    new GoogleGenAI({
      apiKey:
        requireEnv(
          "GEMINI_API_KEY"
        )
    });

  updateJob(
    jobId,
    {

      stage:
        "Gemini",

      progress:
        55,

      message:
        "Gemini is watching the movie and matching scenes..."

    }
  );

  /*
   * Upload the actual movie to Gemini.
   */

  const videoFile =
    await uploadVideoToGemini(
      ai,
      moviePath
    );

  const timeline =
    buildTranscriptTimeline(
      transcript.segments
    );

  let languageInstruction =
    "Write the narration in natural spoken Myanmar (Burmese).";

  if (
    language ===
    "en"
  ) {

    languageInstruction =
      "Write the narration in natural spoken English.";
  }

  let styleInstruction =
    "Use a cinematic movie recap style.";

  if (
    style ===
    "short"
  ) {

    styleInstruction =
      "Use concise, fast-paced movie recap narration.";

  } else if (
    style ===
    "storytelling"
  ) {

    styleInstruction =
      "Use smooth storytelling with suspense and emotional flow.";

  } else if (
    style ===
    "detailed"
  ) {

    styleInstruction =
      "Use detailed but natural movie recap narration.";
  }

  const prompt =
`
You are creating a PROFESSIONAL TIMESTAMP-SYNCHRONIZED MOVIE RECAP.

You have the actual movie video.

You MUST analyze what is visibly happening in the video.

You also have the original Whisper transcript with timestamps.

Your job is to create a chronological scene plan.

IMPORTANT:
The final video will place each narration directly over the
same video time range that you specify.

Therefore visual synchronization is extremely important.

RULES:

1. Watch and analyze the actual video.
2. Detect meaningful scene changes.
3. Every scene MUST have a start_sec and end_sec.
4. Use real timestamps from the video.
5. Keep scenes chronological.
6. Do not overlap scenes.
7. Do not leave gaps between scenes.
8. Cover the complete movie from 0 seconds to ${duration.toFixed(2)} seconds.
9. Prefer approximately 6-14 seconds per scene.
10. Do not create hundreds of scenes.
11. Create approximately 8-${MAX_SCENES} meaningful scenes.
12. A scene should normally begin when the visual situation changes.
13. Pay attention to characters, locations, actions and important objects.
14. Match narration to what is actually visible during that timestamp.
15. Use the transcript only as supporting evidence.
16. Do not invent events.
17. Do not invent characters.
18. Do not invent dialogue.
19. Do not move an event to a different timestamp.
20. If the transcript says something but the visual scene is different,
    do NOT describe the wrong visual event.
21. Keep the major story events and ending.
22. Avoid meaningless filler.
23. Narration should sound natural when spoken aloud.
24. Each narration should fit naturally inside its scene duration.
25. Do not write extremely long narration for a short scene.
26. Do not write extremely short narration for a long scene.
27. Do not use Markdown.
28. Do not use headings inside narration.
29. Do not mention AI.
30. Do not mention this instruction.

LANGUAGE:
${languageInstruction}

STYLE:
${styleInstruction}

SCENE JSON FORMAT:

{
  "scenes": [
    {
      "start_sec": 0,
      "end_sec": 8,
      "visual_summary": "What is visibly happening.",
      "event_summary": "What important story event happens.",
      "narration": "Natural recap narration for this exact scene."
    }
  ]
}

The JSON must contain ONLY the scene plan.

ORIGINAL WHISPER TIMELINE:

${timeline}
`;

  const response =
    await callGeminiWithRetry(
      "Scene-Synchronized Movie Analysis",
      () =>
        ai.interactions.create({

          model:
            GEMINI_MODEL,

          input: [

            {

              type:
                "video",

              uri:
                videoFile.uri,

              mime_type:
                videoFile.mimeType,

              processing: {

                type:
                  "static",

                fps:
                  SCENE_FPS

              }

            },

            {

              type:
                "text",

              text:
                prompt

            }

          ],

          response_format: {

            type:
              "text",

            mime_type:
              "application/json",

            schema:
              scenePlanSchema

          }

        })
    );

  const outputText =
    response?.output_text ||
    response?.outputText ||
    "";

  if (
    !outputText
  ) {

    throw new Error(
      "Gemini returned an empty scene plan."
    );
  }

  const parsed =
    parseGeminiJSON(
      outputText
    );

  const scenes =
    normalizeScenes(
      parsed.scenes,
      duration
    );

  if (
    !scenes.length
  ) {

    throw new Error(
      "Gemini returned no usable scenes."
    );
  }

  console.log(
    `[SCENE SYNC] ${scenes.length} scenes created`
  );

  return {
    scenes
  };
}

/* =========================================================
   TTS VOICE
========================================================= */

function resolveVoice(
  voice
) {

  if (
    voice ===
    "male"
  ) {

    return "Puck";
  }

  if (
    voice ===
    "female"
  ) {

    return "Kore";
  }

  const allowedVoices = [

    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat"

  ];

  if (
    allowedVoices.includes(
      voice
    )
  ) {

    return voice;
  }

  return "Kore";
}

/* =========================================================
   TTS
========================================================= */

async function generateSceneTTS(
  ai,
  scene,
  outputPath,
  voice
) {

  const actualVoice =
    resolveVoice(
      voice
    );

  const text =
    String(
      scene.narration ||
      ""
    ).trim();

  if (
    !text
  ) {

    throw new Error(
      `Scene ${scene.index} has no narration.`
    );
  }

  const response =
    await callGeminiWithRetry(
      `Scene ${scene.index} TTS`,
      () =>
        ai.models.generateContent({

          model:
            GEMINI_TTS_MODEL,

          contents: [

            {

              role:
                "user",

              parts: [

                {

                  text,

                  speech_metadata: {

                    style:
                      "Natural cinematic Myanmar movie recap narration. Clear pronunciation, smooth pacing, emotionally controlled, confident storyteller voice."

                  }

                }

              ]

            }

          ],

          config: {

            responseModalities:
              [
                "AUDIO"
              ],

            speechConfig: {

              voiceConfig: {

                voice:
                  actualVoice

              }

            }

          }

        })
    );

  const base64 =
    response
      ?.candidates?.[0]
      ?.content?.parts?.find(
        part =>
          part?.inlineData?.data
      )
      ?.inlineData?.data;

  if (
    !base64
  ) {

    throw new Error(
      `Gemini returned no audio for scene ${scene.index}.`
    );
  }

  fs.writeFileSync(
    outputPath,
    Buffer.from(
      base64,
      "base64"
    )
  );

  const stats =
    fs.statSync(
      outputPath
    );

  if (
    stats.size <
    100
  ) {

    throw new Error(
      `TTS scene ${scene.index} audio is invalid.`
    );
  }

  return outputPath;
}

/* =========================================================
   AUDIO DURATION
========================================================= */

async function getAudioDuration(
  audioPath
) {

  const result =
    await runCommand(
      "ffprobe",
      [

        "-v",
        "error",

        "-show_entries",
        "format=duration",

        "-of",
        "default=noprint_wrappers=1:nokey=1",

        audioPath

      ]
    );

  const duration =
    Number(
      result.stdout.trim()
    );

  if (
    !Number.isFinite(
      duration
    )
  ) {

    throw new Error(
      "Unable to read generated TTS duration."
    );
  }

  return duration;
}

/* =========================================================
   ATEMPO FILTER
========================================================= */

function buildAtempoFilters(
  factor
) {

  let value =
    factor;

  const filters =
    [];

  while (
    value >
    2
  ) {

    filters.push(
      "atempo=2.0"
    );

    value /=
      2;
  }

  while (
    value <
    0.5
  ) {

    filters.push(
      "atempo=0.5"
    );

    value /=
      0.5;
  }

  if (
    Math.abs(
      value - 1
    ) >
    0.001
  ) {

    filters.push(
      `atempo=${value.toFixed(
        6
      )}`
    );
  }

  return filters;
}

/* =========================================================
   FIT AUDIO TO SCENE
========================================================= */

async function fitAudioToScene(
  inputAudio,
  outputAudio,
  targetDuration
) {

  const sourceDuration =
    await getAudioDuration(
      inputAudio
    );

  if (
    sourceDuration <=
    0
  ) {

    throw new Error(
      "Generated TTS has invalid duration."
    );
  }

  /*
   * Audio duration / target duration
   *
   * Example:
   *
   * 8 sec audio
   * 4 sec scene
   *
   * factor = 2
   *
   * atempo=2
   */

  const factor =
    sourceDuration /
    targetDuration;

  const filters =
    buildAtempoFilters(
      factor
    );

  /*
   * Always pad if narration is
   * shorter, then trim exactly
   * to scene duration.
   */

  filters.push(
    "apad"
  );

  filters.push(
    `atrim=duration=${targetDuration.toFixed(
      3
    )}`
  );

  await runCommand(
    "ffmpeg",
    [

      "-y",

      "-i",
      inputAudio,

      "-af",
      filters.join(","),

      "-ar",
      "24000",

      "-ac",
      "1",

      "-c:a",
      "pcm_s16le",

      outputAudio

    ]
  );

  return {
    sourceDuration,
    targetDuration,
    factor
  };
}

/* =========================================================
   CUT VIDEO SCENE
========================================================= */

async function renderSceneVideo(
  moviePath,
  scene,
  sceneVideoPath
) {

  const duration =
    scene.end -
    scene.start;

  await runCommand(
    "ffmpeg",
    [

      "-y",

      "-ss",
      scene.start.toFixed(
        3
      ),

      "-i",
      moviePath,

      "-t",
      duration.toFixed(
        3
      ),

      "-an",

      "-c:v",
      "libx264",

      "-preset",
      "veryfast",

      "-crf",
      "23",

      "-pix_fmt",
      "yuv420p",

      "-movflags",
      "+faststart",

      sceneVideoPath

    ]
  );

  return sceneVideoPath;
}

/* =========================================================
   MUX SCENE VIDEO + SCENE AUDIO
========================================================= */

async function muxScene(
  sceneVideo,
  sceneAudio,
  outputPath
) {

  await runCommand(
    "ffmpeg",
    [

      "-y",

      "-i",
      sceneVideo,

      "-i",
      sceneAudio,

      "-map",
      "0:v:0",

      "-map",
      "1:a:0",

      "-c:v",
      "copy",

      "-c:a",
      "aac",

      "-b:a",
      "128k",

      "-shortest",

      "-movflags",
      "+faststart",

      outputPath

    ]
  );

  return outputPath;
}

/* =========================================================
   CONCAT FINAL SCENES
========================================================= */

async function concatScenes(
  sceneFiles,
  outputPath,
  listPath
) {

  if (
    !sceneFiles.length
  ) {

    throw new Error(
      "No rendered scenes were found."
    );
  }

  const list =
    sceneFiles
      .map(
        file =>
          `file '${file.replace(
            /'/g,
            "'\\''"
          )}'`
      )
      .join(
        "\n"
      );

  fs.writeFileSync(
    listPath,
    list,
    "utf8"
  );

  await runCommand(
    "ffmpeg",
    [

      "-y",

      "-f",
      "concat",

      "-safe",
      "0",

      "-i",
      listPath,

      "-c",
      "copy",

      "-movflags",
      "+faststart",

      outputPath

    ]
  );

  return outputPath;
}

/* =========================================================
   RENDER SCENE-SYNC VIDEO
========================================================= */

async function renderSceneSyncVideo(
  jobId,
  moviePath,
  scenePlan,
  voice,
  jobFolder
) {

  const ai =
    new GoogleGenAI({
      apiKey:
        requireEnv(
          "GEMINI_API_KEY"
        )
    });

  const sceneDir =
    path.join(
      jobFolder,
      "scenes"
    );

  fs.mkdirSync(
    sceneDir,
    {
      recursive: true
    }
  );

  const renderedScenes =
    [];

  for (
    let i = 0;
    i <
      scenePlan.scenes.length;
    i++
  ) {

    const scene =
      scenePlan.scenes[i];

    const percent =
      65 +
      Math.round(
        (
          i /
          scenePlan.scenes.length
        ) *
        25
      );

    updateJob(
      jobId,
      {

        stage:
          "Voice",

        progress:
          percent,

        message:
          `Scene ${
            scene.index
          } / ${
            scenePlan.scenes.length
          } — generating voice...`

      }
    );

    const rawTTS =
      path.join(
        sceneDir,
        `scene-${String(
          scene.index
        ).padStart(
          3,
          "0"
        )}-raw.wav`
      );

    const fittedTTS =
      path.join(
        sceneDir,
        `scene-${String(
          scene.index
        ).padStart(
          3,
          "0"
        )}-audio.wav`
      );

    const sceneVideo =
      path.join(
        sceneDir,
        `scene-${String(
          scene.index
        ).padStart(
          3,
          "0"
        )}-video.mp4`
      );

    const sceneFinal =
      path.join(
        sceneDir,
        `scene-${String(
          scene.index
        ).padStart(
          3,
          "0"
        )}-final.mp4`
      );

    await generateSceneTTS(
      ai,
      scene,
      rawTTS,
      voice
    );

    await fitAudioToScene(
      rawTTS,
      fittedTTS,
      scene.duration
    );

    updateJob(
      jobId,
      {

        stage:
          "FFmpeg",

        progress:
          percent + 2,

        message:
          `Scene ${
            scene.index
          } / ${
            scenePlan.scenes.length
          } — syncing video and voice...`

      }
    );

    await renderSceneVideo(
      moviePath,
      scene,
      sceneVideo
    );

    await muxScene(
      sceneVideo,
      fittedTTS,
      sceneFinal
    );

    renderedScenes.push(
      sceneFinal
    );
  }

  updateJob(
    jobId,
    {

      stage:
        "FFmpeg",

      progress:
        94,

      message:
        "Joining synchronized scenes..."

    }
  );

  const listPath =
    path.join(
      jobFolder,
      "scene-list.txt"
    );

  const outputPath =
    path.join(
      jobFolder,
      "YNT-One-Clips-Recap.mp4"
    );

  await concatScenes(
    renderedScenes,
    outputPath,
    listPath
  );

  return outputPath;
}

/* =========================================================
   PROCESS ONE CLIP
========================================================= */

async function processOneClip(
  jobId,
  moviePath,
  language,
  style,
  voice
) {

  try {

    const jobFolder =
      path.dirname(
        moviePath
      );

    /* -----------------------------------------
       VIDEO INFO
    ----------------------------------------- */

    updateJob(
      jobId,
      {

        stage:
          "Upload",

        progress:
          8,

        message:
          "Reading movie information..."

      }
    );

    const duration =
      await getVideoDuration(
        moviePath
      );

    updateJob(
      jobId,
      {
        duration
      }
    );

    console.log(
      `[JOB ${jobId}] Video duration: ${duration.toFixed(
        2
      )} seconds`
    );

    /* -----------------------------------------
       EXTRACT AUDIO
    ----------------------------------------- */

    const audioPath =
      path.join(
        jobFolder,
        "movie-audio.mp3"
      );

    updateJob(
      jobId,
      {

        stage:
          "FFmpeg",

        progress:
          10,

        message:
          "Extracting movie audio..."

      }
    );

    await extractAudio(
      moviePath,
      audioPath
    );

    /* -----------------------------------------
       SPLIT AUDIO
    ----------------------------------------- */

    const chunksDir =
      path.join(
        jobFolder,
        "audio-chunks"
      );

    updateJob(
      jobId,
      {

        stage:
          "FFmpeg",

        progress:
          13,

        message:
          "Preparing audio for Whisper..."

      }
    );

    const audioChunks =
      await splitAudio(
        audioPath,
        chunksDir
      );

    updateJob(
      jobId,
      {

        totalChunks:
          audioChunks.length

      }
    );

    /* -----------------------------------------
       WHISPER
    ----------------------------------------- */

    const transcript =
      await transcribeMovie(
        jobId,
        audioChunks
      );

    if (
      !transcript.text
    ) {

      throw new Error(
        "Whisper returned an empty transcript."
      );
    }

    const transcriptPath =
      path.join(
        jobFolder,
        "transcript.json"
      );

    fs.writeFileSync(
      transcriptPath,
      JSON.stringify(
        transcript,
        null,
        2
      ),
      "utf8"
    );

    updateJob(
      jobId,
      {

        stage:
          "Transcript",

        progress:
          52,

        message:
          "Timestamp transcript completed.",

        transcript: {

          characters:
            transcript.text.length,

          segments:
            transcript.segments.length

        }

      }
    );

    /* -----------------------------------------
       GEMINI SCENE-SYNC ANALYSIS
    ----------------------------------------- */

    const scenePlan =
      await generateScenePlan(
        jobId,
        moviePath,
        duration,
        transcript,
        language,
        style
      );

    const scenePlanPath =
      path.join(
        jobFolder,
        "scene-plan.json"
      );

    fs.writeFileSync(
      scenePlanPath,
      JSON.stringify(
        scenePlan,
        null,
        2
      ),
      "utf8"
    );

    updateJob(
      jobId,
      {

        stage:
          "Gemini",

        progress:
          62,

        message:
          `${scenePlan.scenes.length} synchronized scenes created.`,

        scenePlan: {

          scenes:
            scenePlan.scenes.length,

          path:
            scenePlanPath

        }

      }
    );

    /* -----------------------------------------
       SCENE-SYNC VOICE + VIDEO
    ----------------------------------------- */

    const outputPath =
      await renderSceneSyncVideo(
        jobId,
        moviePath,
        scenePlan,
        voice,
        jobFolder
      );

    /* -----------------------------------------
       VERIFY OUTPUT
    ----------------------------------------- */

    if (
      !fs.existsSync(
        outputPath
      )
    ) {

      throw new Error(
        "FFmpeg did not create the final MP4."
      );
    }

    const stats =
      fs.statSync(
        outputPath
      );

    if (
      stats.size <=
      0
    ) {

      throw new Error(
        "Final MP4 file is empty."
      );
    }

    const filename =
      `YNT-One-Clips-${jobId}.mp4`;

    /* -----------------------------------------
       COMPLETED
    ----------------------------------------- */

    updateJob(
      jobId,
      {

        status:
          "completed",

        stage:
          "Ready",

        progress:
          100,

        message:
          "Your synchronized recap video is ready.",

        output: {

          path:
            outputPath,

          filename,

          size:
            stats.size,

          url:
            `/api/download/${jobId}`

        }

      }
    );

    console.log(
      `[JOB ${jobId}] COMPLETED`
    );

  } catch (error) {

    console.error(
      `[JOB ${jobId}] ERROR:`,
      error
    );

    updateJob(
      jobId,
      {

        status:
          "error",

        stage:
          "Error",

        progress:
          0,

        message:
          error.message ||
          "Movie processing failed.",

        error:
          error.message ||
          "Movie processing failed."

      }
    );
  }
}

/* =========================================================
   HEALTH
========================================================= */

app.get(
  "/health",
  (req, res) => {

    res.json({

      ok:
        true,

      name:
        "YNT One Clips",

      status:
        "online",

      models: {

        whisper:
          GROQ_MODEL,

        recap:
          GEMINI_MODEL,

        tts:
          GEMINI_TTS_MODEL

      },

      sceneSync:
        true,

      sceneFPS:
        SCENE_FPS

    });
  }
);

/* =========================================================
   JOB STATUS
========================================================= */

app.get(
  "/api/status/:id",
  (req, res) => {

    const job =
      jobs.get(
        req.params.id
      );

    if (!job) {

      return res
        .status(404)
        .json({

          error:
            "Job not found."

        });
    }

    return res.json(
      job
    );
  }
);

/* =========================================================
   DOWNLOAD
========================================================= */

app.get(
  "/api/download/:id",
  (req, res) => {

    const job =
      jobs.get(
        req.params.id
      );

    if (!job) {

      return res
        .status(404)
        .json({

          error:
            "Job not found."

        });
    }

    if (
      job.status !==
        "completed" ||
      !job.output
    ) {

      return res
        .status(404)
        .json({

          error:
            "Final video is not ready yet."

        });
    }

    if (
      !fs.existsSync(
        job.output.path
      )
    ) {

      return res
        .status(404)
        .json({

          error:
            "Output video file no longer exists."

        });
    }

    return res.download(
      job.output.path,
      job.output.filename
    );
  }
);

/* =========================================================
   ONE CLIP
========================================================= */

app.post(
  "/api/one-clip",

  upload.single(
    "movie"
  ),

  async (
    req,
    res
  ) => {

    try {

      if (
        !req.file
      ) {

        return res
          .status(400)
          .json({

            error:
              "Movie file is required."

          });
      }

      /* -----------------------------------------
         CREATE JOB
      ----------------------------------------- */

      const job =
        createJob();

      const jobFolder =
        path.join(
          JOB_DIR,
          job.id
        );

      fs.mkdirSync(
        jobFolder,
        {
          recursive:
            true
        }
      );

      /* -----------------------------------------
         MOVE MOVIE
      ----------------------------------------- */

      const originalName =
        req.file.originalname ||
        "movie.mp4";

      const extension =
        path.extname(
          originalName
        ) ||
        ".mp4";

      const moviePath =
        path.join(
          jobFolder,
          `movie${extension}`
        );

      fs.renameSync(
        req.file.path,
        moviePath
      );

      /* -----------------------------------------
         OPTIONS
      ----------------------------------------- */

      const language =
        req.body?.language ||
        "my";

      const style =
        req.body?.style ||
        "cinematic";

      const voice =
        req.body?.voice ||
        "female";

      /* -----------------------------------------
         INITIAL JOB
      ----------------------------------------- */

      updateJob(
        job.id,
        {

          status:
            "processing",

          stage:
            "Upload",

          progress:
            5,

          message:
            "Movie uploaded. Processing started."

        }
      );

      console.log(
        "========================================"
      );

      console.log(
        `[JOB ${job.id}] STARTED`
      );

      console.log(
        `[JOB ${job.id}] Language: ${language}`
      );

      console.log(
        `[JOB ${job.id}] Style: ${style}`
      );

      console.log(
        `[JOB ${job.id}] Voice: ${voice}`
      );

      console.log(
        `[JOB ${job.id}] Scene Sync: ENABLED`
      );

      console.log(
        "========================================"
      );

      /* -----------------------------------------
         RETURN IMMEDIATELY
      ----------------------------------------- */

      res
        .status(202)
        .json({

          success:
            true,

          jobId:
            job.id,

          status:
            "processing",

          message:
            "Movie processing started.",

          statusUrl:
            `/api/status/${job.id}`,

          download:
            `/api/download/${job.id}`

        });

      /* -----------------------------------------
         BACKGROUND PROCESSING
      ----------------------------------------- */

      setImmediate(
        () => {

          processOneClip(
            job.id,
            moviePath,
            language,
            style,
            voice
          ).catch(
            error => {

              console.error(
                `[JOB ${job.id}] UNHANDLED ERROR:`,
                error
              );

              updateJob(
                job.id,
                {

                  status:
                    "error",

                  stage:
                    "Error",

                  progress:
                    0,

                  message:
                    error.message ||
                    "Movie processing failed.",

                  error:
                    error.message ||
                    "Movie processing failed."

                }
              );
            }
          );

        }
      );

    } catch (error) {

      console.error(
        "ONE CLIP START ERROR:",
        error
      );

      if (
        req.file?.path &&
        fs.existsSync(
          req.file.path
        )
      ) {

        try {

          fs.unlinkSync(
            req.file.path
          );

        } catch {}
      }

      return res
        .status(500)
        .json({

          error:
            error.message ||
            "Unable to start movie processing."

        });
    }
  }
);

/* =========================================================
   FRONTEND
========================================================= */

app.use(
  express.static(
    PUBLIC_DIR
  )
);

/* =========================================================
   ERROR HANDLER
========================================================= */

app.use(
  (
    err,
    req,
    res,
    next
  ) => {

    console.error(
      "SERVER ERROR:",
      err
    );

    if (
      err instanceof
      multer.MulterError
    ) {

      return res
        .status(400)
        .json({

          error:
            err.message

        });
    }

    return res
      .status(500)
      .json({

        error:
          err.message ||
          "Server error."

      });
  }
);

/* =========================================================
   START SERVER
========================================================= */

app.listen(
  PORT,
  () => {

    console.log(
      "========================================"
    );

    console.log(
      "YNT One Clips"
    );

    console.log(
      `Server running on port ${PORT}`
    );

    console.log(
      `Groq Whisper: ${GROQ_MODEL}`
    );

    console.log(
      `Gemini Recap: ${GEMINI_MODEL}`
    );

    console.log(
      `Gemini TTS: ${GEMINI_TTS_MODEL}`
    );

    console.log(
      `Scene Analysis FPS: ${SCENE_FPS}`
    );

    console.log(
      "Scene Synchronization: READY"
    );

    console.log(
      "Gemini Retry System: READY"
    );

    console.log(
      "Background Jobs: READY"
    );

    console.log(
      "FFmpeg Scene Renderer: READY"
    );

    console.log(
      "========================================"
    );
  }
);
