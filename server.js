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

const GROQ_MODEL = "whisper-large-v3-turbo";
const GEMINI_MODEL = "gemini-3.8-flash";
const GEMINI_TTS_MODEL = "gemini-3.8-flash-tts";

const AUDIO_CHUNK_SECONDS = 90;
const TTS_CHUNK_CHARS = 7000;

for (const dir of [
  PUBLIC_DIR,
  UPLOAD_DIR,
  JOB_DIR
]) {
  fs.mkdirSync(dir, {
    recursive: true
  });
}

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

const upload = multer({
  dest: UPLOAD_DIR,

  limits: {
    fileSize: 500 * 1024 * 1024
  },

  fileFilter: (req, file, cb) => {
    const allowed = [
      "video/mp4",
      "video/webm",
      "video/quicktime",
      "video/x-matroska",
      "video/x-msvideo"
    ];

    if (allowed.includes(file.mimetype)) {
      cb(null, true);
      return;
    }

    cb(
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
  const id = crypto.randomUUID();

  const job = {
    id,

    status: "created",

    stage: "Waiting",

    progress: 0,

    message: "Job created.",

    createdAt: new Date().toISOString(),

    duration: null,

    transcript: null,

    recap: null,

    output: null,

    error: null
  };

  jobs.set(id, job);

  return job;
}


function updateJob(id, data) {
  const job = jobs.get(id);

  if (!job) {
    return;
  }

  Object.assign(job, data);
}


/* =========================================================
   ENVIRONMENT
========================================================= */

function requireEnv(name) {
  const value = process.env[name];

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

async function runCommand(command, args) {
  console.log(
    `[CMD] ${command} ${args.join(" ")}`
  );

  const result = await execFileAsync(
    command,
    args,
    {
      maxBuffer: 50 * 1024 * 1024
    }
  );

  return result;
}


/* =========================================================
   VIDEO DURATION
========================================================= */

async function getVideoDuration(videoPath) {
  const result = await runCommand(
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
    Number(result.stdout.trim());

  if (!Number.isFinite(duration)) {
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
      String(AUDIO_CHUNK_SECONDS),

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

  const files = fs
    .readdirSync(outputDir)
    .filter(
      file =>
        file.startsWith("chunk-") &&
        file.endsWith(".mp3")
    )
    .sort();

  if (!files.length) {
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
  const file = fs.createReadStream(
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
        ["segment"],

      temperature:
        0
    });

  const offset =
    chunkIndex *
    AUDIO_CHUNK_SECONDS;

  const segments =
    Array.isArray(response.segments)
      ? response.segments
      : [];

  return {
    text:
      response.text || "",

    segments:
      segments.map(segment => ({
        start:
          Number(segment.start || 0) +
          offset,

        end:
          Number(segment.end || 0) +
          offset,

        text:
          String(
            segment.text || ""
          ).trim()
      }))
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

  const allSegments = [];
  const allTexts = [];

  for (
    let i = 0;
    i < audioChunks.length;
    i++
  ) {
    const percent =
      15 +
      Math.round(
        (i / audioChunks.length) *
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
          `Groq Whisper: ${i + 1} / ${audioChunks.length}`
      }
    );

    const result =
      await transcribeChunk(
        groq,
        audioChunks[i],
        i
      );

    if (result.text) {
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
      a.start - b.start
  );

  return {
    text:
      allTexts.join(" ").trim(),

    segments:
      allSegments
  };
}


/* =========================================================
   GEMINI RECAP
========================================================= */

async function generateRecap(
  jobId,
  transcriptText,
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
        "Gemini is creating the movie recap..."
    }
  );

  /*
   * Gemini 3.8 Flash supports a large context window,
   * but we still protect the server from unnecessarily
   * huge requests.
   */

  const maxCharacters =
    300000;

  const source =
    transcriptText.length >
    maxCharacters
      ? transcriptText.slice(
          0,
          maxCharacters
        )
      : transcriptText;

  const languageInstruction =
    language === "en"
      ? "Write the final narration in natural English."
      : "Write the final narration in natural spoken Myanmar (Burmese).";

  const styleInstruction =
    style === "short"
      ? "Keep it concise and fast-paced."
      : style === "storytelling"
        ? "Use a smooth storytelling style with natural suspense."
        : "Use a cinematic movie-recap narration style.";

  const prompt = `
You are a professional movie recap writer.

Create a narration script from the movie transcript below.

LANGUAGE:
${languageInstruction}

STYLE:
${styleInstruction}

STRICT RULES:

- Follow the chronological order of the story.
- Explain important characters when necessary.
- Focus on important plot events.
- Keep important twists and the ending.
- Do not invent characters, scenes, events or dialogue.
- Do not claim information that is not supported by the transcript.
- Remove repeated or meaningless transcript fragments.
- Do not mention AI.
- Do not mention this prompt.
- Do not use Markdown.
- Do not use headings.
- Do not use bullet points.
- Write only the narration script.
- Make the narration natural for voice-over.
- Avoid extremely short fragments.
- Make the story easy to follow for someone who has never seen the movie.

MOVIE TRANSCRIPT:

${source}
`;

  const response =
    await ai.models.generateContent({
      model:
        GEMINI_MODEL,

      contents:
        prompt,

      config: {
        thinkingConfig: {
          thinkingLevel:
            "low"
        },

        maxOutputTokens:
          16000
      }
    });

  const text =
    response.text?.trim();

  if (!text) {
    throw new Error(
      "Gemini returned an empty recap script."
    );
  }

  return text;
}


/* =========================================================
   SPLIT TTS TEXT
========================================================= */

function splitTextForTTS(text) {
  const clean =
    String(text || "")
      .replace(/\r/g, "")
      .trim();

  if (!clean) {
    return [];
  }

  const sentences =
    clean.match(
      /[^.!?။！？]+[.!?။！？]*/g
    ) || [clean];

  const chunks = [];
  let current = "";

  for (const sentence of sentences) {
    const part =
      sentence.trim();

    if (!part) {
      continue;
    }

    if (
      current.length +
      part.length +
      1 <=
      TTS_CHUNK_CHARS
    ) {
      current =
        current
          ? `${current} ${part}`
          : part;

      continue;
    }

    if (current) {
      chunks.push(
        current.trim()
      );
    }

    /*
     * If one sentence itself is too long,
     * split it safely by character length.
     */

    if (
      part.length >
      TTS_CHUNK_CHARS
    ) {
      for (
        let i = 0;
        i < part.length;
        i += TTS_CHUNK_CHARS
      ) {
        chunks.push(
          part.slice(
            i,
            i + TTS_CHUNK_CHARS
          )
        );
      }

      current = "";
    } else {
      current = part;
    }
  }

  if (current) {
    chunks.push(
      current.trim()
    );
  }

  return chunks;
}


/* =========================================================
   GEMINI TTS
========================================================= */

async function generateTTS(
  jobId,
  recapText,
  outputDir
) {
  const apiKey =
    requireEnv(
      "GEMINI_API_KEY"
    );

  const ai =
    new GoogleGenAI({
      apiKey
    });

  const chunks =
    splitTextForTTS(
      recapText
    );

  if (!chunks.length) {
    throw new Error(
      "There is no recap text for TTS."
    );
  }

  fs.mkdirSync(
    outputDir,
    {
      recursive: true
    }
  );

  const audioFiles = [];

  for (
    let i = 0;
    i < chunks.length;
    i++
  ) {
    const percent =
      65 +
      Math.round(
        (i / chunks.length) *
        20
      );

    updateJob(
      jobId,
      {
        stage:
          "Voice",

        progress:
          percent,

        message:
          `Gemini Voice: ${i + 1} / ${chunks.length}`
      }
    );

    const interaction =
      await ai.interactions.create({
        model:
          GEMINI_TTS_MODEL,

        input: [
          {
            type:
              "user_input",

            content: [
              {
                type:
                  "text",

                text:
                  chunks[i],

                annotations: [
                  {
                    type:
                      "speech_metadata",

                    style:
                      "natural, cinematic, calm, clear Myanmar movie narration"
                  }
                ]
              }
            ]
          }
        ],

        response_format: {
          type:
            "audio"
        },

        generation_config: {
          speech_config: [
            {
              voice:
                "Kore"
            }
          ]
        }
      });

    const base64 =
      interaction?.output_audio?.data;

    if (!base64) {
      throw new Error(
        `Gemini TTS returned no audio for chunk ${i + 1}.`
      );
    }

    const audioPath =
      path.join(
        outputDir,
        `tts-${String(i).padStart(4, "0")}.wav`
      );

    fs.writeFileSync(
      audioPath,
      Buffer.from(
        base64,
        "base64"
      )
    );

    audioFiles.push(
      audioPath
    );
  }

  return audioFiles;
}


/* =========================================================
   CONCAT TTS AUDIO
========================================================= */

async function concatAudio(
  audioFiles,
  outputPath,
  listPath
) {
  const listContent =
    audioFiles
      .map(
        file =>
          `file '${file.replace(/'/g, "'\\''")}'`
      )
      .join("\n");

  fs.writeFileSync(
    listPath,
    listContent,
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

      "-c:a",
      "pcm_s16le",

      outputPath
    ]
  );
}


/* =========================================================
   FINAL VIDEO
========================================================= */

async function renderFinalVideo(
  jobId,
  moviePath,
  narrationPath,
  outputPath
) {
  updateJob(
    jobId,
    {
      stage:
        "FFmpeg",

      progress:
        90,

      message:
        "Rendering final recap video..."
    }
  );

  /*
   * Loop the movie video if narration is longer.
   * Original movie audio is removed.
   * Generated narration becomes the final audio.
   */

  await runCommand(
    "ffmpeg",
    [
      "-y",

      "-stream_loop",
      "-1",

      "-i",
      moviePath,

      "-i",
      narrationPath,

      "-map",
      "0:v:0",

      "-map",
      "1:a:0",

      "-c:v",
      "libx264",

      "-preset",
      "veryfast",

      "-crf",
      "23",

      "-pix_fmt",
      "yuv420p",

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
}


/* =========================================================
   HEALTH
========================================================= */

app.get(
  "/health",
  (req, res) => {
    res.json({
      ok: true,

      name:
        "YNT One Clips",

      status:
        "online"
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

    res.json(job);
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

    res.download(
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

  upload.single("movie"),

  async (req, res) => {
    let job = null;

    try {
      if (!req.file) {
        return res
          .status(400)
          .json({
            error:
              "Movie file is required."
          });
      }

      job =
        createJob();

      const jobFolder =
        path.join(
          JOB_DIR,
          job.id
        );

      fs.mkdirSync(
        jobFolder,
        {
          recursive: true
        }
      );

      const originalName =
        req.file.originalname ||
        "movie.mp4";

      const extension =
        path.extname(
          originalName
        ) || ".mp4";

      const moviePath =
        path.join(
          jobFolder,
          `movie${extension}`
        );

      fs.renameSync(
        req.file.path,
        moviePath
      );

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
            "Movie uploaded."
        }
      );

      /* -----------------------------------------
         OPTIONS FROM FRONTEND
      ----------------------------------------- */

      const language =
        req.body?.language ||
        "my";

      const style =
        req.body?.style ||
        "cinematic";

      /* -----------------------------------------
         VIDEO INFO
      ----------------------------------------- */

      updateJob(
        job.id,
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
        job.id,
        {
          duration
        }
      );

      /* -----------------------------------------
         AUDIO EXTRACTION
      ----------------------------------------- */

      const audioPath =
        path.join(
          jobFolder,
          "movie-audio.mp3"
        );

      updateJob(
        job.id,
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
         AUDIO CHUNKS
      ----------------------------------------- */

      const chunksDir =
        path.join(
          jobFolder,
          "audio-chunks"
        );

      updateJob(
        job.id,
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
        job.id,
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
          job.id,
          audioChunks
        );

      if (!transcript.text) {
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
        job.id,
        {
          stage:
            "Transcript",

          progress:
            52,

          message:
            "Transcript completed.",

          transcript: {
            characters:
              transcript.text.length,

            segments:
              transcript.segments.length
          }
        }
      );

      /* -----------------------------------------
         GEMINI RECAP
      ----------------------------------------- */

      const recap =
        await generateRecap(
          job.id,

          transcript.text,

          language,

          style
        );

      const recapPath =
        path.join(
          jobFolder,
          "recap.txt"
        );

      fs.writeFileSync(
        recapPath,
        recap,
        "utf8"
      );

      updateJob(
        job.id,
        {
          stage:
            "Gemini",

          progress:
            62,

          message:
            "Myanmar recap script completed.",

          recap: {
            characters:
              recap.length,

            path:
              recapPath
          }
        }
      );

      /* -----------------------------------------
         GEMINI TTS
      ----------------------------------------- */

      const ttsDir =
        path.join(
          jobFolder,
          "tts"
        );

      const ttsFiles =
        await generateTTS(
          job.id,
          recap,
          ttsDir
        );

      /* -----------------------------------------
         CONCAT TTS
      ----------------------------------------- */

      const narrationPath =
        path.join(
          jobFolder,
          "narration.wav"
        );

      const ttsListPath =
        path.join(
          jobFolder,
          "tts-list.txt"
        );

      updateJob(
        job.id,
        {
          stage:
            "Voice",

          progress:
            86,

          message:
            "Combining narration audio..."
        }
      );

      await concatAudio(
        ttsFiles,
        narrationPath,
        ttsListPath
      );

      /* -----------------------------------------
         FINAL VIDEO
      ----------------------------------------- */

      const outputPath =
        path.join(
          jobFolder,
          "YNT-One-Clips-Recap.mp4"
        );

      await renderFinalVideo(
        job.id,
        moviePath,
        narrationPath,
        outputPath
      );

      if (
        !fs.existsSync(
          outputPath
        )
      ) {
        throw new Error(
          "FFmpeg did not create the final MP4."
        );
      }

      const filename =
        `YNT-One-Clips-${job.id}.mp4`;

      updateJob(
        job.id,
        {
          status:
            "completed",

          stage:
            "Ready",

          progress:
            100,

          message:
            "Your recap video is ready.",

          output: {
            path:
              outputPath,

            filename,

            url:
              `/api/download/${job.id}`
          }
        }
      );

      return res.json({
        success:
          true,

        jobId:
          job.id,

        status:
          "completed",

        message:
          "YNT One Clips completed successfully.",

        download:
          `/api/download/${job.id}`
      });

    } catch (error) {

      console.error(
        "ONE CLIP ERROR:",
        error
      );

      if (job) {
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
              error.message,

            error:
              error.message
          }
        );
      }

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
            "Movie processing failed.",

          jobId:
            job?.id || null
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
   MULTER / GENERAL ERROR
========================================================= */

app.use(
  (err, req, res, next) => {
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
   START
========================================================= */

app.listen(
  PORT,
  () => {
    console.log(
      "===================================="
    );

    console.log(
      "YNT One Clips"
    );

    console.log(
      `Server running on port ${PORT}`
    );

    console.log(
      "Groq Whisper: READY"
    );

    console.log(
      "Gemini Recap: READY"
    );

    console.log(
      "Gemini TTS: READY"
    );

    console.log(
      "FFmpeg Renderer: READY"
    );

    console.log(
      "===================================="
    );
  }
);
