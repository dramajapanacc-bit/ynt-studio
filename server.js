import express from "express";
import multer from "multer";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import crypto from "crypto";
import { execFile } from "child_process";
import { promisify } from "util";
import Groq from "groq-sdk";

dotenv.config();

const execFileAsync = promisify(execFile);

const app = express();

const PORT = process.env.PORT || 3000;

const ROOT = process.cwd();

const PUBLIC_DIR = path.join(ROOT, "public");
const UPLOAD_DIR = path.join(ROOT, "uploads");
const JOB_DIR = path.join(ROOT, "jobs");

for (const dir of [
  PUBLIC_DIR,
  UPLOAD_DIR,
  JOB_DIR
]) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, {
      recursive: true
    });
  }
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
    } else {
      cb(
        new Error(
          "Only MP4, MKV, MOV, WEBM or AVI files are allowed."
        )
      );
    }
  }
});


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

    transcript: null,

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


function requireEnv(name) {

  const value = process.env[name];

  if (!value) {
    throw new Error(
      `${name} is not configured on the server.`
    );
  }

  return value;
}


async function runCommand(
  command,
  args
) {

  const result =
    await execFileAsync(
      command,
      args,
      {
        maxBuffer:
          20 * 1024 * 1024
      }
    );

  return result;
}


/*
|--------------------------------------------------------------------------
| FFprobe
|--------------------------------------------------------------------------
*/

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

  if (!Number.isFinite(duration)) {
    throw new Error(
      "Unable to read video duration."
    );
  }

  return duration;
}


/*
|--------------------------------------------------------------------------
| FFmpeg
|--------------------------------------------------------------------------
*/

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


/*
|--------------------------------------------------------------------------
| Split audio into chunks
|--------------------------------------------------------------------------
*/

async function splitAudio(
  audioPath,
  outputDir
) {

  if (!fs.existsSync(outputDir)) {

    fs.mkdirSync(
      outputDir,
      {
        recursive: true
      }
    );
  }

  await runCommand(
    "ffmpeg",
    [
      "-y",

      "-i",
      audioPath,

      "-f",
      "segment",

      "-segment_time",
      "90",

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

  const chunks =
    fs
      .readdirSync(outputDir)
      .filter(
        file =>
          file.startsWith("chunk-") &&
          file.endsWith(".mp3")
      )
      .sort();

  if (!chunks.length) {
    throw new Error(
      "FFmpeg did not create audio chunks."
    );
  }

  return chunks.map(
    file =>
      path.join(
        outputDir,
        file
      )
  );
}


/*
|--------------------------------------------------------------------------
| Groq Whisper
|--------------------------------------------------------------------------
*/

async function transcribeChunk(
  groq,
  audioPath,
  chunkIndex
) {

  const fileStream =
    fs.createReadStream(
      audioPath
    );

  const response =
    await groq.audio.transcriptions.create(
      {
        file: fileStream,

        model:
          "whisper-large-v3-turbo",

        response_format:
          "verbose_json",

        timestamp_granularities:
          ["segment"],

        temperature:
          0
      }
    );

  const offset =
    chunkIndex * 90;

  const segments =
    Array.isArray(
      response.segments
    )
      ? response.segments
      : [];

  return {
    text:
      response.text || "",

    segments:
      segments.map(
        segment => ({
          start:
            Number(segment.start || 0)
            + offset,

          end:
            Number(segment.end || 0)
            + offset,

          text:
            String(
              segment.text || ""
            ).trim()
        })
      )
  };
}


/*
|--------------------------------------------------------------------------
| Complete movie transcription
|--------------------------------------------------------------------------
*/

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

    updateJob(
      jobId,
      {
        stage: "Whisper",

        progress:
          15 +
          Math.round(
            (i /
              audioChunks.length) *
              45
          ),

        message:
          `Transcribing audio ${i + 1} / ${audioChunks.length}...`
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


/*
|--------------------------------------------------------------------------
| Save transcript
|--------------------------------------------------------------------------
*/

function saveTranscript(
  jobFolder,
  transcript
) {

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

  return transcriptPath;
}


/*
|--------------------------------------------------------------------------
| Health
|--------------------------------------------------------------------------
*/

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


/*
|--------------------------------------------------------------------------
| Job status
|--------------------------------------------------------------------------
*/

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


/*
|--------------------------------------------------------------------------
| One Clip
|--------------------------------------------------------------------------
*/

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
        "movie";


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
          stage:
            "Upload",

          progress:
            10,

          message:
            "Movie file saved."
        }
      );


      /*
      |--------------------------------------------------------------------------
      | Read video duration
      |--------------------------------------------------------------------------
      */

      const duration =
        await getVideoDuration(
          moviePath
        );


      updateJob(
        job.id,
        {
          duration,

          stage:
            "FFmpeg",

          progress:
            12,

          message:
            "Extracting movie audio..."
        }
      );


      /*
      |--------------------------------------------------------------------------
      | Extract audio
      |--------------------------------------------------------------------------
      */

      const audioPath =
        path.join(
          jobFolder,
          "movie-audio.mp3"
        );


      await extractAudio(
        moviePath,
        audioPath
      );


      updateJob(
        job.id,
        {
          stage:
            "FFmpeg",

          progress:
            14,

          message:
            "Splitting audio into chunks..."
        }
      );


      /*
      |--------------------------------------------------------------------------
      | Split audio
      |--------------------------------------------------------------------------
      */

      const chunksDir =
        path.join(
          jobFolder,
          "audio-chunks"
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
            audioChunks.length,

          stage:
            "Whisper",

          progress:
            15,

          message:
            "Starting Groq Whisper..."
        }
      );


      /*
      |--------------------------------------------------------------------------
      | Groq transcription
      |--------------------------------------------------------------------------
      */

      const transcript =
        await transcribeMovie(
          job.id,

          audioChunks
        );


      /*
      |--------------------------------------------------------------------------
      | Save transcript
      |--------------------------------------------------------------------------
      */

      saveTranscript(
        jobFolder,

        transcript
      );


      updateJob(
        job.id,
        {
          status:
            "ready",

          stage:
            "Transcript",

          progress:
            65,

          message:
            "Transcript generated successfully.",

          transcript: {
            text:
              transcript.text,

            segments:
              transcript.segments.length
          }
        }
      );


      return res.json({
        success:
          true,

        jobId:
          job.id,

        status:
          "ready",

        duration,

        chunks:
          audioChunks.length,

        message:
          "Movie uploaded and transcribed."
      });


    } catch (error) {

      console.error(
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


      return res
        .status(500)
        .json({
          error:
            error.message ||
            "Processing failed."
        });
    }
  }
);


/*
|--------------------------------------------------------------------------
| Serve frontend
|--------------------------------------------------------------------------
*/

app.use(
  express.static(
    PUBLIC_DIR
  )
);


/*
|--------------------------------------------------------------------------
| Error handler
|--------------------------------------------------------------------------
*/

app.use(
  (err, req, res, next) => {

    console.error(
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


/*
|--------------------------------------------------------------------------
| Start
|--------------------------------------------------------------------------
*/

app.listen(
  PORT,
  () => {

    console.log(
      "------------------------------------"
    );

    console.log(
      "YNT One Clips"
    );

    console.log(
      `Server running on port ${PORT}`
    );

    console.log(
      "Groq Whisper pipeline ready"
    );

    console.log(
      "------------------------------------"
    );
  }
);
