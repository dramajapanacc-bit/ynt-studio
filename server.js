import express from "express";
import multer from "multer";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import crypto from "crypto";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

const ROOT = process.cwd();
const PUBLIC_DIR = path.join(ROOT, "public");
const UPLOAD_DIR = path.join(ROOT, "uploads");
const JOB_DIR = path.join(ROOT, "jobs");

for (const dir of [PUBLIC_DIR, UPLOAD_DIR, JOB_DIR]) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

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
      cb(new Error("Only video files are allowed."));
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
    message: "Job created",
    createdAt: new Date().toISOString(),
    output: null,
    error: null
  };

  jobs.set(id, job);

  return job;
}

function updateJob(id, data) {
  const job = jobs.get(id);

  if (!job) return;

  Object.assign(job, data);
}

app.get("/health", (req, res) => {
  res.json({
    ok: true,
    name: "YNT One Clips",
    status: "online"
  });
});

app.get("/api/status/:id", (req, res) => {
  const job = jobs.get(req.params.id);

  if (!job) {
    return res.status(404).json({
      error: "Job not found"
    });
  }

  res.json(job);
});

app.post("/api/one-clip", upload.single("movie"), async (req, res) => {
  let job = null;

  try {
    if (!req.file) {
      return res.status(400).json({
        error: "Movie file is required."
      });
    }

    job = createJob();

    updateJob(job.id, {
      status: "processing",
      stage: "Upload",
      progress: 5,
      message: "Movie uploaded successfully."
    });

    const jobFolder = path.join(JOB_DIR, job.id);

    fs.mkdirSync(jobFolder, {
      recursive: true
    });

    const originalName = req.file.originalname || "movie";
    const extension = path.extname(originalName) || ".mp4";

    const moviePath = path.join(
      jobFolder,
      `movie${extension}`
    );

    fs.renameSync(
      req.file.path,
      moviePath
    );

    updateJob(job.id, {
      stage: "Upload",
      progress: 10,
      message: "Movie file saved."
    });

    /*
      NEXT STEPS WILL BE CONNECTED HERE:

      1. FFmpeg
      2. Groq Whisper
      3. Transcript
      4. Gemini Recap
      5. Gemini TTS
      6. FFmpeg Final Render
    */

    updateJob(job.id, {
      status: "ready",
      stage: "Backend Ready",
      progress: 10,
      message: "Upload received. Processing pipeline is ready.",
      movie: {
        name: originalName,
        size: req.file.size,
        path: moviePath
      }
    });

    return res.json({
      success: true,
      jobId: job.id,
      status: "ready",
      message: "Movie uploaded successfully."
    });

  } catch (error) {

    console.error(error);

    if (job) {
      updateJob(job.id, {
        status: "error",
        stage: "Error",
        progress: 0,
        message: error.message,
        error: error.message
      });
    }

    return res.status(500).json({
      error: error.message || "Server error."
    });
  }
});

app.use(express.static(PUBLIC_DIR));

app.use((err, req, res, next) => {
  console.error(err);

  if (err instanceof multer.MulterError) {
    return res.status(400).json({
      error: err.message
    });
  }

  return res.status(500).json({
    error: err.message || "Server error."
  });
});

app.listen(PORT, () => {
  console.log("------------------------------------");
  console.log("YNT One Clips");
  console.log(`Server running on port ${PORT}`);
  console.log("------------------------------------");
});
