import { GoogleGenAI } from "@google/genai";
import formidable from "formidable";
import fs from "fs";

export const config = {
  api: {
    bodyParser: false,
  },
};

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
});

function parseForm(req) {
  return new Promise((resolve, reject) => {
    const form = formidable({
      uploadDir: "/tmp",
      keepExtensions: true,
      maxFileSize: 4500000,
      multiples: false,
    });

    form.parse(req, (err, fields, files) => {
      if (err) {
        reject(err);
        return;
      }

      resolve({ fields, files });
    });
  });
}

function getValue(value) {
  if (Array.isArray(value)) {
    return value[0];
  }

  return value || "";
}

export default async function handler(req, res) {

  if (req.method !== "POST") {
    return res.status(405).json({
      error: "POST method only",
    });
  }

  let uploadedFile = null;

  try {

    if (!process.env.GEMINI_API_KEY) {
      return res.status(500).json({
        error: "GEMINI_API_KEY မတွေ့ပါ။",
      });
    }

    /* =========================
       RECEIVE FILE
    ========================= */

    const { fields, files } = await parseForm(req);

    uploadedFile =
      files.file?.[0] ||
      files.file;

    if (!uploadedFile) {
      return res.status(400).json({
        error: "Video / Audio ဖိုင် မရရှိပါ။",
      });
    }

    const sourceLanguage =
      getValue(fields.sourceLanguage) || "zh";

    const languageNames = {
      zh: "Chinese",
      en: "English",
      ja: "Japanese",
      ko: "Korean",
    };

    const sourceName =
      languageNames[sourceLanguage] || "Chinese";


    /* =========================
       UPLOAD TO GEMINI FILES API
    ========================= */

    const geminiFile =
      await ai.files.upload({
        file: uploadedFile.filepath,
        config: {
          mimeType:
            uploadedFile.mimetype ||
            "video/mp4",
          displayName:
            uploadedFile.originalFilename ||
            "YNT Studio Video",
        },
      });


    /* =========================
       WAIT FOR VIDEO PROCESSING
    ========================= */

    let processedFile = geminiFile;

    for (let i = 0; i < 60; i++) {

      if (
        processedFile.state === "ACTIVE" ||
        processedFile.state?.name === "ACTIVE"
      ) {
        break;
      }

      if (
        processedFile.state === "FAILED" ||
        processedFile.state?.name === "FAILED"
      ) {
        throw new Error(
          "Gemini Video Processing Failed"
        );
      }

      await new Promise(resolve =>
        setTimeout(resolve, 3000)
      );

      processedFile =
        await ai.files.get({
          name: geminiFile.name,
        });
    }


    if (
      processedFile.state !== "ACTIVE" &&
      processedFile.state?.name !== "ACTIVE"
    ) {
      throw new Error(
        "Video processing အချိန်ကြာလွန်းနေပါသည်။"
      );
    }


    /* =========================
       AI SRT GENERATION
    ========================= */

    const prompt = `
You are a professional subtitle generator.

The uploaded video/audio is primarily in ${sourceName}.

Create a complete Myanmar (Burmese) subtitle file from the spoken dialogue.

IMPORTANT RULES:

1. Listen to the entire video/audio.
2. Do NOT summarize the video.
3. Do NOT skip dialogue.
4. Translate the actual spoken dialogue into natural Myanmar Burmese.
5. Preserve the meaning and order of every spoken sentence.
6. Create accurate subtitle timestamps.
7. Use standard SRT format.
8. Each subtitle should normally contain 1-2 lines.
9. Keep subtitle duration natural and synchronized with speech.
10. Do not add explanations.
11. Do not use Markdown.
12. Do not use code fences.
13. Output ONLY the SRT content.

Example format:

1
00:00:01,000 --> 00:00:04,000
မင်း ဘယ်ကိုသွားနေတာလဲ။

2
00:00:04,200 --> 00:00:07,000
ငါ အိမ်ကိုပြန်နေတာပါ။

Now process the ENTIRE uploaded media and return the complete Myanmar SRT.
`;


    const response =
      await ai.models.generateContent({
        model: "gemini-3.7-flash",
        contents: [
          processedFile,
          prompt,
        ],
      });


    let srt =
      response.text || "";


    /* =========================
       CLEAN AI OUTPUT
    ========================= */

    srt = srt
      .replace(/```srt/gi, "")
      .replace(/```/g, "")
      .trim();


    if (!srt) {
      throw new Error(
        "AI က SRT ရလဒ် မပြန်ပေးပါ။"
      );
    }


    /* =========================
       RETURN RESULT
    ========================= */

    return res.status(200).json({
      success: true,
      srt: srt,
      sourceLanguage: sourceName,
      fileName:
        uploadedFile.originalFilename ||
        "video",
    });


  } catch (error) {

    console.error(
      "YNT TRANSLATE ERROR:",
      error
    );

    return res.status(500).json({
      success: false,
      error:
        error?.message ||
        "AI Translation Error",
    });


  } finally {

    /* =========================
       DELETE TEMP FILE
    ========================= */

    try {

      if (
        uploadedFile?.filepath &&
        fs.existsSync(uploadedFile.filepath)
      ) {
        fs.unlinkSync(
          uploadedFile.filepath
        );
      }

    } catch (cleanupError) {

      console.error(
        "Cleanup Error:",
        cleanupError
      );

    }

  }
}
