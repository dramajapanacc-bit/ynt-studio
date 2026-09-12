import formidable from "formidable";
import fs from "fs";
import { GoogleGenAI } from "@google/genai";

export const config = {
  api: {
    bodyParser: false,
  },
};

function parseForm(req) {
  return new Promise((resolve, reject) => {
    const form = formidable({
      multiples: false,
      keepExtensions: true,
      maxFileSize: 500 * 1024 * 1024,
    });

    form.parse(req, (error, fields, files) => {
      if (error) {
        reject(error);
        return;
      }

      resolve({ fields, files });
    });
  });
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({
      success: false,
      error: "POST method only",
    });
  }

  let localVideoPath = null;

  try {
    // ==============================
    // 1. CHECK API KEY
    // ==============================

    if (!process.env.GEMINI_API_KEY) {
      return res.status(500).json({
        success: false,
        error: "GEMINI_API_KEY မတွေ့ပါ။",
      });
    }

    // ==============================
    // 2. RECEIVE P1 VIDEO
    // ==============================

    const { files } = await parseForm(req);

    let videoFile = files.file;

    if (Array.isArray(videoFile)) {
      videoFile = videoFile[0];
    }

    if (!videoFile) {
      return res.status(400).json({
        success: false,
        error: "P1 Video file မရရှိပါ။",
      });
    }

    localVideoPath =
      videoFile.filepath ||
      videoFile.path;

    if (!localVideoPath) {
      throw new Error(
        "Uploaded video path မရရှိပါ။"
      );
    }

    // ==============================
    // 3. GEMINI CLIENT
    // ==============================

    const ai = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY,
    });

    // ==============================
    // 4. UPLOAD VIDEO TO GEMINI
    // ==============================

    const uploadedVideo =
      await ai.files.upload({
        file: localVideoPath,
        config: {
          mimeType:
            videoFile.mimetype ||
            "video/mp4",
        },
      });

    if (!uploadedVideo?.uri) {
      throw new Error(
        "Video ကို Gemini သို့ upload မအောင်မြင်ပါ။"
      );
    }

    // ==============================
    // 5. P2 VIDEO ANALYSIS PROMPT
    // ==============================

    const prompt = `
You are the P2 Video Analysis and Myanmar Recap Script Engine for YNT Studio.

Analyze the uploaded video itself.

Do NOT create a generic script.
Do NOT use placeholder text.
Do NOT assume the story.

Understand the actual video content.

Analyze:

- Main story
- Beginning
- Middle
- Ending
- Characters / people
- Important events
- Important dialogue
- Important visual moments
- Scene order
- Emotional moments
- Key information
- Approximate timestamps when possible

Then create a natural Myanmar-language recap script based ONLY on what is actually shown or heard in the video.

Myanmar narration requirements:

- Natural Myanmar language
- Easy to understand
- Sounds like a real human narrator
- Suitable for AI voice generation
- Storytelling style
- Smooth sentence flow
- Do not write like subtitles
- Do not use robotic wording
- Do not invent information
- Do not change the story
- Do not repeatedly use the same sentence
- Do not use ALL CAPS
- Do not make the narrator sound like shouting
- Keep important story details
- Follow the actual order of events

The recap should be suitable for P3 Myanmar AI Voice.

Return ONLY valid JSON.

Use exactly this structure:

{
  "title": "",
  "summary": "",
  "analysis": {
    "beginning": "",
    "middle": "",
    "ending": "",
    "characters": [],
    "importantEvents": [],
    "keyScenes": []
  },
  "recapScript": "",
  "segments": [
    {
      "start": "00:00",
      "end": "00:15",
      "narration": ""
    }
  ]
}
`;

    // ==============================
    // 6. GENERATE P2 RESULT
    // ==============================

    const result =
      await ai.models.generateContent({
        model: "gemini-2.5-flash",

        contents: [
          {
            role: "user",

            parts: [
              {
                fileData: {
                  fileUri:
                    uploadedVideo.uri,

                  mimeType:
                    uploadedVideo.mimeType ||
                    videoFile.mimetype ||
                    "video/mp4",
                },
              },

              {
                text: prompt,
              },
            ],
          },
        ],
      });

    // ==============================
    // 7. GET AI TEXT
    // ==============================

    let output =
      result.text || "";

    output = output
      .replace(/^```json\s*/i, "")
      .replace(/^```\s*/i, "")
      .replace(/\s*```$/i, "")
      .trim();

    if (!output) {
      throw new Error(
        "P2 AI response မရရှိပါ။"
      );
    }

    // ==============================
    // 8. PARSE JSON
    // ==============================

    let data;

    try {
      data = JSON.parse(output);
    } catch (jsonError) {
      console.error(
        "Invalid Gemini JSON:",
        output
      );

      throw new Error(
        "P2 AI က valid JSON မပြန်ပေးပါ။"
      );
    }

    // ==============================
    // 9. CHECK RECAP SCRIPT
    // ==============================

    if (
      !data.recapScript ||
      typeof data.recapScript !== "string"
    ) {
      throw new Error(
        "Myanmar Recap Script မရရှိပါ။"
      );
    }

    // ==============================
    // 10. RETURN P2 DATA
    // ==============================

    return res.status(200).json({
      success: true,

      title:
        data.title ||
        "YNT Myanmar Recap",

      summary:
        data.summary || "",

      analysis:
        data.analysis || {
          beginning: "",
          middle: "",
          ending: "",
          characters: [],
          importantEvents: [],
          keyScenes: [],
        },

      recapScript:
        data.recapScript,

      segments:
        Array.isArray(data.segments)
          ? data.segments
          : [],

      sourceVideo: {
        filename:
          videoFile.originalFilename ||
          videoFile.name ||
          "video",

        mimeType:
          videoFile.mimetype ||
          "video/mp4",
      },
    });

  } catch (error) {

    console.error(
      "YNT P2 ERROR:",
      error
    );

    return res.status(500).json({
      success: false,

      error:
        error.message ||
        "P2 Video Analysis Error",
    });

  } finally {

    // ==============================
    // 11. DELETE TEMP VIDEO
    // ==============================

    if (
      localVideoPath &&
      fs.existsSync(localVideoPath)
    ) {
      try {
        fs.unlinkSync(
          localVideoPath
        );
      } catch (cleanupError) {
        console.warn(
          "Temporary video cleanup failed:",
          cleanupError.message
        );
      }
    }
  }
}
