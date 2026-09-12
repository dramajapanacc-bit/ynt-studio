import { GoogleGenAI } from "@google/genai";
import Busboy from "busboy";

export const config = {
  api: {
    bodyParser: false,
  },
};

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
});

function parseMultipart(req) {
  return new Promise((resolve, reject) => {
    const bb = Busboy({
      headers: req.headers,
      limits: {
        fileSize: 500 * 1024 * 1024,
      },
    });

    let videoBuffer = null;
    let fileName = "";
    let mimeType = "";
    const fields = {};

    const chunks = [];

    bb.on("field", (name, value) => {
      fields[name] = value;
    });

    bb.on("file", (name, file, info) => {
      fileName = info.filename || "video";
      mimeType = info.mimeType || "video/mp4";

      file.on("data", (chunk) => {
        chunks.push(chunk);
      });
    });

    bb.on("error", reject);

    bb.on("finish", () => {
      videoBuffer = Buffer.concat(chunks);

      if (!videoBuffer.length) {
        return reject(
          new Error("Video file is required.")
        );
      }

      resolve({
        videoBuffer,
        fileName,
        mimeType,
        fields,
      });
    });

    req.pipe(bb);
  });
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({
      error: "POST method only",
    });
  }

  try {
    const {
      videoBuffer,
      fileName,
      mimeType,
    } = await parseMultipart(req);

    if (!process.env.GEMINI_API_KEY) {
      return res.status(500).json({
        error:
          "GEMINI_API_KEY is not configured.",
      });
    }

    /*
     * Gemini File API သို့ video upload
     */

    const uploadedFile =
      await ai.files.upload({
        file: new Blob(
          [videoBuffer],
          { type: mimeType }
        ),
        config: {
          displayName: fileName,
        },
      });


    /*
     * Video ကို story အနေနဲ့ Analyze
     */

    const prompt = `
You are an expert movie/video recap writer.

Analyze the uploaded video carefully.

Understand:
1. Main story
2. Important characters
3. Important events
4. Scene progression
5. Beginning, middle and ending
6. Important visual actions
7. Emotional moments

Then create a NATURAL MYANMAR VIDEO RECAP SCRIPT.

Requirements:
- Write in natural spoken Myanmar.
- Do not translate word-for-word.
- Do not invent events that are not in the video.
- Keep the story in correct chronological order.
- Make it suitable for Myanmar narration.
- Do not write headings inside the narration.
- Do not use excessive punctuation.
- The script should sound like a human narrator explaining a movie.
- Make the narration detailed enough for the whole source video.
- Do NOT shorten the entire video into only a few sentences.

Return JSON only:

{
  "title": "short Myanmar recap title",
  "summary": "short Myanmar summary",
  "recapScript": "full Myanmar narration script",
  "analysis": {
    "story": "",
    "characters": [],
    "events": [],
    "ending": ""
  },
  "segments": [
    {
      "start": 0,
      "end": 10,
      "description": "",
      "importance": 1
    }
  ]
}
`;


    const result =
      await ai.models.generateContent({
        model:
          "gemini-2.5-flash",

        contents: [
          {
            role: "user",
            parts: [
              {
                fileData: {
                  fileUri:
                    uploadedFile.uri,
                  mimeType:
                    uploadedFile.mimeType,
                },
              },
              {
                text: prompt,
              },
            ],
          },
        ],
      });


    const text =
      result.text || "";


    /*
     * JSON clean
     */

    let cleaned =
      text
        .replace(/```json/gi, "")
        .replace(/```/g, "")
        .trim();


    let data;

    try {

      data =
        JSON.parse(cleaned);

    } catch {

      /*
       * Gemini JSON မမှန်ရင်
       * text ကို fallback အနေနဲ့ သုံး
       */

      data = {
        title: "YNT Recap",
        summary:
          "Video analysis completed.",
        recapScript: text,
        analysis: {},
        segments: [],
      };

    }


    return res.status(200).json({
      success: true,

      title:
        data.title || "YNT Recap",

      summary:
        data.summary || "",

      recapScript:
        data.recapScript || "",

      analysis:
        data.analysis || {},

      segments:
        Array.isArray(data.segments)
          ? data.segments
          : [],
    });


  } catch (error) {

    console.error(
      "YNT ANALYZE ERROR:",
      error
    );

    return res.status(500).json({
      success: false,
      error:
        error.message ||
        "P2 backend error",
    });

  }
}
