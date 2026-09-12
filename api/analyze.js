import { GoogleGenAI } from "@google/genai";

export const config = {
  api: {
    bodyParser: true,
  },
};

export const maxDuration = 300;

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
});

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function cleanJson(text) {
  return String(text || "")
    .replace(/```json/gi, "")
    .replace(/```/g, "")
    .trim();
}

async function downloadBlob(blobUrl) {
  if (!blobUrl) {
    throw new Error("blobUrl is required.");
  }

  const token = process.env.BLOB_READ_WRITE_TOKEN;

  if (!token) {
    throw new Error(
      "BLOB_READ_WRITE_TOKEN is not configured."
    );
  }

  const response = await fetch(blobUrl, {
    method: "GET",
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");

    throw new Error(
      `Blob download failed: ${response.status} ${errorText}`
    );
  }

  const contentType =
    response.headers.get("content-type") ||
    "video/mp4";

  const arrayBuffer =
    await response.arrayBuffer();

  if (!arrayBuffer.byteLength) {
    throw new Error(
      "Downloaded video is empty."
    );
  }

  return {
    buffer: Buffer.from(arrayBuffer),
    mimeType: contentType,
  };
}

async function waitForGeminiFile(file) {
  let current = file;

  for (let i = 0; i < 120; i++) {
    const state =
      current?.state?.toString?.() ||
      current?.state ||
      "";

    console.log(
      "Gemini video state:",
      state
    );

    if (state === "ACTIVE") {
      return current;
    }

    if (state === "FAILED") {
      throw new Error(
        "Gemini failed to process the video."
      );
    }

    await sleep(5000);

    current =
      await ai.files.get({
        name: current.name,
      });
  }

  throw new Error(
    "Gemini video processing timed out."
  );
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({
      success: false,
      error: "POST method only",
    });
  }

  try {
    if (!process.env.GEMINI_API_KEY) {
      return res.status(500).json({
        success: false,
        error:
          "GEMINI_API_KEY is not configured.",
      });
    }

    if (!process.env.BLOB_READ_WRITE_TOKEN) {
      return res.status(500).json({
        success: false,
        error:
          "BLOB_READ_WRITE_TOKEN is not configured.",
      });
    }

    /*
     * P2 က video file ကို မပို့တော့ပါဘူး။
     *
     * P1 upload ပြီးသား Vercel Blob URL ကိုပဲ ပို့ရပါမယ်။
     */

    const {
      blobUrl,
      fileName,
      mimeType,
    } = req.body || {};

    if (!blobUrl) {
      return res.status(400).json({
        success: false,
        error:
          "blobUrl is required. Upload the P1 video to Vercel Blob first.",
      });
    }

    console.log(
      "P2 Analyze started:",
      fileName || "video"
    );

    /*
     * 1. Vercel Blob ကနေ video ကို download
     */

    const {
      buffer,
      mimeType: downloadedMimeType,
    } =
      await downloadBlob(blobUrl);

    const finalMimeType =
      mimeType ||
      downloadedMimeType ||
      "video/mp4";

    console.log(
      "Video downloaded:",
      buffer.length,
      "bytes"
    );

    /*
     * 2. Gemini Files API သို့ upload
     */

    let uploadedFile =
      await ai.files.upload({
        file: new Blob(
          [buffer],
          {
            type: finalMimeType,
          }
        ),
        config: {
          displayName:
            fileName || "ynt-video",
          mimeType: finalMimeType,
        },
      });

    console.log(
      "Gemini upload complete:",
      uploadedFile.name
    );

    /*
     * 3. Gemini video processing ပြီးတဲ့အထိ စောင့်
     */

    uploadedFile =
      await waitForGeminiFile(
        uploadedFile
      );

    console.log(
      "Gemini video ACTIVE:",
      uploadedFile.name
    );

    /*
     * 4. P2 Myanmar Recap Prompt
     */

    const prompt = `
You are the video analysis and Myanmar movie recap engine for YNT Studio.

Analyze the ENTIRE uploaded video carefully.

Your job is to understand the actual story shown in the video and create a detailed, natural Myanmar recap narration.

IMPORTANT RULES:

1. Analyze the whole video, not only the first few seconds.
2. Follow the real chronological order.
3. Identify important characters.
4. Identify important scenes and actions.
5. Understand the beginning, middle and ending.
6. Pay attention to visual events.
7. Do not invent events that are not shown.
8. Do not hallucinate character names.
9. If a character name is unknown, describe the character naturally.
10. Write natural spoken Myanmar.
11. Do not translate English word-by-word.
12. The narration must sound like a human Myanmar movie recap narrator.
13. Do not use headings inside recapScript.
14. Do not make recapScript extremely short.
15. Cover the important events throughout the entire video.
16. Keep the narration suitable for later Myanmar AI voice generation.
17. Do not output markdown.
18. Return valid JSON only.

Create this JSON:

{
  "title": "Myanmar recap title",
  "summary": "short Myanmar summary",
  "recapScript": "detailed natural Myanmar narration covering the story from beginning to ending",
  "analysis": {
    "story": "detailed story explanation",
    "characters": [
      {
        "name": "",
        "role": "",
        "description": ""
      }
    ],
    "events": [
      {
        "order": 1,
        "description": "",
        "importance": 1
      }
    ],
    "ending": "explain how the video ends"
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

    /*
     * 5. Gemini Video Analysis
     */

    const result =
      await ai.models.generateContent({
        model:
          "gemini-3.8-flash",

        contents: [
          {
            role: "user",

            parts: [
              {
                fileData: {
                  fileUri:
                    uploadedFile.uri,

                  mimeType:
                    uploadedFile.mimeType ||
                    finalMimeType,
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
      result?.text || "";

    if (!text.trim()) {
      throw new Error(
        "Gemini returned an empty response."
      );
    }

    console.log(
      "Gemini response received."
    );

    /*
     * 6. JSON clean
     */

    const cleaned =
      cleanJson(text);

    let data;

    try {
      data =
        JSON.parse(cleaned);
    } catch (jsonError) {
      console.error(
        "Gemini JSON parse error:",
        jsonError
      );

      /*
       * Gemini က JSON မမှန်ရင်
       * raw response ကို script အဖြစ် မသုံးဘဲ
       * error ပြန်ပေးမယ်။
       */

      throw new Error(
        "Gemini returned invalid JSON."
      );
    }

    /*
     * 7. Final response
     */

    return res.status(200).json({
      success: true,

      title:
        data.title ||
        "YNT Recap",

      summary:
        data.summary ||
        "",

      recapScript:
        data.recapScript ||
        "",

      analysis:
        data.analysis ||
        {
          story: "",
          characters: [],
          events: [],
          ending: "",
        },

      segments:
        Array.isArray(
          data.segments
        )
          ? data.segments
          : [],
    });

  } catch (error) {
    console.error(
      "YNT P2 ANALYZE ERROR:",
      error
    );

    return res.status(500).json({
      success: false,

      error:
        error?.message ||
        "P2 Analyze failed.",

      details:
        process.env.NODE_ENV ===
        "development"
          ? String(error)
          : undefined,
    });
  }
}
