import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
});

export default async function handler(req, res) {
  // POST only
  if (req.method !== "POST") {
    return res.status(405).json({
      success: false,
      error: "POST method only",
    });
  }

  try {
    // API key check
    if (!process.env.GEMINI_API_KEY) {
      return res.status(500).json({
        success: false,
        error: "GEMINI_API_KEY မထည့်ရသေးပါ။",
      });
    }

    // Request body
    const body = req.body || {};

    const text =
      typeof body.text === "string"
        ? body.text.trim()
        : "";

    const title =
      typeof body.title === "string"
        ? body.title.trim()
        : "";

    const language =
      typeof body.language === "string"
        ? body.language
        : "my";

    // Empty text
    if (!text) {
      return res.status(400).json({
        success: false,
        error: "သတင်းစာသား အရင်ထည့်ပါ။",
      });
    }

    /*
     * Gemini prompt
     *
     * News text → Story → Scenes
     */

    const prompt = `
You are an expert Myanmar news video storyboard writer.

The user provides a news/story text below.

Your job is to understand ONLY the information contained in the user's text
and convert it into a short-form video storyboard.

IMPORTANT RULES:

1. Do NOT invent facts.
2. Do NOT add people, places, dates, events or numbers that are not supported
   by the provided news text.
3. Keep the original meaning.
4. Organize the story in logical chronological order when possible.
5. Each scene should represent one clear visual moment.
6. Scene descriptions must be useful for generating an AI image.
7. Write the narration idea naturally in Myanmar language.
8. Make the storyboard suitable for TikTok / Reels / Shorts.
9. Keep scenes concise but informative.
10. If the source text is uncertain about something, do not pretend it is certain.
11. Return JSON only.
12. Do not use Markdown code fences.

Language:
${language}

News title:
${title || "မသတ်မှတ်ရသေးပါ"}

News text:
"""
${text}
"""

Return exactly this JSON structure:

{
  "title": "Myanmar video title",
  "summary": "Short Myanmar summary",
  "hook": "Short opening hook for the video",
  "scenes": [
    {
      "id": 1,
      "title": "Scene title in Myanmar",
      "description": "What should be visually shown",
      "imagePrompt": "Detailed English prompt for an AI image generator",
      "narration": "Natural Myanmar narration for this scene",
      "duration": 4,
      "importance": 5
    }
  ]
}

SCENE RULES:

- Create 4 to 10 scenes depending on the amount of information.
- Do not create unnecessary scenes.
- duration should normally be between 3 and 8 seconds.
- importance should be between 1 and 5.
- imagePrompt must describe the actual subject of the scene.
- Do not put text, subtitles, logos or watermarks inside imagePrompt.
- imagePrompt should be visually specific.
- narration must be natural spoken Myanmar.
- The final scene should cover the ending/current outcome if the source text contains one.

The result will later be used by another system to generate images,
Myanmar voice and a vertical short video.
`;

    /*
     * Gemini
     */

    const result = await ai.models.generateContent({
      model: "gemini-2.5-flash",

      contents: [
        {
          role: "user",
          parts: [
            {
              text: prompt,
            },
          ],
        },
      ],

      config: {
        temperature: 0.3,

        responseMimeType: "application/json",
      },
    });

    /*
     * Get Gemini response
     */

    let rawText = result.text || "";

    rawText = rawText
      .replace(/```json/gi, "")
      .replace(/```/g, "")
      .trim();

    if (!rawText) {
      throw new Error(
        "Gemini က response ပြန်မပေးပါ။"
      );
    }

    /*
     * Parse JSON
     */

    let data;

    try {
      data = JSON.parse(rawText);
    } catch (parseError) {
      console.error(
        "GEMINI JSON ERROR:",
        rawText
      );

      throw new Error(
        "AI response ကို JSON အဖြစ် မဖတ်နိုင်ပါ။"
      );
    }

    /*
     * Validate scenes
     */

    const scenes =
      Array.isArray(data.scenes)
        ? data.scenes
        : [];

    if (!scenes.length) {
      throw new Error(
        "AI က Scene မထုတ်ပေးနိုင်သေးပါ။"
      );
    }

    /*
     * Clean scenes
     */

    const cleanScenes =
      scenes.map((scene, index) => ({
        id:
          Number(scene.id) ||
          index + 1,

        title:
          typeof scene.title === "string"
            ? scene.title
            : `Scene ${index + 1}`,

        description:
          typeof scene.description === "string"
            ? scene.description
            : "",

        imagePrompt:
          typeof scene.imagePrompt === "string"
            ? scene.imagePrompt
            : "",

        narration:
          typeof scene.narration === "string"
            ? scene.narration
            : "",

        duration:
          Math.min(
            8,
            Math.max(
              3,
              Number(scene.duration) || 4
            )
          ),

        importance:
          Math.min(
            5,
            Math.max(
              1,
              Number(scene.importance) || 3
            )
          ),
      }));

    /*
     * Final response
     */

    return res.status(200).json({
      success: true,

      title:
        typeof data.title === "string"
          ? data.title
          : title || "YNT AI News Video",

      summary:
        typeof data.summary === "string"
          ? data.summary
          : "",

      hook:
        typeof data.hook === "string"
          ? data.hook
          : "",

      scenes: cleanScenes,
    });

  } catch (error) {
    console.error(
      "YNT ANALYZE NEWS ERROR:",
      error
    );

    return res.status(500).json({
      success: false,
      error:
        error?.message ||
        "AI News Analyze error",
    });
  }
}
