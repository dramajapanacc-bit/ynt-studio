import { GoogleGenAI } from "@google/genai";
import { put } from "@vercel/blob";

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
});

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
        error: "GEMINI_API_KEY မထည့်ရသေးပါ။",
      });
    }

    if (!process.env.BLOB_READ_WRITE_TOKEN) {
      return res.status(500).json({
        success: false,
        error: "BLOB_READ_WRITE_TOKEN မထည့်ရသေးပါ။",
      });
    }

    const body = req.body || {};

    const scenes = Array.isArray(body.scenes)
      ? body.scenes
      : [];

    const style =
      typeof body.style === "string"
        ? body.style
        : "news";

    const ratio =
      typeof body.ratio === "string"
        ? body.ratio
        : "9:16";

    if (!scenes.length) {
      return res.status(400).json({
        success: false,
        error: "Scene မရှိသေးပါ။ P2 Analyze ကိုအရင်လုပ်ပါ။",
      });
    }

    const results = [];

    /*
     * Scene တစ်ခုချင်းစီကို image generate
     */

    for (let i = 0; i < scenes.length; i++) {
      const scene = scenes[i];

      const imagePrompt =
        typeof scene.imagePrompt === "string"
          ? scene.imagePrompt.trim()
          : "";

      if (!imagePrompt) {
        results.push({
          id: scene.id || i + 1,
          success: false,
          imageUrl: "",
          error: "imagePrompt မရှိပါ။",
        });

        continue;
      }

      const finalPrompt = `
Create a high-quality vertical short-video image.

Aspect ratio: ${ratio}

Visual style: ${style}

Scene:
${imagePrompt}

Requirements:
- Follow the scene description accurately.
- Do not invent important facts.
- Cinematic and visually clear.
- Suitable for TikTok, Reels and Shorts.
- Strong subject focus.
- Natural lighting.
- High detail.
- No subtitles.
- No text.
- No logo.
- No watermark.
`;

      try {
        const response =
          await ai.models.generateContent({
            model: "gemini-2.5-flash-image",
            contents: [
              {
                role: "user",
                parts: [
                  {
                    text: finalPrompt,
                  },
                ],
              },
            ],
          });

        let imageBase64 = null;
        let mimeType = "image/png";

        /*
         * Gemini response ထဲက image data ရှာ
         */

        const parts =
          response?.candidates?.[0]?.content?.parts || [];

        for (const part of parts) {
          if (part.inlineData) {
            imageBase64 =
              part.inlineData.data;

            mimeType =
              part.inlineData.mimeType ||
              "image/png";

            break;
          }
        }

        if (!imageBase64) {
          throw new Error(
            "Gemini က image data ပြန်မပေးပါ။"
          );
        }

        /*
         * Base64 → Buffer
         */

        const imageBuffer =
          Buffer.from(
            imageBase64,
            "base64"
          );

        /*
         * Vercel Blob upload
         */

        const extension =
          mimeType.includes("jpeg")
            ? "jpg"
            : "png";

        const fileName =
          `ynt-news-scene-${Date.now()}-${i + 1}.${extension}`;

        const blob =
          await put(
            `ynt-news/${fileName}`,
            imageBuffer,
            {
              access: "public",
              contentType: mimeType,
              token:
                process.env.BLOB_READ_WRITE_TOKEN,
            }
          );

        results.push({
          id:
            scene.id ||
            i + 1,

          success: true,

          imageUrl:
            blob.url,

          mimeType,

          prompt:
            imagePrompt,
        });

      } catch (imageError) {
        console.error(
          `IMAGE GENERATION ERROR SCENE ${i + 1}:`,
          imageError
        );

        results.push({
          id:
            scene.id ||
            i + 1,

          success: false,

          imageUrl: "",

          error:
            imageError?.message ||
            "Image generation failed",
        });
      }
    }

    const successCount =
      results.filter(
        item => item.success
      ).length;

    return res.status(200).json({
      success: successCount > 0,

      total:
        results.length,

      generated:
        successCount,

      scenes:
        results,
    });

  } catch (error) {
    console.error(
      "YNT GENERATE IMAGES ERROR:",
      error
    );

    return res.status(500).json({
      success: false,
      error:
        error?.message ||
        "Image generation error",
    });
  }
}
