const { GoogleGenAI } = require("@google/genai");

module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({
      success: false,
      error: "POST method only"
    });
  }

  try {
    const apiKey = process.env.GEMINI_API_KEY;

    if (!apiKey) {
      return res.status(500).json({
        success: false,
        error: "GEMINI_API_KEY မတွေ့ပါ။"
      });
    }

    const { prompt, theme } = req.body || {};

    if (!prompt && !theme) {
      return res.status(400).json({
        success: false,
        error: "Thumbnail အကြောင်းအရာ ထည့်ပေးပါ။"
      });
    }

    const ai = new GoogleGenAI({
      apiKey
    });

    const thumbnailPrompt = `
Create a professional YouTube/Facebook video thumbnail.

Theme:
${theme || "cinematic"}

Main idea:
${prompt || "Create an eye-catching cinematic thumbnail"}

Requirements:
- 16:9 landscape composition
- High quality
- Cinematic lighting
- Strong contrast
- Dramatic composition
- Professional movie thumbnail style
- Main subject clearly visible
- Leave some clean space for title text
- No watermark
- No logo
- No random text
- Visually attractive and attention grabbing
`;

    console.log("Generating thumbnail...");

    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash-image",
      contents: thumbnailPrompt,
      config: {
        responseModalities: ["IMAGE"]
      }
    });

    let imageData = null;
    let mimeType = "image/png";

    for (const part of response.candidates?.[0]?.content?.parts || []) {
      if (part.inlineData) {
        imageData = part.inlineData.data;
        mimeType =
          part.inlineData.mimeType || "image/png";
        break;
      }
    }

    if (!imageData) {
      return res.status(500).json({
        success: false,
        error: "AI က Thumbnail ပုံ မထုတ်ပေးနိုင်ပါ။"
      });
    }

    console.log("Thumbnail generated successfully.");

    return res.status(200).json({
      success: true,
      image: `data:${mimeType};base64,${imageData}`
    });

  } catch (error) {
    console.error("THUMBNAIL ERROR:", error);

    return res.status(500).json({
      success: false,
      error:
        error?.message ||
        "Thumbnail ထုတ်ရာတွင် အမှားတစ်ခု ဖြစ်နေပါသည်။"
    });
  }
};
