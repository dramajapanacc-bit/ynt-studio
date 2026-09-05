const formidable = require("formidable");
const fs = require("fs");
const { GoogleGenAI } = require("@google/genai");

module.exports.config = {
  api: {
    bodyParser: false,
  },
};

module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({
      success: false,
      error: "POST method only",
    });
  }

  try {
    const apiKey = process.env.GEMINI_API_KEY;

    if (!apiKey) {
      return res.status(500).json({
        success: false,
        error: "GEMINI_API_KEY မတွေ့ပါ။",
      });
    }

    const form = formidable({
      multiples: false,
      keepExtensions: true,
      maxFileSize: 15 * 1024 * 1024,
    });

    const [fields, files] = await form.parse(req);

    const theme = Array.isArray(fields.theme)
      ? fields.theme[0]
      : fields.theme || "Cinematic";

    let uploadedImage = null;

    for (const key of Object.keys(files)) {
      const value = Array.isArray(files[key])
        ? files[key][0]
        : files[key];

      if (value && value.filepath) {
        uploadedImage = value;
        break;
      }
    }

    if (!uploadedImage) {
      return res.status(400).json({
        success: false,
        error: "ပုံဖိုင် မတွေ့ပါ။",
      });
    }

    const imageData = fs.readFileSync(
      uploadedImage.filepath,
      "base64"
    );

    const mimeType =
      uploadedImage.mimetype || "image/jpeg";

    const ai = new GoogleGenAI({
      apiKey: apiKey,
    });

    const prompt = `
Create a professional cinematic video thumbnail
from the provided reference image.

Theme: ${theme}

Requirements:
- 16:9 landscape thumbnail
- Professional movie thumbnail style
- Cinematic lighting
- Dramatic composition
- High contrast
- Premium visual quality
- Keep the main people recognizable
- Make the subjects stand out
- Beautiful cinematic color grading
- Suitable for YouTube and Facebook
- No watermark
- No logo
- No random text
- Do not add unnecessary objects
`;

    console.log("Starting Gemini image generation...");

    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash-image",

      contents: [
        {
          text: prompt,
        },
        {
          inlineData: {
            mimeType: mimeType,
            data: imageData,
          },
        },
      ],

      config: {
        responseFormat: {
          image: {
            aspectRatio: "16:9",
          },
        },
      },
    });

    const parts =
      response?.candidates?.[0]?.content?.parts || [];

    let generatedImage = null;
    let generatedMime = "image/png";

    for (const part of parts) {
      if (part.inlineData?.data) {
        generatedImage = part.inlineData.data;

        generatedMime =
          part.inlineData.mimeType || "image/png";

        break;
      }
    }

    if (!generatedImage) {
      console.error(
        "Gemini returned no image:",
        JSON.stringify(response)
      );

      return res.status(500).json({
        success: false,
        error: "Gemini က ပုံကို ပြန်မပေးနိုင်ပါ။",
      });
    }

    try {
      fs.unlinkSync(uploadedImage.filepath);
    } catch (e) {}

    return res.status(200).json({
      success: true,
      image:
        `data:${generatedMime};base64,${generatedImage}`,
    });

  } catch (error) {
    console.error(
      "THUMBNAIL ERROR:",
      error
    );

    return res.status(500).json({
      success: false,
      error:
        error?.message ||
        "Thumbnail API Error",
    });
  }
};
