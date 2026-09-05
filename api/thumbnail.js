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

  const apiKey = process.env.GEMINI_API_KEY;

  if (!apiKey) {
    return res.status(500).json({
      success: false,
      error: "GEMINI_API_KEY မတွေ့ပါ။",
    });
  }

  let tempPath = null;

  try {
    const contentType = req.headers["content-type"] || "";

    let imagePath = null;
    let imageMime = null;
    let theme = "Cinematic";
    let prompt = "";

    /*
     * WEBSITE က FormData နဲ့ ပို့လာရင်
     */
    if (contentType.includes("multipart/form-data")) {
      const form = formidable({
        multiples: false,
        keepExtensions: true,
        maxFileSize: 15 * 1024 * 1024,
      });

      const [fields, files] = await form.parse(req);

      theme =
        Array.isArray(fields.theme)
          ? fields.theme[0]
          : fields.theme || "Cinematic";

      prompt =
        Array.isArray(fields.prompt)
          ? fields.prompt[0]
          : fields.prompt || "";

      /*
       * file field name ဘာပဲဖြစ်ဖြစ်
       * ပထမဆုံး image file ကိုရှာမယ်
       */
      for (const key of Object.keys(files)) {
        const value = Array.isArray(files[key])
          ? files[key][0]
          : files[key];

        if (value && value.filepath) {
          imagePath = value.filepath;
          imageMime =
            value.mimetype || "image/jpeg";
          tempPath = value.filepath;
          break;
        }
      }
    }

    /*
     * JSON နဲ့ပို့လာတဲ့အခြေအနေကိုလည်း support လုပ်မယ်
     */
    else if (contentType.includes("application/json")) {
      let body = "";

      for await (const chunk of req) {
        body += chunk;
      }

      const data = JSON.parse(body || "{}");

      theme = data.theme || "Cinematic";
      prompt = data.prompt || "";
    }

    if (!imagePath && !prompt) {
      return res.status(400).json({
        success: false,
        error: "Thumbnail အတွက် ပုံ သို့မဟုတ် Prompt မတွေ့ပါ။",
      });
    }

    const ai = new GoogleGenAI({
      apiKey,
    });

    /*
     * AI Prompt
     */
    const thumbnailPrompt = `
Create a professional cinematic video thumbnail.

Theme:
${theme}

Additional instructions:
${prompt || "Transform the provided image into a professional cinematic thumbnail."}

Requirements:
- Landscape 16:9 composition
- Professional YouTube/Facebook movie thumbnail
- Dramatic cinematic lighting
- Strong contrast
- Premium visual quality
- Keep the main person/subject recognizable
- Make the composition visually powerful
- Make the main subject stand out
- Leave suitable clean space for title text
- Do not add random words
- Do not add a watermark
- Do not add a logo
- Do not change the main identity of the person
`;

    const contents = [];

    /*
     * User ရွေးထားတဲ့ ပုံကို AI ထဲထည့်မယ်
     */
    if (imagePath) {
      const base64Image = fs.readFileSync(
        imagePath,
        "base64"
      );

      contents.push({
        inlineData: {
          mimeType: imageMime,
          data: base64Image,
        },
      });
    }

    contents.push({
      text: thumbnailPrompt,
    });

    console.log("Starting thumbnail generation...");

    const response =
      await ai.models.generateContent({
        model: "gemini-2.5-flash-image",

        contents,

        config: {
          responseModalities: ["IMAGE"],

          responseFormat: {
            image: {
              aspectRatio: "16:9",
            },
          },
        },
      });

    let imageData = null;
    let outputMime = "image/png";

    const parts =
      response?.candidates?.[0]?.content?.parts || [];

    for (const part of parts) {
      if (part.inlineData) {
        imageData = part.inlineData.data;

        outputMime =
          part.inlineData.mimeType ||
          "image/png";

        break;
      }
    }

    if (!imageData) {
      console.error(
        "Gemini response:",
        JSON.stringify(response)
      );

      return res.status(500).json({
        success: false,
        error:
          "AI က Thumbnail ပုံကို ပြန်မပေးနိုင်ပါ။",
      });
    }

    console.log(
      "Thumbnail generated successfully."
    );

    return res.status(200).json({
      success: true,
      image:
        `data:${outputMime};base64,${imageData}`,
    });

  } catch (error) {
    console.error(
      "THUMBNAIL API ERROR:",
      error
    );

    return res.status(500).json({
      success: false,
      error:
        error?.message ||
        "Thumbnail ထုတ်ရာတွင် အမှားတစ်ခု ဖြစ်နေပါသည်။",
    });

  } finally {
    if (tempPath) {
      try {
        if (fs.existsSync(tempPath)) {
          fs.unlinkSync(tempPath);
        }
      } catch (e) {
        console.log(
          "Temporary file cleanup failed."
        );
      }
    }
  }
};
