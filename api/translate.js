const { GoogleGenAI } = require("@google/genai");
const formidable = require("formidable");
const fs = require("fs");

module.exports = async (req, res) => {
  if (req.method !== "POST") {
    return res.status(405).json({
      error: "POST method only"
    });
  }

  try {
    const form = formidable({
      multiples: false,
      keepExtensions: true,
      maxFileSize: 4500000
    });

    const [fields, files] = await form.parse(req);

    const uploadedFile = files.file?.[0];

    if (!uploadedFile) {
      return res.status(400).json({
        error: "Video file မတွေ့ပါ။"
      });
    }

    const apiKey = process.env.GEMINI_API_KEY;

    if (!apiKey) {
      return res.status(500).json({
        error: "GEMINI_API_KEY မတွေ့ပါ။ Vercel Environment Variables ကိုစစ်ပါ။"
      });
    }

    const ai = new GoogleGenAI({
      apiKey
    });

    console.log("Uploading file to Gemini...");

    const uploaded = await ai.files.upload({
      file: uploadedFile.filepath,
      config: {
        mimeType: uploadedFile.mimetype || "video/mp4"
      }
    });

    console.log("Gemini file:", uploaded.name);

    let fileInfo = await ai.files.get({
      name: uploaded.name
    });

    let attempts = 0;

    while (fileInfo.state === "PROCESSING" && attempts < 60) {
      await new Promise(resolve => setTimeout(resolve, 3000));

      fileInfo = await ai.files.get({
        name: uploaded.name
      });

      attempts++;
      console.log("Processing:", attempts, fileInfo.state);
    }

    if (fileInfo.state !== "ACTIVE") {
      return res.status(500).json({
        error: "Video ကို AI က processing ပြီးအောင် မလုပ်နိုင်ပါ။",
        state: fileInfo.state
      });
    }

    const sourceLanguage =
      fields.sourceLanguage?.[0] || "auto";

    const prompt = `
ဒီ Video ထဲက ပြောဆိုထားတဲ့ စကားတွေကို နားထောင်ပြီး
မြန်မာဘာသာသို့ သဘာဝကျကျ ဘာသာပြန်ပေးပါ။

Source Language: ${sourceLanguage}

Output ကို SRT subtitle format သာ ထုတ်ပါ။

စည်းမျဉ်းများ:
1. SRT format အတိုင်းရေးပါ။
2. Subtitle number ပါရမည်။
3. Start time --> End time ပါရမည်။
4. မြန်မာစာသာ အသုံးပြုပါ။
5. Dialogue အားလုံးကို တတ်နိုင်သမျှ မကျန်အောင် ထည့်ပါ။
6. Markdown မသုံးပါနှင့်။
7. Code block မသုံးပါနှင့်။
8. ရှင်းလင်းချက် မထည့်ပါနှင့်။

ဥပမာ:

1
00:00:01,000 --> 00:00:04,000
မင်္ဂလာပါ။ ဒီနေ့ ဘယ်လိုနေလဲ။

2
00:00:04,000 --> 00:00:07,000
ကျွန်တော် ကောင်းပါတယ်။
`;

    console.log("Generating Myanmar SRT...");

    const result = await ai.models.generateContent({
      model: "gemini-3.7-flash",
      contents: [
        {
          fileData: {
            fileUri: fileInfo.uri,
            mimeType: fileInfo.mimeType
          }
        },
        {
          text: prompt
        }
      ]
    });

    let srt = result.text || "";

    srt = srt
      .replace(/```srt/gi, "")
      .replace(/```/g, "")
      .trim();

    if (!srt) {
      return res.status(500).json({
        error: "AI က SRT မထုတ်ပေးနိုင်ပါ။"
      });
    }

    try {
      fs.unlinkSync(uploadedFile.filepath);
    } catch (e) {
      console.log("Temp file cleanup skipped");
    }

    return res.status(200).json({
      success: true,
      srt
    });

  } catch (error) {
    console.error("TRANSLATE ERROR:", error);

    return res.status(500).json({
      error: error.message || "AI ဘာသာပြန်ရာတွင် အမှားတစ်ခု ဖြစ်နေပါသည်။",
      details: process.env.NODE_ENV === "development"
        ? String(error.stack || error)
        : undefined
    });
  }
};
