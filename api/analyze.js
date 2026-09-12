import { GoogleGenAI } from "@google/genai";

export const config = {
  api: {
    bodyParser: false,
  },
};

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
});

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({
      error: "POST method only",
    });
  }

  try {
    const contentType = req.headers["content-type"] || "";

    if (!contentType.includes("multipart/form-data")) {
      return res.status(400).json({
        error: "Video file is required.",
      });
    }

    const chunks = [];

    for await (const chunk of req) {
      chunks.push(chunk);
    }

    const body = Buffer.concat(chunks);

    /*
      multipart/form-data ကို ဒီ endpoint မှာ
      လက်တွေ့ production အတွက် formidable / busboy
      သုံးပြီး parse လုပ်သင့်ပါတယ်။
    */

    return res.status(501).json({
      error:
        "Multipart parser မထည့်ရသေးပါ။ အောက်က production version ကိုသုံးပါ။",
    });

  } catch (error) {
    console.error(error);

    return res.status(500).json({
      error: "P2 backend error",
      message: error.message,
    });
  }
}
