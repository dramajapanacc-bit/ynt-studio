export const config = {
  api: {
    bodyParser: true,
  },
};

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({
      success: false,
      error: "POST method only",
    });
  }

  try {
    const {
      fileName,
      mimeType,
      fileSize,
    } = req.body || {};

    if (!fileName || !mimeType || !fileSize) {
      return res.status(400).json({
        success: false,
        error: "fileName, mimeType and fileSize are required.",
      });
    }

    const apiKey = process.env.GEMINI_API_KEY;

    if (!apiKey) {
      return res.status(500).json({
        success: false,
        error: "GEMINI_API_KEY is not configured.",
      });
    }

    const response = await fetch(
      `https://generativelanguage.googleapis.com/upload/v1beta/files?key=${apiKey}`,
      {
        method: "POST",

        headers: {
          "X-Goog-Upload-Protocol": "resumable",
          "X-Goog-Upload-Command": "start",
          "X-Goog-Upload-Header-Content-Length":
            String(fileSize),
          "X-Goog-Upload-Header-Content-Type":
            mimeType,
          "Content-Type":
            "application/json",
        },

        body: JSON.stringify({
          file: {
            display_name: fileName,
          },
        }),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();

      return res.status(response.status).json({
        success: false,
        error: errorText || "Gemini upload session failed.",
      });
    }

    const uploadUrl =
      response.headers.get("x-goog-upload-url");

    if (!uploadUrl) {
      return res.status(500).json({
        success: false,
        error: "Gemini upload URL was not returned.",
      });
    }

    return res.status(200).json({
      success: true,
      uploadUrl,
    });

  } catch (error) {
    console.error(
      "UPLOAD INIT ERROR:",
      error
    );

    return res.status(500).json({
      success: false,
      error:
        error.message ||
        "Upload initialization failed.",
    });
  }
}
