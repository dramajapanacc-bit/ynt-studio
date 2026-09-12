import { handleUpload } from "@vercel/blob/client";

export const config = {
  api: {
    bodyParser: true,
  },
};

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({
      error: "POST method only",
    });
  }

  try {
    const body = req.body;

    const jsonResponse = await handleUpload({
      body,
      request: req,

      onBeforeGenerateToken: async (pathname) => {
        return {
          allowedContentTypes: ["video/*"],

          maximumSizeInBytes:
            500 * 1024 * 1024,

          addRandomSuffix: true,

          tokenPayload: JSON.stringify({
            purpose: "ynt-recap-video",
            pathname,
          }),
        };
      },

      onUploadCompleted: async ({
        blob,
      }) => {
        console.log(
          "YNT VIDEO UPLOADED:",
          blob.url
        );
      },
    });

    return res.status(200).json(jsonResponse);

  } catch (error) {
    console.error(
      "YNT BLOB UPLOAD ERROR:",
      error
    );

    return res.status(500).json({
      error:
        error.message ||
        "Blob upload error",
    });
  }
}
