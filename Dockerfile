FROM node:20-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg python3 python3-pip ca-certificates fonts-noto-core fonts-sil-padauk fontconfig \
 && pip3 install --no-cache-dir --break-system-packages -U yt-dlp edge-tts \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY package*.json ./
RUN npm install --omit=dev
COPY . .
ENV PORT=10000
EXPOSE 10000
CMD ["node", "server.js"]
