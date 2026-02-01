# ---- deps ----
    FROM node:20-alpine AS deps
    WORKDIR /app
    COPY package*.json ./
    RUN npm ci
    
    # ---- build ----
    FROM node:20-alpine AS builder
    WORKDIR /app
    ENV NEXT_TELEMETRY_DISABLED=1
    COPY --from=deps /app/node_modules ./node_modules
    COPY . .
    RUN npm run build
    
    # ---- runner ----
    FROM node:20-alpine AS runner
    WORKDIR /app
    ENV NODE_ENV=production
    ENV NEXT_TELEMETRY_DISABLED=1
    
    # Standalone server + minimal files
    COPY --from=builder /app/.next/standalone ./
    COPY --from=builder /app/.next/static ./.next/static
    COPY --from=builder /app/public ./public
    
    EXPOSE 3000
    # Standalone output creates server.js at root
    CMD ["node", "server.js"]
    