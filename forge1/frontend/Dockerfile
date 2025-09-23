# forge1/frontend/Dockerfile
# Multi-stage Docker build for Forge 1 Frontend

# Build stage
FROM node:18-alpine as builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies with exact versions
RUN npm ci --only=production --no-audit --no-fund

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine as production

# Install curl for health checks
RUN apk add --no-cache curl

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy built application from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Create non-root user
RUN addgroup -g 1001 -S forge1 && \
    adduser -S forge1 -u 1001 -G forge1

# Set ownership
RUN chown -R forge1:forge1 /usr/share/nginx/html && \
    chown -R forge1:forge1 /var/cache/nginx && \
    chown -R forge1:forge1 /var/log/nginx && \
    chown -R forge1:forge1 /etc/nginx/conf.d

# Switch to non-root user
USER forge1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000 || exit 1

# Expose port
EXPOSE 3000

# Start nginx
CMD ["nginx", "-g", "daemon off;"]