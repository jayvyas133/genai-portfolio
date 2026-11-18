# Portfolio gateway: the plain page at /, the React page at /react/, and
# the reverse proxy to the three APIs (web/nginx.conf, ADR 0021).
# Build context is web/.

# Stage 1: build the React app. Node exists only in this stage; the final
# image is plain nginx serving static files.
FROM node:24-alpine AS react
WORKDIR /build/react
COPY react/package.json react/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY site /build/site
COPY react ./
RUN npm run build

# Stage 2: serve.
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY site /usr/share/nginx/html
COPY --from=react /build/react/dist /usr/share/nginx/html/react
