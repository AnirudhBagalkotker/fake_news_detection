FROM node:20

WORKDIR /app

COPY frontend/package.json /app/package.json
RUN npm install

COPY frontend /app

RUN npm run build

EXPOSE 3000

# CMD ["npx", "serve", "-s", "build", "-l", "3000"]
