## README

1. download nodejs from using [nvm](https://nodejs.org/en/download/package-manager/current)

then configure mosquitto enable websocket protocol
add below to end of /etc/mosquitto/mosquitto.conf

```bash
# this will listen for mqtt on tcp
listener 1883

# this will expect websockets connections
listener 1884
protocol websockets
allow_anonymous true
```

2. see chess_test() in chess_helper.py to publish svg content via mqtt  

3. run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.