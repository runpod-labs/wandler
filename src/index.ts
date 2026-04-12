import { loadConfig } from "./config.js";
import { loadModels } from "./models/manager.js";
import { createServer } from "./server.js";

const config = loadConfig();
const models = await loadModels(config);
const server = createServer(config, models);

server.listen(config.port, () => {
  console.log(`[wandler] http://localhost:${config.port}`);
});
