import type { ServerConfig } from "../config.js";
import type { LoadedModels } from "../models/manager.js";
import type { LLMBackend } from "./types.js";
import { TransformersJsBackend } from "./transformersjs.js";
import { WandlerBackend } from "./wandler.js";

export type BackendName = LLMBackend["name"];

export function createLLMBackend(config: ServerConfig, models: LoadedModels): LLMBackend {
  switch (config.backend) {
    case "transformersjs":
      return new TransformersJsBackend(models);
    case "wandler":
      return new WandlerBackend(models);
    default:
      return new WandlerBackend(models);
  }
}

export type { LLMBackend } from "./types.js";
