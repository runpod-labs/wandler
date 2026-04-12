declare module "wandler" {
	export interface Message {
		role: "system" | "user" | "assistant";
		content: string;
	}

	export interface ProgressInfo {
		status: "progress";
		loaded: number;
		total: number;
	}

	export interface LoadModelOptions {
		onProgress?: (info: ProgressInfo) => void;
	}

	export interface StreamChunk {
		type: string;
		text?: string;
	}

	export interface ModelCapabilities {
		textGeneration: boolean;
		textClassification?: boolean;
		imageGeneration?: boolean;
		audioProcessing?: boolean;
		vision?: boolean;
	}

	export interface BaseModel {
		id: string;
		provider?: string;
		capabilities: ModelCapabilities;
		[key: string]: unknown;
	}

	export interface StreamTextOptions {
		model: BaseModel;
		messages: Message[];
		abortSignal?: AbortSignal;
		onChunk?: (chunk: StreamChunk) => void;
	}

	export interface StreamTextResult {
		result: Promise<unknown>;
		textStream?: AsyncIterable<string>;
		fullStream?: AsyncIterable<unknown>;
	}

	export function loadModel(modelId: string, options?: LoadModelOptions): Promise<BaseModel>;
	export function streamText(options: StreamTextOptions): Promise<StreamTextResult>;
}
