export interface ModelInfo {
	id: string;
	name: string;
	humanReadableName: string;
	description: string;
	capabilities: string[];
	tags: string[];
	huggingfaceRepo: string;
	size: string;
	quantization?: string;
}

export const AVAILABLE_MODELS: ModelInfo[] = [
	{
		id: "deepseek-r1-distill-qwen-1.5b",
		name: "DeepSeek-R1-Distill-Qwen-1.5B",
		humanReadableName: "DeepSeek r1 (distill Qwen 1.5B)",
		description: "A powerful reasoning model, distilled from Qwen2.5-Math-1.5B",
		capabilities: ["text", "reasoning"],
		tags: ["text", "llm", "reasoning"],
		huggingfaceRepo: "onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX",
		size: "1.5B",
		quantization: "GGUF",
	},
	{
		id: "qwen2.5-coder-0.5b",
		name: "Qwen2.5-Coder-0.5B",
		humanReadableName: "Qwen 2.5 Coder (0.5B)",
		description: "Specialized for code generation, code reasoning and code fixing.",
		capabilities: ["text", "code"],
		tags: ["text", "llm", "code"],
		huggingfaceRepo: "onnx-community/Qwen2.5-Coder-0.5B-Instruct",
		size: "0.5B",
		quantization: "GGUF",
	},
];

export function getModelById(id: string): ModelInfo | undefined {
	return AVAILABLE_MODELS.find(model => model.id === id);
}

export function getModelsByCapability(capability: string): ModelInfo[] {
	return AVAILABLE_MODELS.filter(model => model.capabilities.includes(capability));
}

export const DEFAULT_MODEL_ID = "deepseek-r1-distill-qwen-1.5b";
