import { type Demo } from "@/types/demos";

/**
 * Collection of all available demos
 * This is the single source of truth for demo information
 */
export const demos: Demo[] = [
	{
		id: "chat",
		title: "Chat",
		description: "Chat with an AI model running directly in your browser",
		path: "/d/chat",
		tags: ["chat", "streaming"],
		modelId: "onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX",
		thumbnail: "/images/demos/chat-thumbnail.png",
		featured: true,
	},
	// Add more demos here as needed
];

/**
 * Get a demo by its ID
 */
export function getDemoById(id: string): Demo | undefined {
	return demos.find(demo => demo.id === id);
}

/**
 * Get all featured demos
 */
export function getFeaturedDemos(): Demo[] {
	return demos.filter(demo => demo.featured);
}

/**
 * Get all demos with a specific tag
 */
export function getDemosByTag(tag: string): Demo[] {
	return demos.filter(demo => demo.tags.includes(tag));
}
