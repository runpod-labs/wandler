"use client";

import { ExtendedMessage, useChat } from "@wandler/react";
import { ChevronDown, ChevronUp, Send, Square, Trash2 } from "lucide-react";
import { marked } from "marked";
import React, { useEffect, useRef, useState } from "react";
import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom";
import type { BaseModel, ProgressInfo } from "wandler";
import { loadModel } from "wandler";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";

// Available models for the selector
const AVAILABLE_MODELS = [
	{
		id: "qwen2.5-coder-0.5b",
		value: "onnx-community/Qwen2.5-Coder-0.5B-Instruct",
		label: "Qwen 0.5B (Coder)",
	},
	{
		id: "deepseek-r1-distill-qwen-1.5b",
		value: "onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX",
		label: "DeepSeek 1.5B",
	},
];

// Function to render markdown content
const renderMarkdown = (content: string) => {
	if (!content) return "";

	try {
		return marked.parse(content);
	} catch (error) {
		console.error("Error parsing markdown:", error);
		return content;
	}
};

// ScrollToBottom component that uses the StickToBottom context
function ScrollToBottom(): React.ReactElement | null {
	const { isAtBottom, scrollToBottom } = useStickToBottomContext();

	if (isAtBottom) return null;

	return (
		<div className="absolute bottom-24 left-1/2 -translate-x-1/2 z-10">
			<button
				onClick={() => scrollToBottom("smooth")}
				className="bg-primary/90 text-black p-2 rounded-full shadow-lg hover:bg-primary transition-colors"
				aria-label="Scroll to bottom"
			>
				<ChevronDown size={24} />
			</button>
		</div>
	);
}

// Define message part interface
interface MessagePart {
	type: string;
	content?: string;
	reasoning?: string;
}

interface ReasoningMessagePart extends MessagePart {
	type: "reasoning";
	reasoning: string;
}

// Reasoning bubble component
interface ReasoningBubbleProps {
	text: string;
	isStreaming?: boolean;
}

function ReasoningBubble({ text, isStreaming }: ReasoningBubbleProps): React.ReactElement {
	const [collapsed, setCollapsed] = useState(false);
	const contentRef = useRef<HTMLDivElement>(null);
	const [contentHeight, setContentHeight] = useState<number | null>(null);

	// Measure the content height when it first renders or when the text changes
	useEffect(() => {
		if (contentRef.current && !collapsed) {
			setContentHeight(contentRef.current.scrollHeight);
		}
	}, [text, collapsed]);

	return (
		<div className="bg-black/60 border border-red-500/30 rounded p-2 mb-2 relative">
			<div className="flex justify-between items-center">
				<div className="text-red-400 text-xs font-bold mb-1">Reasoning:</div>
				<div className="flex gap-2">
					<button
						onClick={() => setCollapsed(!collapsed)}
						className="text-xs text-gray-400 hover:text-white flex items-center gap-1"
						aria-label={collapsed ? "Expand reasoning" : "Collapse reasoning"}
					>
						{collapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
					</button>
				</div>
			</div>
			<div
				className="text-sm overflow-hidden transition-all duration-200"
				style={{ height: collapsed ? 0 : contentHeight ? `${contentHeight}px` : "auto" }}
			>
				<div ref={contentRef} className="whitespace-pre-wrap">
					{text}
					{isStreaming && (
						<span className="inline-block w-2 h-5 ml-1 bg-primary animate-blink"></span>
					)}
				</div>
			</div>
		</div>
	);
}

interface ChatInterfaceProps {
	className?: string;
	modelId?: string;
}

// ChatBox component that uses the StickToBottom context
function ChatBox({
	input,
	handleInputChange,
	handleSubmit,
	status,
	stop,
	clearMessages,
	modelLoaded = false,
}: {
	input: string;
	handleInputChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
	handleSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
	status: string;
	stop: () => void;
	clearMessages: () => void;
	modelLoaded?: boolean;
}): React.ReactElement {
	const { scrollToBottom } = useStickToBottomContext();
	const inputRef = useRef<HTMLInputElement>(null);

	// Focus input on mount and when model changes
	useEffect(() => {
		if (inputRef.current && modelLoaded) {
			inputRef.current.focus();
		}
	}, [modelLoaded]);

	// Custom submit handler that scrolls to bottom
	const onSubmit = (e: React.FormEvent<HTMLFormElement>) => {
		handleSubmit(e);
		// Scroll to bottom after submitting
		setTimeout(() => scrollToBottom("smooth"), 100);
	};

	return (
		<div className="mt-auto pt-4 pb-2">
			<form onSubmit={onSubmit} className="flex items-center gap-3">
				<input
					ref={inputRef}
					type="text"
					value={input}
					onChange={handleInputChange}
					placeholder="Ask anything..."
					className="flex-1 bg-black border-2 border-primary/30 focus:border-primary/70 rounded-md px-4 py-2 text-white placeholder-gray-500 focus:outline-none"
					disabled={status === "streaming"}
				/>
				{status === "streaming" ? (
					<Button
						type="button"
						onClick={stop}
						variant="destructive"
						className="bg-red-900 hover:bg-red-800 text-white"
					>
						<Square className="w-4 h-4" />
					</Button>
				) : (
					<Button
						type="submit"
						className="bg-primary hover:bg-primary/90 text-black font-medium"
						disabled={!input.trim()}
					>
						<Send className="w-4 h-4" />
					</Button>
				)}
				<Button
					type="button"
					onClick={clearMessages}
					variant="outline"
					className="border-primary/50 text-primary hover:bg-primary/10"
				>
					<Trash2 className="w-4 h-4" />
				</Button>
			</form>
		</div>
	);
}

export function ChatInterface({ className = "", modelId }: ChatInterfaceProps): React.ReactElement {
	// Find the model configuration based on the modelId prop
	const initialModel = modelId
		? AVAILABLE_MODELS.find(m => m.id === modelId) || AVAILABLE_MODELS[1]
		: AVAILABLE_MODELS[1];

	const [model, setModel] = useState<BaseModel | null>(null);
	const [selectedModel, setSelectedModel] = useState(initialModel.value);
	const [isModelLoading, setIsModelLoading] = useState(false);
	const [loadingProgress, setLoadingProgress] = useState({ loaded: 0, total: 0 });
	const [error, setError] = useState<string | null>(null);
	const [status, setStatus] = useState("Model not loaded");
	const [isLoading, setIsLoading] = useState(false);

	// Update selected model when modelId prop changes
	useEffect(() => {
		if (modelId) {
			const newModel = AVAILABLE_MODELS.find(m => m.id === modelId);
			if (newModel && newModel.value !== selectedModel) {
				setSelectedModel(newModel.value);
				// Reset the model to trigger a reload
				setModel(null);
			}
		}
	}, [modelId, selectedModel]);

	// Create a placeholder model for when the real model is not loaded
	const placeholderModel = model || {
		id: "placeholder-model",
		name: "Model Not Loaded",
		capabilities: {
			textGeneration: false,
		},
	};

	// Use the chat hook with a stable model reference
	const [chatModel, setChatModel] = useState<BaseModel>(placeholderModel as BaseModel);

	// Keep a stable reference to the model for the useChat hook
	useEffect(() => {
		if (model) {
			// Only update the chat model once we have a real model
			setChatModel(model);
		}
	}, [model]);

	const {
		messages,
		input,
		handleSubmit,
		handleInputChange,
		stop,
		clearChat: clearMessages,
		status: chatStatus,
		error: chatError,
	} = useChat({
		model: chatModel,
	});

	// For debugging
	useEffect(() => {
		console.log("Current model status:", chatStatus);
	}, [chatStatus]);

	// For debugging chat errors
	useEffect(() => {
		if (chatError) {
			console.error("Chat error:", chatError);
		}
	}, [chatError]);

	// Debug messages changes
	useEffect(() => {
		console.log("Messages updated:", messages);
	}, [messages]);

	// Calculate the last assistant message
	const lastAssistantMessage = [...messages]
		.reverse()
		.find(message => message.role === "assistant");

	const handleLoadModel = async () => {
		try {
			setStatus("Loading model...");
			setIsLoading(true);
			setLoadingProgress({ loaded: 0, total: 0 });

			const model = await loadModel(selectedModel, {
				onProgress: (info: ProgressInfo) => {
					if (info.status === "progress") {
						setLoadingProgress({
							loaded: info.loaded,
							total: info.total,
						});
					}
				},
			});

			setModel(model);
			setStatus(`Model loaded (${model.provider === "worker" ? "Worker" : "Main"} thread)`);
		} catch (error) {
			setStatus(`Error loading model: ${error instanceof Error ? error.message : String(error)}`);
		} finally {
			setIsLoading(false);
		}
	};

	const handleUnloadModel = () => {
		if (model) {
			setModel(null);
		}
	};

	const progressPercentage =
		loadingProgress.total > 0
			? Math.round((loadingProgress.loaded / loadingProgress.total) * 100)
			: 0;

	const formatBytes = (bytes: number) => {
		if (bytes === 0) return "0 Bytes";
		const k = 1024;
		const sizes = ["Bytes", "KB", "MB", "GB"];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
	};

	// Function to get reasoning content from a message
	const getReasoningContent = (message: ExtendedMessage): string => {
		const typedMessage = message as unknown as { parts?: MessagePart[] };
		const reasoningPart = typedMessage.parts?.find(part => part.type === "reasoning") as
			| ReasoningMessagePart
			| undefined;
		return reasoningPart?.reasoning || "";
	};

	// Function to check if a message has reasoning
	const hasReasoning = (message: ExtendedMessage): boolean => {
		const typedMessage = message as unknown as { parts?: MessagePart[] };
		return typedMessage.parts?.some(part => part.type === "reasoning") || false;
	};

	return (
		<div className={`flex flex-col h-[70vh] ${className}`}>
			{/* Chat interface or model loader */}
			<div className="flex-grow flex flex-col h-full">
				{!model ? (
					<div className="flex flex-col items-center justify-center h-full space-y-8">
						{isLoading ? (
							<div className="w-full max-w-md space-y-4">
								<Progress value={progressPercentage} className="h-4 bg-black" />
								<div className="flex justify-between text-sm text-muted-foreground">
									<span>{formatBytes(loadingProgress.loaded)}</span>
									<span>{progressPercentage}%</span>
									<span>{formatBytes(loadingProgress.total)}</span>
								</div>
							</div>
						) : (
							<Button
								onClick={handleLoadModel}
								size="lg"
								className="relative group overflow-hidden bg-black hover:bg-black/90 text-white font-bold text-xl py-8 px-16 rounded-lg shadow-lg transform transition-all hover:scale-105 border-2 border-transparent hover:border-primary/50"
								disabled={isLoading}
							>
								<span className="relative z-10 flex items-center gap-2 text-2xl">LOAD MODEL</span>
								<div className="absolute inset-0 bg-gradient-to-r from-[#9c27b0] via-[#673ab7] to-[#3f51b5] opacity-20 group-hover:opacity-30 transition-opacity"></div>
							</Button>
						)}

						{error && (
							<div className="mt-4 p-4 bg-red-900/20 border border-red-500 rounded-md text-red-300 max-w-md">
								<p className="font-bold mb-2">Error Loading Model:</p>
								<p className="text-sm">{error}</p>
							</div>
						)}
					</div>
				) : (
					<>
						<div className="flex-1 overflow-hidden relative">
							<StickToBottom
								className="h-full overflow-y-auto pr-2 relative flex flex-col"
								resize="smooth"
								initial="smooth"
							>
								{/* Scroll to bottom button - positioned absolutely */}
								<ScrollToBottom />

								<StickToBottom.Content className="flex-1 flex flex-col">
									{messages.length === 0 ? (
										<div className="flex items-center justify-center h-full">
											{/* Empty state - no text needed */}
										</div>
									) : (
										<div className="space-y-4">
											{messages.map((message, index) => {
												// Skip system messages in the UI
												if (message.role === "system") return null;

												const isUser = message.role === "user";
												const isStreaming =
													chatStatus === "streaming" && message.id === lastAssistantMessage?.id;
												const reasoningContent = !isUser ? getReasoningContent(message) : "";

												return (
													<div
														key={index}
														className={`flex ${isUser ? "justify-end" : "justify-start"}`}
													>
														<div
															className={`max-w-[80%] rounded-lg p-4 ${
																isUser ? "bg-primary/20 text-white" : "bg-black text-white"
															}`}
														>
															{!isUser && hasReasoning(message) && reasoningContent && (
																<ReasoningBubble
																	text={reasoningContent}
																	isStreaming={isStreaming && !message.isComplete}
																/>
															)}

															{isUser ? (
																<p>{message.content}</p>
															) : (
																<div
																	className="prose prose-invert prose-sm max-w-none chat-markdown"
																	dangerouslySetInnerHTML={{
																		__html: renderMarkdown(message.content || ""),
																	}}
																/>
															)}
														</div>
													</div>
												);
											})}
										</div>
									)}
								</StickToBottom.Content>

								{/* Chat input at the bottom of the StickToBottom component */}
								<ChatBox
									input={input}
									handleInputChange={handleInputChange}
									handleSubmit={handleSubmit}
									status={chatStatus}
									stop={stop}
									clearMessages={clearMessages}
									modelLoaded={!!model}
								/>
							</StickToBottom>
						</div>
					</>
				)}
			</div>
		</div>
	);
}
