"use client";

// Import global styles
import "@/styles/chat-markdown.css";

import { notFound } from "next/navigation";
import React, { use, useEffect, useState } from "react";

import { ChatInterface } from "@/components/chat/chat-interface";
import { ModelHeader } from "@/components/hub/model-header";
import {
	AVAILABLE_MODELS,
	DEFAULT_MODEL_ID,
	getModelById,
	getModelsByCapability,
	ModelInfo,
} from "@/lib/models";

// Define available demos
const AVAILABLE_DEMOS = ["chat"];
const AVAILABLE_MODEL_PAGES = AVAILABLE_MODELS.map(model => model.id);

interface HubDemoPageProps {
	params: Promise<{
		slug: string[];
	}>;
}

export default function HubDemoPage({ params }: HubDemoPageProps): React.ReactElement {
	// Unwrap the params Promise using React.use()
	const unwrappedParams = use(params);

	// Get the first segment of the slug as the demo type or model ID
	const firstSlug = unwrappedParams.slug[0];

	// Determine if this is a demo page or a model page
	const isDemoPage = AVAILABLE_DEMOS.includes(firstSlug);
	const isModelPage = AVAILABLE_MODEL_PAGES.includes(firstSlug);

	// If neither a valid demo nor a valid model, 404
	if (!isDemoPage && !isModelPage) {
		notFound();
	}

	// Set the initial model ID based on the URL
	const initialModelId = isModelPage ? firstSlug : DEFAULT_MODEL_ID;
	const [currentModelId, setCurrentModelId] = useState(initialModelId);
	const [currentModel, setCurrentModel] = useState<ModelInfo | undefined>(
		getModelById(initialModelId)
	);

	// Update the model when the ID changes
	useEffect(() => {
		setCurrentModel(getModelById(currentModelId));
	}, [currentModelId]);

	// If we don't have a valid model, 404
	if (!currentModel) {
		notFound();
	}

	// Get models that can be used for chat
	const chatModels = getModelsByCapability("text");

	// Determine the page title
	const pageTitle = isDemoPage ? "Chat" : currentModel.humanReadableName;

	return (
		<div className="space-y-8">
			{/* Page title and model header */}
			<div>
				<h1 className="text-4xl md:text-6xl font-bold tracking-tighter mb-6">{pageTitle}</h1>

				<ModelHeader
					model={currentModel}
					showSelector={isDemoPage} // Only show selector on the demo page, not model-specific pages
					onModelChange={setCurrentModelId}
					availableModels={chatModels}
				/>
			</div>

			{/* Render the chat interface */}
			<div className="cyberpunk-theme bg-black/60 backdrop-blur-sm border border-cyan-500/20 rounded-lg p-4 md:p-6 w-full flex flex-col relative">
				<ChatInterface modelId={currentModelId} />
			</div>
		</div>
	);
}
