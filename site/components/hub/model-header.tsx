"use client";

import { Github } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

import { ModelInfo } from "@/lib/models";

interface ModelHeaderProps {
	model: ModelInfo;
	showSelector?: boolean;
	onModelChange?: (modelId: string) => void;
	availableModels?: ModelInfo[];
}

export function ModelHeader({
	model,
	showSelector = false,
	onModelChange,
	availableModels = [],
}: ModelHeaderProps): React.ReactElement {
	const [isSelectOpen, setIsSelectOpen] = useState(false);

	return (
		<div className="flex flex-col space-y-4">
			{/* Model selector - only shown on the chat page */}
			{showSelector && (
				<div className="w-full mb-4">
					<div className="relative">
						<button
							onClick={() => setIsSelectOpen(!isSelectOpen)}
							className="w-full px-4 py-3 text-base bg-black/60 border border-cyan-500/50 text-cyan-400 rounded-md hover:border-cyan-500 hover:bg-black/80 font-medium flex justify-between items-center transition-all"
						>
							<span className="text-lg">{model.humanReadableName}</span>
							<span className="text-cyan-400">▼</span>
						</button>

						{isSelectOpen && (
							<div className="absolute top-full left-0 right-0 mt-1 bg-black/90 backdrop-blur-sm border border-cyan-500/30 rounded-md shadow-lg z-50">
								<div className="p-2">
									{availableModels.map(availableModel => (
										<button
											key={availableModel.id}
											className={`w-full text-left px-4 py-3 rounded-md text-lg ${
												availableModel.id === model.id
													? "bg-cyan-500/20 text-cyan-300"
													: "text-white hover:bg-cyan-500/10 hover:text-cyan-300"
											} transition-colors`}
											onClick={() => {
												onModelChange?.(availableModel.id);
												setIsSelectOpen(false);
											}}
										>
											{availableModel.humanReadableName}
										</button>
									))}
								</div>
							</div>
						)}
					</div>
				</div>
			)}

			<div className="flex flex-wrap items-center gap-3">
				{/* HuggingFace link */}
				<Link
					href={`https://huggingface.co/${model.huggingfaceRepo}`}
					target="_blank"
					rel="noopener noreferrer"
					className="flex items-center gap-1 px-3 py-2 text-sm bg-black/30 border border-yellow-500/50 text-white rounded-md hover:bg-black/50 hover:border-yellow-500 transition-colors font-medium"
				>
					<span>🤗</span> huggingface
				</Link>

				{/* GitHub link */}
				<Link
					href="https://github.com/wandlerai/wandler"
					target="_blank"
					rel="noopener noreferrer"
					className="flex items-center gap-1 px-3 py-2 text-sm bg-gray-700 text-white rounded-md hover:bg-gray-600 transition-colors font-medium"
				>
					<Github size={16} />
					<span>GitHub</span>
				</Link>

				{/* Tags moved to the same row */}
				{model.tags.map(tag => (
					<span
						key={tag}
						className="px-3 py-2 text-sm bg-primary/10 text-primary rounded-md font-medium"
					>
						{tag}
					</span>
				))}
			</div>

			{/* Description */}
			<p className="text-xl text-muted-foreground max-w-3xl">{model.description}</p>
		</div>
	);
}
