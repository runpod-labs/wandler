"use client";

import { AudioWaveform,ImageIcon, Search, SlidersHorizontal, TextIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export function ModelsSection() {
	return (
		<section id="models" className="py-16 relative">
			<div className="absolute inset-0 dot-pattern opacity-10"></div>
			<div className="container mx-auto px-4 relative z-10">
				<div className="flex flex-col gap-8">
					<div className="flex flex-wrap items-center justify-between gap-4">
						<h2 className="text-3xl font-bold">Explore Models</h2>
						<div className="flex gap-4">
							<div className="relative">
								<Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
								<Input
									placeholder="Search models..."
									className="pl-9 bg-secondary border-primary/20 w-[200px]"
								/>
							</div>
							<Button variant="outline" className="border-primary/20">
								<SlidersHorizontal className="w-4 h-4 mr-2" />
								Filters
							</Button>
						</div>
					</div>

					<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
						{[
							{
								name: "GPT-2 Small",
								type: "text",
								icon: TextIcon,
								provider: "HuggingFace",
								description: "Efficient text generation model",
								image: "/placeholder.svg?height=200&width=400",
							},
							{
								name: "ResNet-50",
								type: "image",
								icon: ImageIcon,
								provider: "ONNX",
								description: "Image classification model",
								image: "/placeholder.svg?height=200&width=400",
							},
							{
								name: "Whisper Small",
								type: "audio",
								icon: AudioWaveform,
								provider: "OpenAI",
								description: "Speech recognition model",
								image: "/placeholder.svg?height=200&width=400",
							},
							{
								name: "YOLO v8",
								type: "image",
								icon: ImageIcon,
								provider: "Ultralytics",
								description: "Object detection model",
								image: "/placeholder.svg?height=200&width=400",
							},
							{
								name: "Stable Diffusion",
								type: "image",
								icon: ImageIcon,
								provider: "StabilityAI",
								description: "Text-to-image generation",
								image: "/placeholder.svg?height=200&width=400",
							},
							{
								name: "Wav2Vec",
								type: "audio",
								icon: AudioWaveform,
								provider: "Meta",
								description: "Speech processing model",
								image: "/placeholder.svg?height=200&width=400",
							},
						].map(model => (
							<div key={model.name} className="cyberpunk-corners bg-secondary p-4 space-y-4">
								<div className="flex items-center justify-between">
									<div className="flex items-center gap-2">
										<model.icon className="w-5 h-5 text-primary" />
										<h3 className="font-semibold">{model.name}</h3>
									</div>
									<span className="text-xs text-muted-foreground">{model.provider}</span>
								</div>
								<p className="text-sm text-muted-foreground">{model.description}</p>
								<Button variant="outline" className="w-full border-primary/20">
									Learn More
								</Button>
							</div>
						))}
					</div>
				</div>
			</div>
		</section>
	);
}
