import { BookOpen, Code, Cpu, MessageSquare } from "lucide-react";
import Link from "next/link";

import { AVAILABLE_MODELS } from "@/lib/models";

interface DemoCardProps {
	title: string;
	description: string;
	icon: React.ReactNode;
	href: string;
	isAvailable?: boolean;
}

function DemoCard({ title, description, icon, href, isAvailable = true }: DemoCardProps) {
	return (
		<Link
			href={isAvailable ? href : "#"}
			className={`block p-6 border rounded-lg transition-all ${
				isAvailable
					? "border-primary/30 hover:border-primary hover:bg-primary/5 cursor-pointer"
					: "border-gray-700 opacity-60 cursor-not-allowed"
			}`}
		>
			<div className="flex items-center mb-4">
				<div className="mr-3 text-primary">{icon}</div>
				<h3 className="text-xl font-bold">{title}</h3>
				{!isAvailable && (
					<span className="ml-2 px-2 py-1 text-xs bg-gray-800 rounded-full">Coming Soon</span>
				)}
			</div>
			<p className="text-muted-foreground">{description}</p>
		</Link>
	);
}

export default function HubPage() {
	return (
		<div className="space-y-12">
			<div className="space-y-4">
				<h1 className="text-4xl md:text-6xl font-bold tracking-tighter">hub</h1>
			</div>

			<section>
				<h2 className="text-2xl md:text-4xl font-bold mb-6 flex items-center">apps</h2>
				<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
					<DemoCard
						title="Chat"
						description="Chat with AI models running directly in your browser."
						icon={<MessageSquare className="h-6 w-6" />}
						href="/hub/chat"
					/>
				</div>
			</section>

			<section>
				<h2 className="text-2xl md:text-4xl font-bold mb-6 flex items-center">models</h2>
				<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
					{AVAILABLE_MODELS.map(model => (
						<DemoCard
							key={model.id}
							title={model.humanReadableName}
							description={model.description}
							icon={<Cpu className="h-6 w-6" />}
							href={`/hub/${model.id}`}
						/>
					))}
				</div>
			</section>

			<section>
				<h2 className="text-2xl md:text-4xl font-bold mb-6 flex items-center">resources</h2>
				<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
					<DemoCard
						title="Documentation"
						description="Learn how to use Wandler in your own projects with our comprehensive docs."
						icon={<BookOpen className="h-6 w-6" />}
						href="/docs"
					/>
					<DemoCard
						title="GitHub"
						description="Check out our open-source code and contribute to the project."
						icon={<Code className="h-6 w-6" />}
						href="https://github.com/wandlerai/wandler"
					/>
				</div>
			</section>
		</div>
	);
}
