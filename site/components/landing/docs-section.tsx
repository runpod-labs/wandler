import { Book, Code, Code2, FileCode } from "lucide-react";
import Link from "next/link";

interface DocsSectionProps {
	sampleCode: React.ReactNode;
}

export function DocsSection({ sampleCode }: DocsSectionProps) {
	return (
		<section id="docs" className="py-16 relative overflow-hidden border-b border-primary/20">
			<div className="absolute inset-0 bg-black">
				<div className="absolute inset-0 bg-noise"></div>
			</div>
			<div className="container mx-auto px-4 relative z-10">
				<div className="grid md:grid-cols-2 gap-12 items-center">
					<div className="space-y-6">
						<h2 className="text-4xl font-bold text-primary">Documentation</h2>
						<p className="text-lg text-muted-foreground">
							Comprehensive guides and API references to help you integrate AI models into your
							applications with ease.
						</p>
						<div className="grid gap-4">
							<div className="flex items-start gap-4">
								<div className="p-2 bg-secondary rounded-md">
									<Book className="w-6 h-6 text-primary" />
								</div>
								<div>
									<h3 className="font-semibold mb-1">Getting Started Guide</h3>
									<p className="text-sm text-muted-foreground">
										Learn the basics and get your first AI model running in minutes.
									</p>
								</div>
							</div>
							<div className="flex items-start gap-4">
								<div className="p-2 bg-secondary rounded-md">
									<Code2 className="w-6 h-6 text-primary" />
								</div>
								<div>
									<h3 className="font-semibold mb-1">API Reference</h3>
									<p className="text-sm text-muted-foreground">
										Detailed documentation of all available methods and options.
									</p>
								</div>
							</div>
							<div className="flex items-start gap-4">
								<div className="p-2 bg-secondary rounded-md">
									<FileCode className="w-6 h-6 text-primary" />
								</div>
								<div>
									<h3 className="font-semibold mb-1">Examples</h3>
									<p className="text-sm text-muted-foreground">
										Real-world examples and use cases to help you get started.
									</p>
								</div>
							</div>
						</div>
						<div className="flex gap-4">
							<Link href="/docs" className="button-cyber">
								View Documentation
							</Link>
							<Link
								href="/docs/installation"
								className="flex items-center gap-2 px-4 py-2 bg-secondary text-primary rounded hover:bg-primary/10 transition-colors"
							>
								<Code className="w-5 h-5" />
								<span>Installation Guide</span>
							</Link>
						</div>
					</div>
					<div className="relative">
						<div className="cyberpunk-corners bg-secondary p-6">{sampleCode}</div>
					</div>
				</div>
			</div>
		</section>
	);
}
