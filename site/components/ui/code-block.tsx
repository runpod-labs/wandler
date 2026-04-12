"use client";

import { Check, Copy } from "lucide-react";
import { useState } from "react";

import { cn } from "@/lib/utils";

interface CopyButtonProps {
	code: string; 
}

function CopyButton({ code }: CopyButtonProps) {
	const [copied, setCopied] = useState(false);

	const handleCopy = async () => {
		try {
			await navigator.clipboard.writeText(code);
			setCopied(true);
			setTimeout(() => setCopied(false), 2000);
		} catch (err) {
			console.error("Failed to copy code:", err);
		}
	};

	return (
		<div className="absolute right-4 top-4 opacity-0 group-hover:opacity-100 transition-opacity z-10">
			<button
				onClick={handleCopy}
				className="p-2 hover:text-primary transition-colors"
				title="Copy to clipboard"
			>
				{copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
			</button>
		</div>
	);
}

interface CodeBlockProps {
	code: string;
	className?: string;
	variant?: "default" | "install";
	highlightedCode?: string;
	language?: string;
	showLineNumbers?: boolean;
}

export function CodeBlock({
	code,
	className,
	variant = "default",
	highlightedCode,
}: CodeBlockProps) {
	if (variant === "install") {
		return (
			<div className="cyberpunk-corners bg-secondary p-4">
				<div className="install-tabs">
					<div className="install-tab active">npm</div>
					<div className="install-tab">yarn</div>
					<div className="install-tab">pnpm</div>
				</div>
				<div className="install-content flex justify-between items-center">
					<code>{code}</code>
					<CopyButton code={code} />
				</div>
			</div>
		);
	}

	return (
		<div className={cn("relative group cyberpunk-code", className)}>
			<CopyButton code={code} />
			<div className="cyberpunk-corners bg-black/50 p-2 sm:p-4 border border-primary/50">
				{highlightedCode ? (
					<div
						className="[&_pre]:!bg-transparent [&_pre]:!p-0 text-primary font-mono text-sm overflow-x-auto [&_pre]:overflow-y-auto"
						dangerouslySetInnerHTML={{ __html: highlightedCode }}
					/>
				) : (
					<pre className="text-primary font-mono text-sm">
						<code>{code}</code>
					</pre>
				)}
			</div>
		</div>
	);
}
