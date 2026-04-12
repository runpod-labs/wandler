import { Suspense } from "react";

import { CodeBlockServer } from "@/components/ui/code-block-server";

interface CodeExampleProps {
	code: string;
	language: string;
}

function LoadingFallback() {
	return (
		<div className="cyberpunk-corners bg-primary/10 p-4">
			<div className="animate-pulse space-y-2">
				<div className="h-4 bg-primary/20 rounded w-3/4" />
				<div className="h-4 bg-primary/20 rounded w-1/2" />
				<div className="h-4 bg-primary/20 rounded w-2/3" />
			</div>
		</div>
	);
}

export function CodeExample({ code, language }: CodeExampleProps) {
	return (
		<Suspense fallback={<LoadingFallback />}>
			<CodeBlockServer code={code} language={language} className="bg-primary/10" />
		</Suspense>
	);
}
