import { cache } from "react";

import { highlight } from "@/lib/syntax-highlighter";

import { CodeBlock } from "./code-block";

interface CodeBlockServerProps {
	code: string;
	language?: string;
	showLineNumbers?: boolean;
	className?: string;
	variant?: "default" | "install";
}

// Cache the highlight function to avoid re-running it unnecessarily
const getHighlightedCode = cache(
	async (code: string, language: string, showLineNumbers: boolean) => {
		return highlight(code, language, showLineNumbers);
	}
);

export async function CodeBlockServer(props: CodeBlockServerProps) {
	const highlightedCode = await getHighlightedCode(
		props.code,
		props.language || "typescript",
		props.showLineNumbers || false
	);
	return <CodeBlock {...props} highlightedCode={highlightedCode} />;
}
