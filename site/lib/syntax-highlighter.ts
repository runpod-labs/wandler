import { createHighlighter } from "shiki";

let highlighter: Awaited<ReturnType<typeof createHighlighter>>;

// eslint-disable-next-line unused-imports/no-unused-vars
export async function highlight(code: string, lang: string, showLineNumbers = false) {
	if (!highlighter) {
		highlighter = await createHighlighter({
			themes: ["github-dark"],
			langs: ["typescript", "javascript", "bash", "json", "markdown", "yaml", "tsx", "jsx"],
		});
	}

	try {
		// Trim empty lines at start and end
		code = code.replace(/^\n+|\n+$/g, "");

		return highlighter.codeToHtml(code, {
			lang,
			theme: "github-dark",
		});
	} catch {
		// Fallback to plain text if language is not supported
		return highlighter.codeToHtml(code.replace(/^\n+|\n+$/g, ""), {
			lang: "txt",
			theme: "github-dark",
		});
	}
}
