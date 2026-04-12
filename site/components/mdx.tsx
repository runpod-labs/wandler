import { MDXRemote } from "next-mdx-remote/rsc";
import * as React from "react";
import { Suspense } from "react";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import rehypeSlug from "rehype-slug";
import remarkGfm from "remark-gfm";

import { CodeBlock } from "@/components/ui/code-block";
import { highlight } from "@/lib/syntax-highlighter";
import { cn } from "@/lib/utils";

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

interface PreProps extends React.ComponentPropsWithoutRef<"pre"> {
	children?: React.ReactElement & {
		props: {
			children: string;
			className?: string;
		};
	};
}

export const components = {
	h1: ({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
		<h1
			className={cn("mt-2 scroll-m-20 text-4xl font-bold tracking-tight px-4 sm:px-6", className)}
			{...props}
		/>
	),
	h2: ({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
		<h2
			className={cn(
				"mt-10 scroll-m-20 border-b border-primary/20 pb-2 text-3xl font-semibold tracking-tight first:mt-0 px-4 sm:px-6",
				className
			)}
			{...props}
		/>
	),
	h3: ({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
		<h3
			className={cn(
				"mt-8 scroll-m-20 text-2xl font-semibold tracking-tight px-4 sm:px-6",
				className
			)}
			{...props}
		/>
	),
	h4: ({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
		<h4
			className={cn(
				"mt-8 scroll-m-20 text-xl font-semibold tracking-tight px-4 sm:px-6",
				className
			)}
			{...props}
		/>
	),
	p: ({ className, ...props }: React.HTMLAttributes<HTMLParagraphElement>) => (
		<p className={cn("leading-7 [&:not(:first-child)]:mt-6 px-4 sm:px-6", className)} {...props} />
	),
	ul: ({ className, ...props }: React.HTMLAttributes<HTMLUListElement>) => (
		<ul className={cn("my-6 ml-6 list-disc [&>li]:mt-2 px-4 sm:px-6", className)} {...props} />
	),
	ol: ({ className, ...props }: React.HTMLAttributes<HTMLOListElement>) => (
		<ol className={cn("my-6 ml-6 list-decimal [&>li]:mt-2 px-4 sm:px-6", className)} {...props} />
	),
	li: ({ className, ...props }: React.HTMLAttributes<HTMLLIElement>) => (
		<li className={cn("mt-2 px-4 sm:px-6", className)} {...props} />
	),
	blockquote: ({ className, ...props }: React.HTMLAttributes<HTMLQuoteElement>) => (
		<blockquote
			className={cn(
				"mt-6 border-l-2 border-primary pl-6 italic [&>*]:text-muted-foreground px-4 sm:px-6",
				className
			)}
			{...props}
		/>
	),
	code: ({ className, ...props }: React.HTMLAttributes<HTMLElement>) => (
		<code
			className={cn(
				"relative rounded bg-secondary px-[0.3rem] py-[0.2rem] font-mono text-sm",
				className
			)}
			{...props}
		/>
	),
	pre: async ({ children, className, ...props }: PreProps) => {
		const code = children?.props?.children || "";
		const language = children?.props?.className?.replace("language-", "") || "typescript";

		const highlightedCode = await highlight(code, language, true);

		return (
			<Suspense fallback={<LoadingFallback />}>
				<CodeBlock
					code={code}
					language={language}
					highlightedCode={highlightedCode}
					showLineNumbers
					className={cn("my-6", className)}
					{...props}
				/>
			</Suspense>
		);
	},
};

interface MdxProps {
	source: string;
}

export function Mdx({ source }: MdxProps) {
	return (
		<div className="mdx">
			<MDXRemote
				source={source}
				components={components}
				options={{
					parseFrontmatter: true,
					mdxOptions: {
						remarkPlugins: [remarkGfm],
						rehypePlugins: [rehypeSlug, rehypeAutolinkHeadings],
					},
				}}
			/>
		</div>
	);
}
