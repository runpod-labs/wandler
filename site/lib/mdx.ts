import { compileMDX } from "next-mdx-remote/rsc";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import rehypeSlug from "rehype-slug";
import remarkGfm from "remark-gfm";

import { components } from "@/components/mdx";

export async function getMdxContent(source: string) {
	const { content } = await compileMDX({
		source,
		components,
		options: {
			parseFrontmatter: true,
			mdxOptions: {
				remarkPlugins: [remarkGfm],
				rehypePlugins: [rehypeSlug, rehypeAutolinkHeadings],
			},
		},
	});

	return content;
}
