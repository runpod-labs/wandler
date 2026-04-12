"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { Header } from "@/components/header";
import { cn } from "@/lib/utils";

// Define interfaces for the sidebar items
interface SubItem {
	title: string;
	href: string;
}

interface SidebarItem {
	title: string;
	href?: string;
	items?: SubItem[];
}

interface SidebarSection {
	title: string;
	items: SidebarItem[];
}

const sidebarItems: SidebarSection[] = [
	{
		title: "get started",
		items: [
			{
				title: "install",
				href: "/docs#install",
			},
			{
				title: "hello, world!",
				href: "/docs#hello-world",
			},
		],
	},
	{
		title: "wandler",
		items: [
			{
				title: "loadModel",
				href: "/docs#loadmodel",
			},
			{
				title: "generateText",
				href: "/docs#generatetext",
			},
			{
				title: "streamText",
				href: "/docs#streamtext",
			},
		],
	},
	{
		title: "react",
		items: [
			{
				title: "install",
				href: "/docs#install-react-package",
			},
			{
				title: "useChat",
				href: "/docs#usechat",
			},
		],
	},
];

function Sidebar() {
	const pathname = usePathname();

	return (
		<aside className="w-48 border-r border-primary/20 bg-black lg:fixed h-[calc(100vh-4rem)] lg:top-16">
			<div className="overflow-y-auto h-full p-4">
				<nav className="space-y-8">
					{sidebarItems.map((section, i) => (
						<div key={i}>
							<h2 className="mb-4 text-lg font-semibold text-primary">{section.title}</h2>
							<ul className="space-y-2">
								{section.items.map((item, j) => (
									<li key={j}>
										<Link
											href={item.href || "#"}
											className={cn(
												"block py-1.5 text-sm transition-colors hover:text-primary",
												pathname === item.href
													? "text-primary font-medium"
													: "text-muted-foreground"
											)}
										>
											{item.title}
										</Link>
										{item.items && (
											<ul className="ml-4 mt-2 space-y-1 border-l border-primary/20">
												{item.items.map((subItem, k) => (
													<li key={k}>
														<Link
															href={subItem.href}
															className={cn(
																"block py-1 text-sm transition-colors hover:text-primary",
																pathname === subItem.href
																	? "text-primary font-medium"
																	: "text-muted-foreground"
															)}
														>
															{subItem.title}
														</Link>
													</li>
												))}
											</ul>
										)}
									</li>
								))}
							</ul>
						</div>
					))}
				</nav>
			</div>
		</aside>
	);
}

export default function DocsLayout({ children }: { children: React.ReactNode }) {
	return (
		<div className="flex flex-col min-h-screen bg-black text-white">
			<Header />

			{/* Main content */}
			<div className="container mx-auto">
				<div className="flex pt-16">
					{/* Desktop Sidebar */}
					<div className="hidden lg:block">
						<Sidebar />
					</div>

					{/* Main content */}
					<main className="flex-1 lg:ml-48">
						<div className="max-w-3xl py-12">{children}</div>
					</main>
				</div>
			</div>
		</div>
	);
}
