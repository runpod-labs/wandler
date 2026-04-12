import { ArrowLeft } from "lucide-react";
import Link from "next/link";

import { Button } from "@/components/ui/button";

export default function HubNotFound() {
	return (
		<div className="flex flex-col items-center justify-center min-h-[60vh] text-center space-y-8">
			<div className="space-y-4">
				<h1 className="text-4xl md:text-6xl font-bold tracking-tighter">
					Demo <span className="text-primary">Not Found</span>
				</h1>
				<p className="text-xl text-muted-foreground max-w-3xl">
					The demo you&apos;re looking for doesn&apos;t exist or isn&apos;t available yet.
				</p>
			</div>

			<Link href="/hub">
				<Button variant="outline" className="flex items-center gap-2">
					<ArrowLeft className="h-4 w-4" />
					Back to Hub
				</Button>
			</Link>
		</div>
	);
}
