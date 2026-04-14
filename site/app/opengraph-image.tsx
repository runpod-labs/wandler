import { ImageResponse } from "next/og";

export const runtime = "edge";
export const alt = "wandler — OpenAI-compatible inference server for transformers.js";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default async function Image() {
	return new ImageResponse(
		(
			<div
				style={{
					width: "100%",
					height: "100%",
					display: "flex",
					flexDirection: "column",
					justifyContent: "center",
					alignItems: "center",
					background: "#000",
					position: "relative",
					overflow: "hidden",
				}}
			>
				{/* Subtle grid pattern */}
				<div
					style={{
						position: "absolute",
						inset: 0,
						backgroundImage:
							"linear-gradient(rgba(255,236,25,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(255,236,25,0.06) 1px, transparent 1px)",
						backgroundSize: "40px 40px",
					}}
				/>

				{/* Top hazard stripe */}
				<div
					style={{
						position: "absolute",
						top: 0,
						left: 0,
						right: 0,
						height: 6,
						background:
							"repeating-linear-gradient(90deg, #ffec19 0px, #ffec19 30px, #000 30px, #000 60px)",
					}}
				/>

				{/* Bottom hazard stripe */}
				<div
					style={{
						position: "absolute",
						bottom: 0,
						left: 0,
						right: 0,
						height: 6,
						background:
							"repeating-linear-gradient(90deg, #ffec19 0px, #ffec19 30px, #000 30px, #000 60px)",
					}}
				/>

				{/* Corner brackets */}
				<div
					style={{
						position: "absolute",
						top: 24,
						left: 24,
						width: 40,
						height: 40,
						borderTop: "3px solid #ffec19",
						borderLeft: "3px solid #ffec19",
					}}
				/>
				<div
					style={{
						position: "absolute",
						top: 24,
						right: 24,
						width: 40,
						height: 40,
						borderTop: "3px solid #ffec19",
						borderRight: "3px solid #ffec19",
					}}
				/>
				<div
					style={{
						position: "absolute",
						bottom: 24,
						left: 24,
						width: 40,
						height: 40,
						borderBottom: "3px solid #ffec19",
						borderLeft: "3px solid #ffec19",
					}}
				/>
				<div
					style={{
						position: "absolute",
						bottom: 24,
						right: 24,
						width: 40,
						height: 40,
						borderBottom: "3px solid #ffec19",
						borderRight: "3px solid #ffec19",
					}}
				/>

				{/* Logo text */}
				<div
					style={{
						display: "flex",
						fontSize: 96,
						fontWeight: 900,
						letterSpacing: "-0.04em",
						color: "#ffec19",
						marginBottom: 16,
						textTransform: "uppercase",
					}}
				>
					WANDLER
				</div>

				{/* Tagline */}
				<div
					style={{
						display: "flex",
						fontSize: 28,
						color: "#b3b3b3",
						letterSpacing: "0.02em",
					}}
				>
					OpenAI-compatible inference server
				</div>

				{/* Powered by line */}
				<div
					style={{
						display: "flex",
						gap: 8,
						alignItems: "center",
						marginTop: 12,
						fontSize: 22,
						color: "#666",
					}}
				>
					<span>powered by</span>
					<span style={{ color: "#ffec19" }}>transformers.js</span>
					<span style={{ margin: "0 4px" }}>·</span>
					<span>WebGPU accelerated</span>
				</div>

				{/* Command line */}
				<div
					style={{
						display: "flex",
						alignItems: "center",
						marginTop: 40,
						background: "#1a1a1a",
						border: "1px solid rgba(255,236,25,0.3)",
						padding: "12px 24px",
						fontSize: 18,
						fontFamily: "monospace",
						color: "#00ffff",
					}}
				>
					$ npx wandler --llm LiquidAI/LFM2.5-1.2B-Instruct-ONNX
				</div>
			</div>
		),
		{ ...size }
	);
}
