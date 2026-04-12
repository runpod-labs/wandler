/** @type {import('next').NextConfig} */
const config = {
	images: {
		remotePatterns: [
			{
				protocol: "https",
				hostname: "5xvkmufwzznj1ey2.public.blob.vercel-storage.com",
				port: "",
				pathname: "/**",
			},
		],
	},

	// External packages that should not be bundled by the server
	serverExternalPackages: ["onnxruntime-node", "@huggingface/transformers", "wandler"],

	// Override the default webpack configuration
	webpack: (config, { isServer }) => {
		// Ignore node-specific modules when bundling for the browser
		config.resolve.alias = {
			...config.resolve.alias,
			"onnxruntime-node": false,
		};

		if (isServer) {
			config.externals = [
				...(Array.isArray(config.externals) ? config.externals : []),
				"onnxruntime-node",
				"@huggingface/transformers",
				"wandler",
			];
		}

		return config;
	},
};

export default config;
