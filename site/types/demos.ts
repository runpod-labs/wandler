/**
 * Demo interface representing a single demo in the application
 */
export interface Demo {
	/**
	 * Unique identifier for the demo
	 */
	id: string;

	/**
	 * Display title for the demo
	 */
	title: string;

	/**
	 * Short description of what the demo does
	 */
	description: string;

	/**
	 * URL path to the demo
	 */
	path: string;

	/**
	 * Tags for categorizing and filtering demos
	 */
	tags: string[];

	/**
	 * Model ID to use for this demo (from Hugging Face)
	 */
	modelId: string;

	/**
	 * Path to the thumbnail image for the demo
	 */
	thumbnail: string;

	/**
	 * Whether this demo should be featured on the homepage
	 */
	featured: boolean;
}
