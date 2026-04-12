import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";
import tseslint from "@typescript-eslint/eslint-plugin";
import tsparser from "@typescript-eslint/parser";
import unusedImports from "eslint-plugin-unused-imports";
import simpleImportSort from "eslint-plugin-simple-import-sort";
import importPlugin from "eslint-plugin-import";
import prettier from "eslint-plugin-prettier";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
	baseDirectory: __dirname,
});

const eslintConfig = [
	{
		ignores: [
			// Build output
			"**/.next/**",
			"**/out/**",
			"**/dist/**",
			"**/build/**",

			// Dependencies
			"**/node_modules/**",

			// Cache
			"**/.cache/**",
			"**/.eslintcache",

			// Misc
			"**/.DS_Store",
			"**/*.pem",

			// Debug
			"**/npm-debug.log*",
			"**/yarn-debug.log*",
			"**/yarn-error.log*",

			// Local env files
			"**/.env*.local",

			// TypeScript
			"**/*.tsbuildinfo",
			"**/next-env.d.ts",
		],
	},
	...compat.extends("next/core-web-vitals", "next/typescript"),
	{
		files: ["**/*.ts", "**/*.tsx"],
		plugins: {
			"@typescript-eslint": tseslint,
			"unused-imports": unusedImports,
			"simple-import-sort": simpleImportSort,
			import: importPlugin,
			prettier: prettier,
		},
		languageOptions: {
			parser: tsparser,
			parserOptions: {
				project: "./tsconfig.json",
			},
		},
		rules: {
			"no-unused-vars": "off",
			"@typescript-eslint/no-unused-vars": "off",
			"unused-imports/no-unused-imports": "error",
			"unused-imports/no-unused-vars": [
				"warn",
				{
					vars: "all",
					varsIgnorePattern: "^_",
					args: "after-used",
					argsIgnorePattern: "^_",
				},
			],
			"simple-import-sort/imports": "error",
			"simple-import-sort/exports": "error",
			"import/first": "error",
			"import/newline-after-import": "error",
			"import/no-duplicates": "error",
		},
	},
];

export default eslintConfig;
