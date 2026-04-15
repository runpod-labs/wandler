export interface CatalogModel {
  id: string;
  name: string;
  type: "llm" | "embedding" | "stt";
  size: string;
  precisions: string[];
  defaultPrecision: string;
  capabilities: string[];
  dimensions?: number;
  status: "stable" | "experimental" | "deprecated";
}

export interface Catalog {
  version: string;
  models: CatalogModel[];
}
