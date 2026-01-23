/**
 * Chat Types - Feature modes, options, and API keys
 */

export type FeatureMode =
  | "chat"
  | "rag"
  | "multi-agent"
  | "knowledge-graph"
  | "audio"
  | "ocr"
  | "google"
  | "web-search";

export type GoogleService =
  | "docs"
  | "drive"
  | "gmail"
  | "calendar"
  | "sheets";

export interface APIKeys {
  openai?: string;
  anthropic?: string;
  google?: string;
  tavily?: string;
}

export interface FeatureOptions {
  // RAG
  documents_path?: string;
  collection_name?: string;
  chunk_size?: number;
  top_k?: number;

  // Multi-Agent
  agent_count?: number;
  strategy?: "debate" | "hierarchical" | "graph";

  // Knowledge Graph
  graph_name?: string;
  neo4j_uri?: string;

  // Audio
  audio_file?: File;
  audio_engine?: "whisper" | "google";

  // OCR
  image_file?: File;
  ocr_engine?: "tesseract" | "google";

  // Google
  google_services?: GoogleService[];

  // Web Search
  search_query?: string;
  search_max_results?: number;
}

export interface ProviderConfig {
  providers: string[];
  config: Record<string, any>;
}

export interface AvailableModels {
  [provider: string]: string[];
}
