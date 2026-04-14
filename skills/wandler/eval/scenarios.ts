export interface Check {
  name: string;
  test: (response: string) => boolean;
}

export interface Scenario {
  name: string;
  description: string;
  prompt: string;
  checks: Check[];
}

export const scenarios: Scenario[] = [
  // --- Basic setup ---
  {
    name: "start-llm-server",
    description: "User wants to run a local LLM and chat through an API",
    prompt:
      "I want to run a local language model on my machine and use it through an API. I heard wandler can do this. How do I get started?",
    checks: [
      {
        name: "uses --llm flag",
        test: (r) => /--llm/.test(r),
      },
      {
        name: "shows a real ONNX model ID",
        test: (r) => /onnx-community\/.*ONNX|LiquidAI\/.*ONNX/i.test(r),
      },
      {
        name: "mentions precision suffix",
        test: (r) => /:q4|:q8|:fp16|:fp32/.test(r),
      },
      {
        name: "mentions default port 8000",
        test: (r) => /8000/.test(r),
      },
      {
        name: "shows chat completions endpoint",
        test: (r) => /\/v1\/chat\/completions/.test(r),
      },
      {
        name: "mentions npx or npm install",
        test: (r) => /npx wandler|npm install.*wandler/i.test(r),
      },
    ],
  },

  // --- OpenAI SDK integration ---
  {
    name: "openai-sdk-drop-in",
    description: "User has existing OpenAI SDK code and wants to swap in a local model",
    prompt:
      "I have a Node.js app using the OpenAI SDK for chat completions. I want to switch to a local model with wandler instead of paying for the OpenAI API. How do I do this without rewriting my code?",
    checks: [
      {
        name: "mentions changing baseURL",
        test: (r) => /base_?[Uu][Rr][Ll]/i.test(r),
      },
      {
        name: "shows localhost:8000 or 127.0.0.1:8000",
        test: (r) => /localhost:8000|127\.0\.0\.1:8000/.test(r),
      },
      {
        name: "shows wandler start command",
        test: (r) => /wandler\s+--llm/.test(r),
      },
      {
        name: "mentions OpenAI compatibility",
        test: (r) => /[Oo]pen[Aa][Ii].*compat|compat.*[Oo]pen[Aa][Ii]|drop.in/i.test(r),
      },
    ],
  },

  // --- Embeddings for RAG ---
  {
    name: "embeddings-for-rag",
    description: "User is building a RAG pipeline and needs local embeddings",
    prompt:
      "I'm building a retrieval-augmented generation system. I need to generate embeddings for my documents locally. Can wandler do this? Show me how to set it up and call the API.",
    checks: [
      {
        name: "uses --embedding flag",
        test: (r) => /--embedding/.test(r),
      },
      {
        name: "mentions a real embedding model",
        test: (r) => /MiniLM|Xenova|all-MiniLM/i.test(r),
      },
      {
        name: "shows /v1/embeddings endpoint",
        test: (r) => /\/v1\/embeddings/.test(r),
      },
      {
        name: "shows combined --llm and --embedding flags",
        test: (r) =>
          /--llm\s+\S+.*--embedding|--embedding\s+\S+.*--llm/.test(r),
      },
    ],
  },

  // --- Speech-to-text ---
  {
    name: "transcribe-meeting-audio",
    description: "User wants to transcribe a WAV recording of a meeting",
    prompt:
      "I have a WAV recording of a team meeting. How can I use wandler to transcribe it to text? I need the curl command or API call.",
    checks: [
      {
        name: "mentions STT or Whisper",
        test: (r) => /stt|whisper|speech.to.text|transcri/i.test(r),
      },
      {
        name: "shows /v1/audio/transcriptions endpoint",
        test: (r) => /\/v1\/audio\/transcriptions/.test(r),
      },
      {
        name: "mentions multipart or file upload",
        test: (r) => /multipart|form-data|form_data|-F\s|file.*upload|upload.*file/i.test(r),
      },
      {
        name: "shows a wandler command or mentions server must be running",
        test: (r) => /wandler|server.*running|start.*server/i.test(r),
      },
      {
        name: "recommends wandler (does not say unsupported)",
        test: (r) =>
          !/does not support|doesn't support|not implement|cannot|can't transcribe|need a different tool/i.test(r),
      },
    ],
  },

  // --- CPU-only setup ---
  {
    name: "cpu-only-no-gpu",
    description: "User has no GPU and wants to run inference on CPU",
    prompt:
      "I'm on a Linux server with no GPU. Can I still use wandler? What flags do I need and which model should I pick for decent performance on CPU?",
    checks: [
      {
        name: "uses --device cpu",
        test: (r) => /--device\s+cpu/.test(r),
      },
      {
        name: "suggests a quantized model",
        test: (r) => /:q4|:q8|quantiz/i.test(r),
      },
      {
        name: "shows a complete wandler command with --llm",
        test: (r) => /wandler\s+.*--llm\s+\S+/.test(r),
      },
    ],
  },

  // --- Secure endpoint ---
  {
    name: "secure-team-endpoint",
    description: "User wants to expose wandler to their team with authentication",
    prompt:
      "I want to run wandler on a shared server for my team. How do I add API key authentication so only authorized users can access it?",
    checks: [
      {
        name: "uses --api-key flag",
        test: (r) => /--api-key/.test(r),
      },
      {
        name: "mentions WANDLER_API_KEY env var",
        test: (r) => /WANDLER_API_KEY/.test(r),
      },
      {
        name: "mentions --host for binding",
        test: (r) => /--host/.test(r),
      },
    ],
  },

  // --- Multi-model full stack ---
  {
    name: "multi-model-full-stack",
    description: "User wants LLM + embeddings + STT in a single server",
    prompt:
      "I need a single wandler instance that serves a chat model, an embedding model for search, and speech-to-text for voice input. Show me the command.",
    checks: [
      {
        name: "uses --llm flag",
        test: (r) => /--llm\s+\S+/.test(r),
      },
      {
        name: "uses --embedding flag",
        test: (r) => /--embedding\s+\S+/.test(r),
      },
      {
        name: "mentions --stt or default Whisper",
        test: (r) => /--stt|whisper/i.test(r),
      },
      {
        name: "shows a single wandler command with all three",
        test: (r) => /wandler\s+.*--llm\s+\S+.*--embedding\s+\S+/.test(r),
      },
    ],
  },

  // --- Streaming responses ---
  {
    name: "streaming-chat-responses",
    description: "User wants token-by-token streaming like ChatGPT",
    prompt:
      'How do I get streaming responses from wandler? I want tokens to appear in real-time in my frontend, like ChatGPT. Show me the API call with stream: true.',
    checks: [
      {
        name: 'mentions "stream": true in request body',
        test: (r) => /["']?stream["']?\s*:\s*true/.test(r),
      },
      {
        name: "mentions SSE or server-sent events or text/event-stream",
        test: (r) => /SSE|server.sent.event|text\/event-stream/i.test(r),
      },
      {
        name: "shows /v1/chat/completions endpoint",
        test: (r) => /\/v1\/chat\/completions/.test(r),
      },
      {
        name: "mentions data: prefix or chunk parsing",
        test: (r) => /data:\s|chunk|delta|event/i.test(r),
      },
      {
        name: "uses correct default port 8000",
        test: (r) => /localhost:8000|127\.0\.0\.1:8000/.test(r),
      },
    ],
  },

  // --- Model precision tradeoffs ---
  {
    name: "precision-selection",
    description: "User wants to understand precision options and pick the right one",
    prompt:
      "What precision options does wandler support for models? I want to understand the tradeoff between q4, q8, fp16, and fp32. Which should I use for my 16GB RAM machine?",
    checks: [
      {
        name: "mentions q4",
        test: (r) => /q4/.test(r),
      },
      {
        name: "mentions q8",
        test: (r) => /q8/.test(r),
      },
      {
        name: "mentions fp16",
        test: (r) => /fp16/.test(r),
      },
      {
        name: "explains colon suffix syntax",
        test: (r) => /:\s*q4|:\s*q8|:\s*fp16|:\s*fp32|suffix|append/i.test(r),
      },
      {
        name: "discusses size/quality tradeoff",
        test: (r) =>
          /quality|accura|speed|memory|smaller|larger|tradeoff|trade-off/i.test(r),
      },
    ],
  },

  // --- Tool/function calling ---
  {
    name: "tool-calling-with-functions",
    description: "User wants to use function calling like with OpenAI",
    prompt:
      "I use OpenAI function calling (tools) heavily in my app. Does wandler support tool calling? Show me an example request body with a tool definition.",
    checks: [
      {
        name: "confirms tool calling support",
        test: (r) => /tool|function.call/i.test(r),
      },
      {
        name: "shows a tools array or function definition",
        test: (r) => /"tools"\s*:|"functions"\s*:|tools.*\[/.test(r),
      },
      {
        name: "references /v1/chat/completions",
        test: (r) => /\/v1\/chat\/completions/.test(r),
      },
      {
        name: "shows a JSON request body example",
        test: (r) => /"model"\s*:|"messages"\s*:/.test(r),
      },
      {
        name: "uses a real wandler model ID or correct port",
        test: (r) =>
          /onnx-community|gemma.*ONNX|LiquidAI|localhost:8000|127\.0\.0\.1:8000/i.test(r),
      },
    ],
  },
];
