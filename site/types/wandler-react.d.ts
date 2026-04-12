declare module '@wandler/react' {
  import { Message } from 'wandler';

  // Extended message type that matches the actual implementation
  export interface ExtendedMessage {
    id: string;
    role: "user" | "assistant" | "system";
    content: string;
    isComplete?: boolean;
    timestamp: number;
    metadata?: {
      reasoning?: string;
      [key: string]: unknown;
    };
  }

  // Generation config type
  export interface GenerationConfig {
    temperature?: number;
    maxTokens?: number;
    topP?: number;
    frequencyPenalty?: number;
    presencePenalty?: number;
    stopSequences?: string[];
    [key: string]: unknown;
  }

  // This is the actual API of the useChat hook
  export interface UseChatOptions {
    model: {
      id?: string;  
      [key: string]: unknown;
    };
    initialMessages?: Message[];
    initialInput?: string;
    onError?: (error: Error) => void;
    onFinish?: (messages: ExtendedMessage[]) => void;
    abortOnUnmount?: boolean;
    generationOptions?: Partial<GenerationConfig>;
  }

  // Actual return type from the useChat hook
  export interface UseChatResult {
    messages: ExtendedMessage[];
    isGenerating: boolean;
    error: Error | null;
    append: (message: string | Message, options?: Partial<GenerationConfig>) => Promise<string | void>;
    stop: () => void;
    setMessages: (messages: ExtendedMessage[] | ((current: ExtendedMessage[]) => ExtendedMessage[])) => void;
    status: 'idle' | 'loading' | 'streaming' | 'error';
    input: string;
    setInput: (input: string) => void;
    handleInputChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
    handleSubmit: (e: React.FormEvent<HTMLFormElement>) => Promise<void>;
    clearChat: () => void;
  }

  // Re-export the actual hook with the correct return type
  export function useChat(options?: Partial<UseChatOptions>): UseChatResult;
}
