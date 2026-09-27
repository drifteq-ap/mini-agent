import OpenAI from "openai";
import { readFileSync } from "node:fs";
import * as readline from "node:readline/promises";

// Config. Any OpenAI-compatible server works. To use a hosted provider, change
// these three values (or set the env vars), e.g. for OpenAI itself:
//   BASE_URL=https://api.openai.com/v1  API_KEY=sk-...  MODEL=gpt-4.1
const BASE_URL = process.env.BASE_URL ?? "http://localhost:11434/v1"; // local Ollama
const API_KEY = process.env.API_KEY ?? "ollama"; // dummy, Ollama ignores it
const MODEL = process.env.MODEL ?? "qwen2.5";

const client = new OpenAI({ baseURL: BASE_URL, apiKey: API_KEY });

// ---------- Tools, part 1: the schemas the model sees ----------
// Same JSON schema as before, but wrapped in {type: "function", function: {...}},
// and the schema moves from `input_schema` to `parameters`.
const tools: OpenAI.Chat.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "read_file",
      description: "Read a UTF-8 text file and return its full contents.",
      parameters: {
        type: "object",
        properties: { path: { type: "string", description: "File path, relative to the current directory" } },
        required: ["path"],
      },
    },
  },
  // YOUR TURN: add the write_file schema here (inputs: path, content), in the same {type: "function", ...} wrapper.
  // YOUR TURN: add the run_bash schema here (input: command).
];

// ---------- Tools, part 2: the functions we actually run ----------
// Errors come back as strings so the model can read them and try again.
function runTool(name: string, input: Record<string, string>): string {
  try {
    switch (name) {
      case "read_file":
        return readFileSync(input.path, "utf8");
      // YOUR TURN: case "write_file" -> writeFileSync(input.path, input.content), then return a short confirmation.
      // YOUR TURN: case "run_bash"   -> execSync(input.command, { encoding: "utf8" }), then return its output.
      default:
        return `Error: unknown tool "${name}"`;
    }
  } catch (err) {
    return `Error: ${(err as Error).message}`;
  }
}

// ---------- The agent loop ----------

// The whole conversation, as the API sees it. It lives outside the REPL, so
// every new task still has the earlier ones in context. Nothing is ever
// removed, so it only grows (on purpose, for now).
const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [];

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

// OUTER LOOP: one pass per task you type.
while (true) {
  const task = (await rl.question("\n> ")).trim();
  if (!task) continue;
  if (task === "exit") break;

  // Your task becomes a new "user" turn.
  messages.push({ role: "user", content: task });

  // INNER LOOP: one pass per model call. A single task can take many passes:
  // the model asks for a tool, we run it, send back the result, and call again.
  while (true) {
    // 1. Send the WHOLE conversation plus the tool list. The API keeps no
    //    state between calls. We resend everything every time.
    const response = await client.chat.completions.create({
      model: MODEL,
      tools,
      messages,
    });

    // The reply is a single message inside choices[0], not a list of content
    // blocks. Text is in .content, and tool requests are in .tool_calls.
    const choice = response.choices[0];
    const reply = choice.message;

    // 2. Record the model's reply as an "assistant" turn, exactly as it came
    //    back, tool_calls included. It has to come BEFORE the tool results,
    //    because each result points back at a tool_call id in this message.
    messages.push(reply);

    // 3. Show the text to the human.
    if (reply.content) console.log(reply.content);

    // 4. Check whether the model asked for any tools. There is no reliable
    //    stop_reason here: some servers (including Ollama) report
    //    finish_reason "stop" even when they return tool calls. So look at
    //    tool_calls directly. If there are none, the task is done and we go
    //    back to the prompt.
    if (!reply.tool_calls?.length) {
      if (choice.finish_reason !== "stop") console.log(`[stopped: ${choice.finish_reason}]`);
      break;
    }

    // 5. Run every tool the model asked for. Each call has an id, and its
    //    result must quote that same id so the model can tell which result
    //    answers which call.
    for (const call of reply.tool_calls) {
      if (call.type !== "function") continue; // we only define function tools
      const { name, arguments: args } = call.function;
      console.log(`[tool] ${name}(${args})`);

      // Arguments arrive as a JSON STRING, not an object. Parse it first.
      // Small local models sometimes send broken JSON. That becomes an error
      // string for the model, like any other tool error.
      let result: string;
      try {
        result = runTool(name, JSON.parse(args));
      } catch {
        result = `Error: tool arguments were not valid JSON: ${args}`;
      }

      // 6. Each result is its OWN message with role "tool", linked back by
      //    tool_call_id. (Anthropic put all the results in one "user" message.)
      //    Then we loop back to step 1 so the model can read them.
      messages.push({ role: "tool", tool_call_id: call.id, content: result });
    }
  }
}

rl.close();
