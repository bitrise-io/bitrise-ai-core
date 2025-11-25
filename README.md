# Bitrise AI Agent Core Framework - Educational Version
This repository contains a stripped-down version of Bitrise's internal AI agent framework, released for educational purposes. It is designed to complement this article: [How we brought AI to the Bitrise Build Cloud
](https://bitrise.io/blog/post/how-we-brought-ai-to-the-bitrise-build-cloud).

# Example
The `example` directory contains a simple example project demonstrating how to use the core framework. It consists of two sub-agents:
1. **File reviewer**: we run multiple ones in parallel to review different files in a repository. It shows tool usage via a very simple file reading tool.
2. **Summary agent**: summarizes the reviews collected from the file reviewers.

To run the example set an LLM API key environment variable:
- For OpenAI: `export OPENAI_API_KEY="your_api_key_here"`
- For Anthropic: `export ANTHROPIC_API_KEY="your_api_key_here"`
- For Gemini: `export GEMINI_API_KEY="your_api_key_here"`

Then execute the following command from the repository root:
```bash
go run ./example <path to directory to review> # for example: go run ./example pkg/llm
```
The example will read all files in the specified directory, have the file reviewer agents review them in parallel, and then summarize the reviews.
