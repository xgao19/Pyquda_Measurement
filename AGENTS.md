# Repository Expectations

## Code Comments
- All code comments must be written in English only.
- Never write comments in Chinese or any other non-English language.
- This applies to both new code and modifications to existing code.

## Commit Memory
- Before each commit, update `log.md`.
- Update `SESSION_MEMORY.md` only when reusable environment facts, stable
  conventions, or easy-to-repeat pitfalls actually change.
- Do not add commit summaries, completed-work notes, validation diaries, or old
  output paths to `SESSION_MEMORY.md`; keep those in `log.md` and Git history.

## Behavioral Guidelines
These guidelines reduce common LLM coding mistakes. Use judgment for trivial tasks.

### 1. Think Before Coding
- State assumptions explicitly.
- If something is uncertain, ask rather than guessing.
- If multiple interpretations exist, surface them instead of choosing silently.
- If a simpler approach exists, say so.
- Push back when a request looks overcomplicated or internally inconsistent.
- If something is unclear, stop and name what is confusing.

### 2. Simplicity First
- Implement the minimum code that solves the problem.
- Do not add features beyond what was asked.
- Do not introduce abstractions for single-use code.
- Do not add flexibility or configurability that was not requested.
- Do not add error handling for impossible scenarios.
- If a change feels too large, simplify it.

### 3. Surgical Changes
- Touch only what is necessary for the request.
- Do not "improve" adjacent code, comments, or formatting.
- Do not refactor unrelated code.
- Match existing style, even if you would do it differently.
- If you notice unrelated dead code, mention it instead of deleting it.
- Remove imports, variables, or functions only if your own change made them unused.
- Do not remove pre-existing dead code unless asked.

### 4. Goal-Driven Execution
- Turn tasks into verifiable goals.
- For multi-step work, state a brief plan with explicit checks.
- Loop until the result is verified.

Example:
```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

### What Good Looks Like
- Fewer unnecessary changes in diffs
- Fewer rewrites due to overcomplication
- Clarifying questions before implementation, not after mistakes
