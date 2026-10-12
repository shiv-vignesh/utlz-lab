# Getting Comfortable Reading vLLM's Codebase

## Goal

Build comfort with vLLM's software architecture without being spoon-fed implementation details. The goal is to learn to trace ownership, state, and control flow—not just recognize API names.

The key shift is from thinking like a PyTorch model developer to thinking like a runtime engineer. A model is one component in a system that manages requests, GPU execution, memory allocation, and scheduling.

## A Useful Mental Model

Think of the system as a pipeline:

1. **Request interface** — accepts a prompt and generation parameters.
2. **Request lifecycle and scheduler** — tracks requests and decides which work runs next.
3. **KV-cache manager** — manages the attention state retained between decoding steps.
4. **Model runner** — turns scheduled work into model inputs and GPU execution.
5. **Outputs and next engine step** — samples tokens, updates request state, and schedules more work.

This is a conceptual map, not necessarily a literal call stack. In V1, the frontend and engine core are separated, and execution may happen in another process.

## The Code Areas to Explore, in Order

### 1. Public API and Object Graph

Start with:

- `vllm/entrypoints/llm.py`
- `LLM`
- `SamplingParams`
- `LLM.generate()`

Questions to answer:

- What objects does `LLM` construct?
- Which object owns the engine?
- Where does `generate()` hand the request off?
- Which component owns the generation loop?

Draw the object graph before following every function call.

### 2. Request Lifecycle and Scheduler

Explore:

- `vllm/v1/core/sched/scheduler.py`
- The request state and transitions between waiting, running, and completed
- Continuous or iteration-level batching

Questions to answer:

- What state is stored for each request?
- How does the scheduler decide what runs in the next step?
- What does it admit, defer, or retire?
- How does it account for prompt processing versus token-by-token decoding?

### 3. KV-Cache Management

Explore the concepts and relevant implementation around:

- `KVCacheManager`
- `BlockPool`
- Block tables
- Logical-token-to-physical-block mapping

The central question is: **what state must the runtime preserve between forward passes that a normal `nn.Module` does not manage by itself?**

Try drawing a tiny example with a few requests and a few KV blocks. Track which blocks are free, which are allocated, and how each request maps logical tokens to physical blocks.

### 4. Model Runner

Explore:

- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/model_executor/models/`

Focus on the boundary between scheduling and execution:

- How are scheduled requests converted into model inputs?
- What are positions, sequence lengths, slot mappings, and query-start locations used for?
- Where are new K/V values written?
- Where do attention, logits, and sampling happen?

You do not need to understand every kernel at first. First understand the data passed into and out of the model runner.

### 5. Frontend, Engine Core, and Workers

Explore:

- `vllm/v1/engine/`
- `EngineCore`
- `AsyncLLM`
- `EngineCoreClient`

Initially, focus on **message contracts and process boundaries**, not all the multiprocessing or ZMQ details. Learn which component sends what to which other component and who owns each piece of state.

> Version note: names and paths can change between vLLM versions. Use the code installed in your environment as the source of truth.

## A First Code-Reading Exercise

Trace one call to `LLM.generate()`:

1. Find the public entry point and its direct collaborators.
2. Draw the object graph and label who owns the engine, scheduler, and model executor.
3. Follow a prompt until it reaches the engine.
4. Find the engine-step loop.
5. Trace one generated token from submission through model execution and sampling, then identify what changes before the next scheduling decision.

When the call chain becomes deep, pause. For each function you inspect, record:

- **Inputs:** what information comes in?
- **Outputs:** what information goes out?
- **State changes:** what persistent state is updated?
- **Responsibility:** what decision or transformation belongs here?

Avoid trying to understand the entire repository in one pass.

## A Small Experiment to Anchor the Reading

Use one workload and compare Hugging Face native inference with vLLM. Keep input and output token lengths consistent.

Start with vLLM's offline `LLM` API to avoid adding HTTP and network behavior to the first experiment. Verify token counts and throughput before drawing conclusions.

Then change **one configuration variable at a time**, for example:

- `max_num_seqs`
- `max_num_batched_tokens`
- `enforce_eager`
- Prefix caching

Before each run, write down what you predict will change. Afterward, measure it and explain the result by connecting it to the code you have been reading.

For latency and per-token timing, remember that `LLM.generate()` is blocking end-to-end. A streaming server/client setup is more suitable for measuring user-visible time to first token (TTFT) and inter-token latency (ITL).

## Keep These Responsibilities Separate

| Component | Main responsibility |
|---|---|
| Model | Defines the computation |
| Scheduler | Chooses which request work runs next |
| KV-cache manager | Manages the attention state and memory blocks |
| Model runner | Converts scheduled work into GPU execution |

This separation is one of the most useful mental models for reading the codebase.

## How to Make Progress Without Being Spoon-Fed

For each code area, produce a small artifact rather than trying to memorize it:

- An object graph
- A short call graph
- A request-state diagram
- A diagram of KV-block allocation
- A trace of one token through the engine

Use this rule: **understand the current function's contract before descending into its implementation details.**

## First Task

Start at `LLM.generate()` and trace downward until you find the engine-step loop. Bring back the call graph and the functions where ownership or control flow becomes unclear. Work through those questions one at a time.

The first-principles question to keep returning to is:

**What state must the runtime maintain between forward passes that a normal `nn.Module` does not need to manage?**
