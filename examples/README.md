# Eshkol Examples

Run any example with:
```bash
./build/eshkol-run examples/hello.esk
```

Or compile to a standalone binary:
```bash
./build/eshkol-run examples/autodiff.esk -o autodiff
./autodiff
```

---

## Getting Started

| Example | What It Shows |
|---------|--------------|
| **[hello.esk](hello.esk)** | Basic output — the simplest Eshkol program |
| **[autodiff.esk](autodiff.esk)** | Forward and reverse-mode automatic differentiation — Eshkol's signature feature |
| **[tensors.esk](tensors.esk)** | Matrix creation, multiplication, reductions — dispatches to BLAS/GPU automatically |
| **[parallel.esk](parallel.esk)** | Parallel execution, futures, thread pool — multi-core computation |
| **[consciousness.esk](consciousness.esk)** | Knowledge base, factor graphs, global workspace — the 22-primitive consciousness engine |

## Advanced

These examples demonstrate research-grade capabilities and may require additional libraries.

| Example | What It Shows |
|---------|--------------|
| **[agent.esk](agent.esk)** | Agent architecture with tool dispatch and knowledge base |
| **[consciousness_inference.esk](consciousness_inference.esk)** | Active inference with factor graphs and belief propagation |
| **[consciousness_model_analysis.esk](consciousness_model_analysis.esk)** | Geometric signal analysis for consciousness models |

## What to Try First

1. **Start here**: `hello.esk` — verify your installation works
2. **The killer feature**: `autodiff.esk` — see the compiler differentiate functions
3. **Real computing**: `tensors.esk` — matrix operations with automatic GPU dispatch
4. **AI primitives**: `consciousness.esk` — logic programming and probabilistic inference

## Try in the Browser

Visit **[eshkol.ai](https://eshkol.ai)** to run Eshkol code without installing anything. The website includes a REPL, 8 interactive tutorials, and 10 runnable examples.
