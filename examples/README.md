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

These examples exercise the production language surface beyond the basics.

| Example | What It Shows |
|---------|--------------|
| **[milli_mag_bohrification.esk](milli_mag_bohrification.esk)** | Physical-constants demonstration (v1.2.1-scale): CODATA-based magnetic-moment calculation through the exact numeric tower |

> A larger set of research-grade examples — agent harnesses (`agent.esk`, `selene_*.esk`) and consciousness-engine variants (`consciousness_inference.esk`, `consciousness_model_analysis.esk`, `consciousness_grr_inference.esk`) — lives in the development tree but is not part of the public release because each carries an evolving external dependency or experimental surface. The tutorials at [`../docs/tutorials/`](../docs/tutorials/) cover the same techniques in self-contained form (see Tutorial 04 for the consciousness engine and Tutorials 21–26 for project-scale examples).

## What to Try First

1. **Start here**: `hello.esk` — verify your installation works
2. **The killer feature**: `autodiff.esk` — see the compiler differentiate functions
3. **Real computing**: `tensors.esk` — matrix operations with automatic GPU dispatch
4. **AI primitives**: `consciousness.esk` — logic programming and probabilistic inference

## Try in the Browser

Visit **[eshkol.ai](https://eshkol.ai)** to run Eshkol code without installing anything. The website ships a browser REPL alongside an evolving subset of the tutorial collection (the repository hosts 29 tutorials at [`../docs/tutorials/`](../docs/tutorials/) and 6 public examples in this directory).
