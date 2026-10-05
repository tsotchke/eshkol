#!/usr/bin/env python3
"""Check or render source-reviewed example guides; never execute an example."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile


class CatalogueError(ValueError):
    pass


FAMILIES = ("ESHKOL_NS_EXAMPLES", "ESHKOL_IPM_EXAMPLES")
FIELDS = ("title", "purpose", "algorithm", "domain", "arithmetic", "validation", "limitations", "prerequisites", "literature_review")
AI_NAMES = {"mathematics_jacobian_counterexample", "mathematics_alphatensor_gf2", "mathematics_alphatensor_3x3_gf2", "mathematics_funsearch_cap_set"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CatalogueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def inventory(root):
    result = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", "examples"], capture_output=True)
    if result.returncode:
        raise CatalogueError("Git example inventory failed; no outputs may be written")
    try:
        paths = sorted(p for p in result.stdout.decode("utf-8").split("\0") if p.endswith(".esk"))
    except UnicodeError as exc:
        raise CatalogueError("Git inventory is not UTF-8") from exc
    if not paths or len(set(paths)) != len(paths):
        raise CatalogueError("Git example inventory is empty or duplicated; refusing to write")
    for path in paths:
        if not re.fullmatch(r"examples/(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+\.esk", path):
            raise CatalogueError(f"unsupported example path: {path}")
        if not (root / path).is_file() or (root / path).is_symlink():
            raise CatalogueError(f"tracked example missing or symlinked: {path}")
    return paths


def registration_matrix(root, paths):
    text = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    groups = {}
    for family in FAMILIES:
        matches = list(re.finditer(rf"\bset\({family}\s+([^)]*)\)", text))
        if len(matches) != 1:
            raise CatalogueError(f"expected one nonempty registration list: {family}")
        tokens = re.sub(r"#[^\n]*", "", matches[0][1]).split()
        if not tokens or len(tokens) % 2:
            raise CatalogueError(f"empty or odd criterion/source list: {family}")
        prior_if = text.rfind("if(", 0, matches[0].start())
        if prior_if < 0 or not text[prior_if:].startswith("if(ESHKOL_BUILD_TESTS AND TARGET eshkol-run)"):
            raise CatalogueError(f"authored build/target condition changed: {family}")
        prefix = "ns" if family == "ESHKOL_NS_EXAMPLES" else "ipm"
        start = matches[0].end()
        loop_end = text.find("endforeach()", start)
        block = text[start:loop_end] if loop_end >= 0 else ""
        for mode in ("jit", "aot"):
            if block.count(f"add_test(NAME ${{_{prefix}_name}}_{mode}") != 1:
                raise CatalogueError(f"missing/duplicate authored {mode} registration for {family}")
        if f'"${{CMAKE_CURRENT_SOURCE_DIR}}/examples/${{_{prefix}_file}}.esk"' not in block or 'PASS_REGULAR_EXPRESSION "RESULT: ALL PASS"' not in block:
            raise CatalogueError(f"unrecognized source/verdict registration contract: {family}")
        if f'COMMAND $<TARGET_FILE:eshkol-run> -r "${{_{prefix}_src}}"' not in block or f"'${{_{prefix}_src}}' && '${{CMAKE_CURRENT_BINARY_DIR}}/${{_{prefix}_file}}_aot'" not in block:
            raise CatalogueError(f"native JIT/AOT command routing changed: {family}")
        if f'list(GET {family} ${{_{prefix}_i}} _{prefix}_name)' not in block or f'list(GET {family} ${{_{prefix}_j}} _{prefix}_file)' not in block:
            raise CatalogueError(f"criterion/source routing changed: {family}")
        rows, names = [], set()
        for criterion, stem in zip(tokens[::2], tokens[1::2]):
            source = f"examples/{stem}.esk"
            if not re.fullmatch(r"[A-Za-z0-9_]+", criterion) or criterion in names or source not in paths:
                raise CatalogueError(f"invalid, duplicate or untracked registration: {criterion}/{source}")
            names.add(criterion)
            rows.append({"criterion": criterion, "source": source, "modes": ["jit", "aot"]})
        groups[family] = {"criteria": len(rows), "distinct_sources": len({r["source"] for r in rows}), "ctest_entries": 2 * len(rows), "rows": rows}
    names = [r["criterion"] for group in groups.values() for r in group["rows"]]
    if len(set(names)) != len(names):
        raise CatalogueError("criterion is registered in competing families")
    return groups


def load_catalogue(root, path, paths, matrix):
    try:
        catalogue = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    except (OSError, ValueError) as exc:
        raise CatalogueError(f"invalid catalogue: {exc}") from exc
    if not isinstance(catalogue, dict) or catalogue.get("schema") != "eshkol.example-catalogue.v1" or not isinstance(catalogue.get("entries"), list):
        raise CatalogueError("unsupported catalogue schema or entries")
    entries, seen = catalogue["entries"], set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("path") not in paths or entry["path"] in seen:
            raise CatalogueError("catalogue has unknown or duplicate example")
        seen.add(entry["path"])
        if any(not isinstance(entry.get(field), str) or not entry[field].strip() for field in FIELDS):
            raise CatalogueError(f"missing reviewed semantics: {entry['path']}")
        if entry.get("review_status") != "source-inspected" or entry.get("kind") not in {"mathematics", "example"}:
            raise CatalogueError("invalid review/kind status")
        if entry["kind"] != ("mathematics" if Path(entry["path"]).name.startswith("mathematics_") else "example"):
            raise CatalogueError("mathematics inventory classification drift")
        if entry.get("execution_evidence") != "No execution performed for this catalogue; checks and registrations describe source intent.":
            raise CatalogueError("catalogue cannot invent execution evidence")
        source = root / entry["path"]
        if entry.get("source_sha256") != digest(source):
            raise CatalogueError(f"source changed; semantic review required before updating fingerprint: {entry['path']}")
        lines = source.read_text(encoding="utf-8").splitlines()
        anchors = entry.get("source_anchors")
        if not isinstance(anchors, list) or len(anchors) < 2:
            raise CatalogueError("implementation/validation source anchors required")
        for anchor in anchors:
            if (not isinstance(anchor, dict) or type(anchor.get("line")) is not int or not 1 <= anchor["line"] <= len(lines)
                    or anchor.get("text") != lines[anchor["line"] - 1]):
                raise CatalogueError(f"source anchor changed: {entry['path']}")
        expected = [{"family": family, "criterion": row["criterion"], "modes": row["modes"]}
                    for family, group in matrix.items() for row in group["rows"] if row["source"] == entry["path"]]
        if entry.get("registrations") != expected:
            raise CatalogueError(f"catalogue/registration matrix mismatch: {entry['path']}")
        if any(r["family"] == "ESHKOL_NS_EXAMPLES" for r in expected) and not isinstance(entry.get("paper_sections"), str):
            raise CatalogueError("Navier–Stokes source sections must be curated")
    if seen != set(paths):
        raise CatalogueError("catalogue omits tracked examples: " + ", ".join(sorted(set(paths) - seen)))
    runner = catalogue.get("runner_contract")
    if not isinstance(runner, dict) or runner.get("path") != "scripts/run_examples_tests.sh" or runner.get("source_sha256") != digest(root / runner["path"]):
        raise CatalogueError("general examples runner changed; review scope/skip policy")
    text = (root / runner["path"]).read_text()
    if (runner.get("discovery") != "examples/*.esk" or runner.get("mode") != "aot"
            or 'for test_file in examples/*.esk; do' not in text or '-L./"$BUILD_DIR" "$test_file" -o "$ESHKOL_TEST_BIN"' not in text):
        raise CatalogueError("invalid general runner discovery/mode")
    skip = text.split("example_should_skip() {", 1)[-1].split("print_empty_examples_summary()", 1)[0]
    quantum = re.search(r"\n\s*([a-z0-9_.|]+)\)\s*\n\s*if \[.*ESHKOL_QUANTUM_ENABLED", skip)
    excluded = re.search(r"\n\s*(selene_[^\n]+)\)\s*\n\s*return 0", skip)
    if not quantum or not excluded or runner.get("quantum_sources") != ["examples/" + p for p in quantum[1].split("|")] or runner.get("excluded_patterns") != excluded[1].split("|") or runner.get("quantum_condition") != "ESHKOL_QUANTUM_ENABLED=ON":
        raise CatalogueError("general runner conditional/exclusion scope disagrees with review")
    return catalogue


def anchor(entry):
    return Path(entry["path"]).stem.replace("_", "-")


def source_link(entry, line=1):
    return f"[{Path(entry['path']).name}](../{entry['path']}#L{line})"


def manual(entry):
    if entry["path"] == "examples/wgsl_artifact/generate.esk":
        return "python3 examples/wgsl_artifact/build_artifact.py"
    source = shlex.quote(entry["path"])
    output = shlex.quote(".scratch/example-manual/" + Path(entry["path"]).stem)
    return f"./build/eshkol-run -r {source}\n./build/eshkol-run -L./build {source} -o {output} && {output}"


def entry_text(entry):
    fields = (('Purpose', 'purpose'), ('Implemented algorithm', 'algorithm'), ('Domain and parameters', 'domain'),
              ('Arithmetic', 'arithmetic'), ('Checks in the source', 'validation'), ('Limits', 'limitations'), ('Prerequisites', 'prerequisites'))
    rows = [f'<a id="{anchor(entry)}"></a>', f"## {entry['title']}", f"Source: {source_link(entry)}. SHA-256: `{entry['source_sha256']}`."]
    rows += [f"**{label}:** {entry[key]}" for label, key in fields]
    refs = ", ".join(f"[line {a['line']}](../{entry['path']}#L{a['line']})" for a in entry['source_anchors'])
    rows += [f"Implementation/check anchors: {refs}.",
             "Manual commands from the repository root, after satisfying the prerequisites:", "```bash\n" + manual(entry) + "\n```"]
    return "\n\n".join(rows)


def marker(text, name, content):
    start, end = f"<!-- example-catalogue:{name}:start -->", f"<!-- example-catalogue:{name}:end -->"
    if text.count(start) != 1 or text.count(end) != 1 or text.index(start) >= text.index(end):
        raise CatalogueError(f"missing, duplicate or inverted generated markers: {name}")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    return before + start + "\n" + content.rstrip() + "\n" + end + after


def render(root, catalogue, matrix):
    entries = sorted(catalogue['entries'], key=lambda e:e['path'])
    math = [e for e in entries if e['kind'] == 'mathematics']
    flat = [e for e in entries if len(Path(e['path']).parts) == 2]
    runner = catalogue['runner_contract']
    excluded = [e for e in flat if any(fnmatch.fnmatch(Path(e['path']).name, pattern) for pattern in runner['excluded_patterns'])]
    quantum = [e for e in flat if e['path'] in runner['quantum_sources']]
    index = ["# Example catalogue", "<!-- Generated by scripts/build_example_catalogue.py from reviewed docs/examples/catalogue.json. -->",
        f"This catalogue covers **{len(entries)} tracked programs**, including **{len(math)} mathematics programs**. It documents source behavior and authored test registrations; it records no new execution results.",
        "The [mathematics guide](MATHEMATICS_EXAMPLES.md) gives each mathematical program’s algorithm, domain, arithmetic, checks and limits. The [Navier–Stokes guide](NAVIER_STOKES_EXAMPLES.md) and [AI mathematics guide](AI_MATHEMATICS_EXAMPLES.md) retain their explanatory narratives and references.",
        f"The general [examples runner](../scripts/run_examples_tests.sh) discovers **{len(flat)} flat programs** and compiles/runs them with native AOT. **{len(quantum)}** require `ESHKOL_QUANTUM_ENABLED=ON`; **{len(excluded)}** current paths match the existing proprietary/unreleased exclusion patterns. The **{len(entries)-len(flat)} nested artifact generator** is outside that glob. Discovery and a successful process are distinct from an asserted mathematical verdict.",
        "The NS/IPM CMake lists declare native JIT/AOT tests under `ESHKOL_BUILD_TESTS AND TARGET eshkol-run`; this matrix does not claim that any particular build configured, ran or passed them, or that every example is supported by the bytecode VM.",
        "For manual AOT commands below, create `.scratch/example-manual` once from the repository root.",
        "To check documentation drift: `python3 scripts/build_example_catalogue.py --check`. After deliberate source/semantic review, update the catalogue and run `--write`; the generator never updates fingerprints or executes examples.",
        "| Program | Reviewed purpose | Authored dedicated criteria | General AOT scope |\n|---|---|---|---|"]
    table_rows=[]
    for entry in entries:
        target = 'MATHEMATICS_EXAMPLES.md' if entry['kind']=='mathematics' else 'EXAMPLES.md'
        title = f"[{entry['title']}]({target}#{anchor(entry)})"
        criteria = ', '.join(f"`{r['criterion']}` (JIT/AOT)" for r in entry['registrations']) or 'No NS/IPM criterion'
        scope = 'Nested artifact pipeline' if entry not in flat else 'Conditional quantum lane' if entry in quantum else 'Declared exclusion' if entry in excluded else 'Flat discovery'
        table_rows.append(f"| {source_link(entry)} | {title} | {criteria} | {scope} |")
    index[-1] += '\n' + '\n'.join(table_rows)
    index += ['\n# Language and application examples', 'These descriptions report implemented checks or printed diagnostics; output prose is not treated as an assertion.']
    index += [entry_text(e) for e in entries if e['kind']!='mathematics']
    math_text = ['# Mathematics examples', '<!-- Generated by scripts/build_example_catalogue.py from reviewed docs/examples/catalogue.json. -->',
        f"**{len(math)} source-reviewed programs**. Exact finite constructions, bilinear/polynomial identity certificates, finite formal expansions and numerical comparisons have different scopes, stated per entry. No new execution is claimed.",
        "[All examples and run scope](EXAMPLES.md) · [Navier–Stokes narrative](NAVIER_STOKES_EXAMPLES.md) · [AI witness narrative](AI_MATHEMATICS_EXAMPLES.md)",
        '| Authored CMake family | Programs | Criteria | Native JIT/AOT entries |\n|---|---:|---:|---:|']
    matrix_rows=[]
    for family, group in matrix.items():
        matrix_rows.append(f"| `{family}` | {group['distinct_sources']} | {group['criteria']} | {group['ctest_entries']} |")
    math_text[-1] += '\n' + '\n'.join(matrix_rows)
    math_text += ['\nRegistration is conditional on `ESHKOL_BUILD_TESTS AND TARGET eshkol-run`. Repeated criteria for one program are counted separately; the localization source has three NS criteria. Runtime receipts remain the authority for measured outcomes.']
    math_text += [entry_text(e) for e in math]
    ns_entries = [e for e in entries if any(r['family']=='ESHKOL_NS_EXAMPLES' for r in e['registrations'])]
    ns = (root / 'docs/NAVIER_STOKES_EXAMPLES.md').read_text()
    group = matrix['ESHKOL_NS_EXAMPLES']
    summary = f"The authored matrix contains **{group['distinct_sources']} programs, {group['criteria']} criteria and {group['ctest_entries']} CTest entries**, each with native JIT and AOT variants, under `ESHKOL_BUILD_TESTS AND TARGET eshkol-run`. Localization supplies three criteria. The actual run supplies outcomes and timing; this documentation update reports source inspection only."
    table = ['| Program | Paper sections | Implemented calculation | CTest criteria |','|---|---|---|---|']
    for entry in ns_entries:
        criteria=', '.join('`'+r['criterion']+'`' for r in entry['registrations'])
        table.append(f"| {source_link(entry)} | {entry['paper_sections']} | {entry['algorithm']} | {criteria} |")
    commands = 'Prerequisites and scope are stated in the [complete mathematics catalogue](MATHEMATICS_EXAMPLES.md). Run from the repository root after building the compiler and stdlib.\n\n```bash\nmkdir -p .scratch/example-manual\n' + '\n\n'.join(manual(e) for e in ns_entries) + "\n\nctest --test-dir build --output-on-failure -R '^ns_'\n```\n\nThe criterion table above declares `_jit` and `_aot` variants; it is not a receipt that either ran."
    ns = marker(marker(marker(ns,'ns-summary',summary),'ns-table','\n'.join(table)),'ns-commands',commands)
    ai=(root/'docs/AI_MATHEMATICS_EXAMPLES.md').read_text()
    ai_entries=[e for e in entries if Path(e['path']).stem in AI_NAMES]
    ai_block=f"The source catalogue contains **{len(ai_entries)} public witness programs** in this family. The lane status above records earlier family verification; this source-documentation update adds no execution receipt. Each complete description separates exact identity checks from numerical AD comparisons.\n\n"+'\n'.join(f"- [{e['title']}](MATHEMATICS_EXAMPLES.md#{anchor(e)}): {source_link(e)}." for e in ai_entries)+'\n\n```bash\nmkdir -p .scratch/example-manual\n'+'\n\n'.join(manual(e) for e in ai_entries)+'\n```'
    ai=marker(ai,'ai-inventory',ai_block)
    readme=(root/'examples/README.md').read_text()
    overview=[f"The source-reviewed catalogue covers **{len(entries)} programs**, including **{len(math)} mathematics programs**. Start with the [complete guide](../docs/EXAMPLES.md) or the [mathematics guide](../docs/MATHEMATICS_EXAMPLES.md) for algorithms, domains, arithmetic, checks, limits and per-program commands.",
        f"The general runner discovers **{len(flat)} flat sources** for native AOT, with **{len(quantum)} quantum-conditional programs** and the existing declared exclusions. The nested [WGSL generator](wgsl_artifact/README.md) follows its artifact pipeline. Dedicated mathematics registration contains NS **{matrix['ESHKOL_NS_EXAMPLES']['distinct_sources']}/{matrix['ESHKOL_NS_EXAMPLES']['criteria']}/{matrix['ESHKOL_NS_EXAMPLES']['ctest_entries']}** and IPM **{matrix['ESHKOL_IPM_EXAMPLES']['distinct_sources']}/{matrix['ESHKOL_IPM_EXAMPLES']['criteria']}/{matrix['ESHKOL_IPM_EXAMPLES']['ctest_entries']}** programs/criteria/JIT-AOT entries, under the authored build condition.",
        "These are source and registration facts, not a new execution receipt. Each entry distinguishes asserted checks from printed diagnostics. Runtime, convergence, backend dispatch and cross-host byte equality require their own measured evidence.",
        "Run from the repository root with an already built compiler and stdlib, as described in the [Quickstart](../docs/QUICKSTART.md):\n\n```bash\nmkdir -p .scratch/example-manual\n./build/eshkol-run -r examples/hello.esk\n./build/eshkol-run -L./build examples/hello.esk -o .scratch/example-manual/hello\n.scratch/example-manual/hello\n```",
        "| Program | Source-backed purpose | Detailed guide |\n|---|---|---|"]
    rows=[]
    for entry in entries:
        local=Path(entry['path']).relative_to('examples').as_posix()
        guide='MATHEMATICS_EXAMPLES.md' if entry['kind']=='mathematics' else 'EXAMPLES.md'
        rows.append(f"| [{local}]({local}#L1) | {entry['purpose']} | [{entry['title']}](../docs/{guide}#{anchor(entry)}) |")
    overview[-1]+='\n'+'\n'.join(rows)
    readme=marker(readme,'examples-overview','\n\n'.join(overview))
    data={'schema':'eshkol.example-test-matrix.v1','authored_condition':'ESHKOL_BUILD_TESTS AND TARGET eshkol-run','programs':len(entries),'mathematics_programs':len(math),'families':matrix,'general_runner':{'discovery':runner['discovery'],'mode':runner['mode'],'flat_programs':len(flat),'quantum_sources':runner['quantum_sources'],'excluded_patterns':runner['excluded_patterns'],'current_excluded_sources':[e['path'] for e in excluded],'nested_sources':[e['path'] for e in entries if e not in flat]},'execution_claim':'No new execution; source inventory and authored registrations only.'}
    return {'docs/EXAMPLES.md':'\n\n'.join(index)+'\n','docs/MATHEMATICS_EXAMPLES.md':'\n\n'.join(math_text)+'\n','docs/NAVIER_STOKES_EXAMPLES.md':ns,'docs/AI_MATHEMATICS_EXAMPLES.md':ai,'docs/examples/test-matrix.json':json.dumps(data,indent=2,ensure_ascii=False)+'\n','examples/README.md':readme}


def verify_links(root, outputs):
    for relative, text in outputs.items():
        if not relative.endswith('.md'): continue
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
            if target.startswith(('http:', 'https:', 'mailto:', '#')): continue
            destination, _, fragment=target.partition('#')
            path=(root/relative).parent/destination
            try: path.resolve().relative_to(root.resolve())
            except ValueError as exc: raise CatalogueError(f'link escapes repository: {target}') from exc
            key=path.resolve().relative_to(root.resolve()).as_posix()
            if key not in outputs and not path.exists():
                raise CatalogueError(f'broken local link in {relative}: {target}')
            if fragment.startswith('L') and fragment[1:].isdigit():
                if not path.is_file() or not 1 <= int(fragment[1:]) <= len(path.read_text().splitlines()):
                    raise CatalogueError(f'broken source line anchor: {target}')
            elif fragment and key in outputs and f'id="{fragment}"' not in outputs[key]:
                raise CatalogueError(f'broken generated section anchor: {target}')


def build(root, write=False):
    root=Path(root).resolve()
    paths=inventory(root)
    matrix=registration_matrix(root,paths)
    catalogue=load_catalogue(root,root/'docs/examples/catalogue.json',paths,matrix)
    outputs=render(root,catalogue,matrix)
    verify_links(root,outputs)
    changed=[name for name,text in outputs.items() if not (root/name).is_file() or (root/name).read_text()!=text]
    if changed and not write:
        raise CatalogueError('generated documentation drift: '+', '.join(changed))
    # Every read, semantic/matrix check and link check finishes before writes.
    for name in changed:
        destination=root/name
        if destination.is_symlink(): raise CatalogueError('generated destination cannot be a symlink')
    for name in changed:
        destination=root/name; destination.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=destination.parent,prefix='.'+destination.name+'.',delete=False) as handle:
            handle.write(outputs[name]); temporary=Path(handle.name)
        temporary.replace(destination)
    return {'programs':len(paths),'mathematics':sum(e['kind']=='mathematics' for e in catalogue['entries']),'changed':changed}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--write',action='store_true')
    mode.add_argument('--check',action='store_true')
    args=parser.parse_args()
    try:
        result=build(args.root,args.write)
    except (CatalogueError,OSError,UnicodeError,ValueError) as exc:
        print(f'example catalogue: FAIL: {exc}',file=sys.stderr);return 1
    print('example catalogue: PASS '+json.dumps(result,sort_keys=True));return 0


if __name__=='__main__':
    raise SystemExit(main())
