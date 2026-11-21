"""Source bindings and native fault tests for BLC heap replay."""
import re
import shlex
import shutil
import sys
import subprocess

HEAP_TAGS = {"START", "ALLOC", "FORCE", "LEVEL", "RETURN", "WRITE", "PUBLISH", "HIT", "STOP"}


# Keep whitespace for source-to-source instrumentation; the arm readers discard it.
SCHEME_LEX = r'''"(?:\\.|[^"\\])*"|;[^\n]*|[()']|\s+|[^\s();']+'''


def tokens(text):
    return [t for t in re.findall(SCHEME_LEX, text)
            if not t.isspace() and not t.startswith(";")]


def source_program(source):
    # Read only this three-instruction arm. Ignore comments, preserve strings
    # as tokens, and require the addressing/return bindings around the arm.
    header = tokens("""((eq? mode 'value)
      (let* ((frame (car frames)) (action (car frame))
             (data (cdr frame)) (rest (cdr frames)))
        (cond ((eq? action 'update)""")
    stream = tokens(source)
    starts = [i + len(header) for i in range(len(stream)) if stream[i:i+len(header)] == header]
    if len(starts) != 1:
        raise ValueError("ambiguous or changed update-frame binding")
    cursor = starts[0]
    ops = []
    forms = [(tokens("(set-cdr! data item)"), ".writeCdr"),
             (tokens("(set-car! data 'memo)"), ".tagMemo"),
             (tokens("(loop 'value item '() rest (+ steps 1))"), ".tailValue")]
    while cursor < len(stream) and stream[cursor] != ")":
        for form, op in forms:
            if stream[cursor:cursor + len(form)] == form:
                ops.append(op)
                cursor += len(form)
                break
        else:
            raise ValueError("unsupported instruction in publication arm")
    if cursor == len(stream):
        raise ValueError("unterminated publication arm")
    return "[" + ", ".join(ops) + "]"


def lookup_program(source):
    stream = tokens(source)
    header = tokens("((eq? mode 'lookup) (cond")
    positions = [i + len(header) for i in range(len(stream)) if stream[i:i+len(header)] == header]
    if len(positions) != 1:
        raise ValueError("ambiguous or changed lookup dispatch")
    cursor = positions[0]

    def consume(form):
        nonlocal cursor
        seq = tokens(form)
        if stream[cursor:cursor + len(seq)] != seq:
            raise ValueError("unsupported lookup guard or branch boundary")
        cursor += len(seq)

    forms = [(tokens("(loop 'value (- 0 item) '() frames (+ steps 1))"), ".freeHead"),
             (tokens("(loop 'force (car env) '() frames (+ steps 1))"), ".forceCar"),
             (tokens("(loop 'lookup (- item 1) (cdr env) frames (+ steps 1))"), ".lookupCdr")]
    ops = []
    for guard in ("((null? env)", "((= item 1)", "(else"):
        consume(guard)
        for form, op in forms:
            if stream[cursor:cursor + len(form)] == form:
                ops.append(op)
                cursor += len(form)
                break
        else:
            raise ValueError("unsupported lookup tail call")
        consume(")")
    consume("))")
    return "⟨" + ", ".join(ops) + "⟩"


def check_source(work, env, lean, execute, source):
    bound = work / "Bound.lean"
    bound.write_text("import Publication\nimport Lookup\nopen Verificatum.Publication\n"
                     f"example : ({source_program(source)} : List Op) = updateArm := by decide\n"
                     f"example : ({lookup_program(source)} : Verificatum.Lookup.Arm) "
                     "= Verificatum.Lookup.lookupArm := by decide\n")
    execute([lean, bound.name], work, env)
    # Both alternatives are legal tail calls. Substituting one must reach
    # Lean and fail its program equality, even for calls returning equal values.
    wrong = source.replace("(loop 'force (car env) '() frames (+ steps 1))",
                           "(loop 'lookup (- item 1) (cdr env) frames (+ steps 1))")
    mutation = work / "WrongLookup.lean"
    mutation.write_text("import Lookup\nopen Verificatum.Lookup\n"
                        f"example : ({lookup_program(wrong)} : Arm) = lookupArm := by decide\n")
    must_reject([lean, mutation.name], env, work, "is false")


def heap_rows(output):
    return [row for row in output.splitlines() if row.split(" ", 1)[0] in HEAP_TAGS]

def observe_pair_stores(source):
    """Intercept the two pair mutators in the test module before compiling it."""
    renames = {"set-cdr!": "audit-raw-cdr!", "set-car!": "audit-raw-car!"}
    # Match whole Scheme tokens and preserve comments and strings. Insert the
    # wrappers afterwards so their calls still resolve to the runtime primitives.
    body = re.sub(SCHEME_LEX, lambda m: renames.get(m[0], m[0]), source)
    wrappers = """
(define (audit-raw-cdr! pair payload)
  (set-cdr! pair payload)
  (audit-value-event "RAW-WRITE" pair (cdr pair)))
(define (audit-raw-car! pair tag)
  (set-car! pair tag)
  (if (not (eq? (car pair) 'memo)) (error "unexpected thunk tag after store"))
  (audit-value-event "RAW-PUBLISH" pair (cdr pair)))
"""
    site = "(define (audit-write! destination returned)"
    if body.count(site) != 1:
        raise ValueError("missing pair observer insertion site")
    return body.replace(site, wrappers + site)


def check_store_observations(output):
    """Each observed pair mutation must have one matching publication record."""
    pending = None
    stores = 0
    for line in output.splitlines():
        if line.startswith(("RAW-WRITE ", "RAW-PUBLISH ")):
            if pending is not None:
                raise ValueError(f"unreported pair store: {pending} before {line}")
            pending = line.removeprefix("RAW-")
        elif line.startswith(("WRITE ", "PUBLISH ")):
            if line != pending:
                raise ValueError(f"publication record {line} has no matching pair store")
            pending = None
            stores += 1
        elif pending is not None:
            raise ValueError(f"unreported pair store: {pending} before {line}")
    if pending is not None:
        raise ValueError(f"unreported pair store at end: {pending}")
    return stores


def cross_nested_return(rows):
    """Cross two active destinations while preserving the returned value."""
    pending = []
    deepest = 0
    altered = None
    for index, row in enumerate(rows):
        fields = row.split()
        if fields[0] == "START":
            pending = []
        elif fields[0] == "FORCE":
            pending.append(fields[1])
            deepest = max(deepest, len(pending))
        elif fields[0] == "RETURN":
            if not pending or fields[1] != pending[-1]:
                raise ValueError("native return did not match the top update frame")
            if altered is None and len(pending) > 1 and pending[-2] != pending[-1]:
                altered = rows.copy()
                altered[index] = f"RETURN {pending[-2]} " + row.split(" ", 2)[2]
        elif fields[0] == "PUBLISH":
            if not pending or pending.pop() != fields[1]:
                raise ValueError("publication did not discharge its update frame")
        elif fields[:2] == ["STOP", "normal"] and pending:
            raise ValueError("completed run retained pending frames")
    if altered is None:
        raise ValueError("no distinct nested frames tested")
    return altered, deepest


def must_reject(args, env, work, reason):
    r = subprocess.run(args, cwd=work, env=env, capture_output=True, text=True, timeout=120)
    if r.returncode == 0 or reason not in r.stdout + r.stderr:
        raise ValueError(f"expected rejection at {reason}: {r.stdout[-500:]} {r.stderr[-500:]}")


# Only the fault binary can redirect a store or raise an exception between
# stores. The ordinary traced binary always executes both writes to 'data'.
HOOKS = """
(define audit-fault (get-environment-variable "BLC_FAULT"))
(define audit-first #f)
(define audit-abandoned #f)
(define audit-written #f)
(define (audit-allocate term env)
  (if (and (equal? audit-fault "recycle") audit-first)
      audit-first
      (cons 'pending (cons term env))))
(define (audit-track cell)
  (if (= 0 (audit-id cell)) (set! audit-first cell))
  (if (eq? cell audit-abandoned) (error "fresh run reused a retained pair")))
(define (audit-abandoned-pending?)
  (and audit-abandoned (eq? (car audit-abandoned) 'pending)
       (eq? (cdr audit-abandoned) audit-written)))
(define (audit-destination requested)
  (if (and (equal? audit-fault "cross") (= 1 (audit-id requested)))
      (begin
        (if (or (eq? requested audit-first) (not (eq? (car audit-first) 'pending)))
            (error "fault requires a second live pending cell"))
        audit-first)
      requested))
(define (audit-cut target)
  (if (and (equal? audit-fault "interrupt") (not audit-abandoned)
           (= 1 (audit-id target)))
      (begin
        (set! audit-abandoned target)
        (set! audit-written (cdr target))
        (if (not (audit-abandoned-pending?)) (error "invalid publication cut"))
        (display "CUT") (newline)
        (error "test interruption after payload write"))))
"""


def check_faults(work, env, lean, runner, execute, module, source, here, runtime_probe=False, linked_refinement=False):
    def patch(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f"fault site changed: {old}")
        source = source.replace(old, new)

    patch("(provide blc-fast-run)", "(provide blc-fast-run audit-abandoned-pending?)")
    patch("(define (blc-fast-run term fuel)", HOOKS + "\n(define (blc-fast-run term fuel)")
    patch("(let ((cell (cons 'pending (cons (blc-app-arg item) env))))",
          "(let ((cell (audit-allocate (blc-app-arg item) env)))")
    patch("(audit-save cell) (set!", "(audit-save cell) (audit-track cell) (set!")
    # Bind the destination once, so both actual pair mutations hit the same
    # cell. RETURN still names the cell from the actual pending frame.
    patch("(audit-write! data item)\n              (audit-publish! data)",
          "(let ((target (audit-destination data))) "
          "(audit-write! target item) (audit-cut target) (audit-publish! target))")
    # Only the observer notification is conditional in the omission case.
    # The redundant-store mode also executes a second real, idempotent store.
    patch('(audit-value-event "WRITE" destination (cdr destination))', """
      (if (and (equal? audit-fault "extra-store") (= (audit-id destination) 1))
          (set-cdr! destination returned))
      (if (not (and (equal? audit-fault "omit-write") (= (audit-id destination) 1)))
          (audit-value-event "WRITE" destination (cdr destination)))""")
    module.write_text(observe_pair_stores(source))
    exe = work / "publication"
    execute([runner, "-I", work, "-O", "2", "-o", exe, here / "publication_test.esk"], work, env)
    event_file = work / "heap.events"
    audit = [lean, "Audit.lean"]
    normal = execute([exe], work, {**env, "BLC_FAULT": "healthy"})
    event_file.write_text("\n".join(heap_rows(normal)) + "\n")
    execute(audit, work, env)
    check_store_observations(normal)
    check_missing_write(work, env, lean, exe, execute, normal)
    if runtime_probe:
        check_runtime_probe(work, env, lean, runner, execute, here, normal, linked_refinement)
    must_reject([exe], {**env, "BLC_FAULT": "recycle"}, work, "duplicate live suspension identity")
    print("Reallocation of a still-live native pair rejected at ALLOC.")

    crossed = execute([exe], work, {**env, "BLC_FAULT": "cross"})
    rows = heap_rows(crossed)
    ret = next(i for i, row in enumerate(rows) if row.startswith("RETURN 1 "))
    shape = rows[ret].split(" ", 2)[2]
    if (rows[ret + 1:ret + 3] != [f"WRITE 0 {shape}", f"PUBLISH 0 {shape}"]
            or f"HIT 0 {shape}" not in rows[ret + 3:] or "IDENTITY" not in crossed.splitlines()):
        raise ValueError("fault did not redirect the actual write and reuse its payload")
    # The independent value-only oracle accepts each head, including B's.
    heads = [line.removeprefix("HEAD-") for line in crossed.splitlines()
             if line.startswith(("FILL ", "HEAD-HIT "))]
    head_file = work / "cross.heads"
    head_file.write_text("\n".join(heads) + "\n")
    execute([lean, "--run", here / "Lazy.lean", "--heads", head_file], work, env)
    event_file.write_text("\n".join(rows) + "\n")
    must_reject(audit, env, work, f'expected some "WRITE 1 {shape}", saw WRITE 0 {shape}')
    print("Native wrong-target write 1 → 0 returned identity and passed head checks; rejected at WRITE.")

    interrupted = execute([exe], work, {**env, "BLC_FAULT": "interrupt"})
    before, cut, after = interrupted.partition("CUT\n")
    if not cut or not all(x in after.splitlines() for x in ("CAUGHT", "IDENTITY", "ABANDONED-STILL-PENDING")):
        raise ValueError("interruption or same-process retry failed")
    partial = heap_rows(before)
    if not partial[-1].startswith("WRITE 1 ") or heap_rows(after) != heap_rows(normal):
        raise ValueError("retry did not reproduce the fresh invocation")
    event_file.write_text("\n".join(partial) + "\n")
    interrupt_check = work / "Cut.lean"
    interrupt_check.write_text('import Heap\n#eval Verificatum.Heap.auditHeap "heap.events" true\n')
    execute([lean, interrupt_check.name], work, env)
    must_reject(audit, env, work, "incomplete heap trace")
    for forbidden in ("FORCE 1", "HIT 1 " + partial[-1].split(" ", 2)[2]):
        event_file.write_text("\n".join(partial + [forbidden]) + "\n")
        must_reject(audit, env, work, 'expected some "PUBLISH 1 ')
    print("Interrupted write remained pending; premature force/hit rejected. Fresh retry matched healthy trace.")


def check_missing_write(work, env, lean, exe, execute, healthy):
    lines = healthy.splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith("WRITE 1 "))
    write = lines[at]
    raw = "RAW-" + write
    publish = write.replace("WRITE ", "PUBLISH ", 1)
    if lines[at-1:at+3] != [raw, write, "RAW-" + publish, publish]:
        raise ValueError("healthy test did not reach both pair stores")

    for fault in ("omit-write", "extra-store"):
        out = execute([exe], work, {**env, "BLC_FAULT": fault})
        # Match all other output including the independent readbacks, FILL,
        # final status, budget, normal form and the litmus driver's IDENTITY.
        delta = [] if fault == "omit-write" else [raw, write]
        if out.splitlines() != lines[:at] + delta + lines[at+1:]:
            raise ValueError(f"{fault} changed more than the selected store or notification")
        (work / "heap.events").write_text("\n".join(heap_rows(out)) + "\n")
        if fault == "omit-write":
            must_reject([lean, "Audit.lean"], env, work, f'expected some "{write}", saw {publish}')
        else:
            execute([lean, "Audit.lean"], work, env)
        (work / "observed.log").write_text(out)
        must_reject([lean, "stores.lean"], env, work, "unaccounted store")
        try:
            check_store_observations(out)
        except ValueError as err:
            if not str(err).startswith(f"unreported pair store: {write} before RAW-"):
                raise
        else:
            raise ValueError(f"{fault} escaped primitive store accounting")
    print("Native omitted WRITE rejected at PUBLISH. Extra silent store passed semantic replay and failed pair-store accounting.")


def check_runtime_probe(work, env, lean, runner, execute, here, control, linked_refinement):
    """Interpose only in a separately linked AOT binary for the native litmus."""
    root = here.parents[1]
    cxx = shutil.which("c++")
    if sys.platform != "linux" or not cxx:
        raise ValueError("runtime probe requires Linux ELF, a C++ compiler and GNU linker wrapping")
    hook = work / "store-probe.o"
    execute([cxx, "-O1", "-I", root, "-I", root / "inc", "-c", here / "runtime_store_probe.cpp",
             "-o", hook], work, env)
    driver = work / "probe-link"
    driver.write_text(f'#!/bin/sh\nexec {shlex.quote(cxx)} -Wl,--export-dynamic '
                      '-Wl,--wrap=arena_tagged_cons_set_tagged_value "$@"\n')
    driver.chmod(0o700)
    exe = work / "linked-publication"
    execute([runner, "-I", work, "-O", "2", "-o", exe, here / "publication_test.esk", hook],
            work, {**env, "ESHKOL_CXX_COMPILER": str(driver)})
    clean = {k: v for k, v in env.items() if k not in {"BLC_PROBE_DUPLICATE", "BLC_PROBE_DIRECT"}}
    clean["BLC_FAULT"] = "healthy"
    audit = work / "runtime.lean"
    audit.write_text('import Observation\n#eval Verificatum.Observation.audit ["runtime","observed.log"]\n')
    for duplicate in (False, True):
        out = execute([exe], work, {**clean, **({"BLC_PROBE_DUPLICATE": "1"} if duplicate else {})})
        expected = []
        for line in control.splitlines():
            if line.startswith("RAW-WRITE "):
                expected.extend(["RUNTIME CDR"] * (2 if duplicate else 1))
            elif line.startswith("RAW-PUBLISH "):
                expected.append("RUNTIME CAR")
            expected.append(line)
        if "RUNTIME CDR" not in expected or "RUNTIME CAR" not in expected or out.splitlines() != expected:
            raise ValueError("linked setter observations disagree with the native publication litmus")
        (work / "observed.log").write_text(out)
        (work / "heap.events").write_text("\n".join(heap_rows(out)) + "\n")
        execute([lean, "Audit.lean"], work, env)
        execute([lean, "stores.lean"], work, env)
        if duplicate:
            must_reject([lean, audit.name], env, work, "unaccounted runtime setter")
            calls = sum(line.startswith("RUNTIME ") for line in expected)
            print(f"Linked setter calls {calls}; "
                  "heap replay and Scheme store checks accepted, runtime accounting rejected.")
        else:
            print(execute([lean, audit.name], work, env).strip())

    if linked_refinement:
        from linked_pair import check_linked, check_direct_store
        check_linked(work, exe, lean, env, execute)
        check_direct_store(work, exe, lean, clean, execute, control, heap_rows)
