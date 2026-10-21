'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '../..');
const bundlePath = process.argv[2];
if (!bundlePath) {
  process.stderr.write('usage: node product_wasm_embedded_modules_test.cjs <product-bundle.js>\n');
  process.exit(2);
}

async function main() {
  const lines = [];
  const errors = [];
  const bytes = [];
  const mod = await require(path.resolve(bundlePath))({
    print: text => lines.push(String(text)),
    printErr: text => errors.push(String(text)),
    stdout: byte => { if (byte !== undefined && byte !== null) bytes.push(byte); },
    noExitRuntime: true,
  });
  const batch = mod.cwrap('run_program', null, ['string']);
  const evalFn = mod.cwrap('repl_eval', 'string', ['string']);
  const flush = () => {
    try { mod.ccall('fflush', 'number', ['number'], [0]); } catch (_) { /* optional */ }
  };
  function capture(fn) {
    const lineStart = lines.length;
    const errorStart = errors.length;
    const byteStart = bytes.length;
    try { fn(); } catch (error) {
      if (!(error && error.name === 'ExitStatus' && error.status === 0)) throw error;
    }
    flush();
    return {
      lines: lines.slice(lineStart),
      stderr: errors.slice(errorStart),
      stdout: Buffer.from(bytes.slice(byteStart)),
    };
  }
  const runFile = relative => capture(() => batch(fs.readFileSync(path.join(root, relative), 'utf8')));

  const msgpack = runFile('tests/v1_2_edge_cases/msgpack_test.esk');
  assert.match(msgpack.stdout.toString(), /Passed: 42/);
  assert.match(msgpack.stdout.toString(), /Failed: 0/);
  assert.match(msgpack.stdout.toString(), /RESULT: OK/);
  assert.deepEqual(msgpack.stderr, []);

  const utf8 = runFile('tests/v1_2_edge_cases/msgpack_utf8_test.esk');
  assert.match(utf8.stdout.toString(), /MessagePack UTF-8 80 checks/);
  assert.deepEqual(utf8.stderr, []);

  const reset = runFile('tests/vm/logic_registry_reset_vm_regression.esk');
  assert.match(reset.stdout.toString(), /RESULT: PASS/);
  assert.deepEqual(reset.stderr, []);

  const bytevector = runFile('tests/vm/bytevector_constructor_regression.esk');
  assert.match(bytevector.stdout.toString(), /RESULT: PASS/);
  assert.deepEqual(bytevector.stderr, []);

  const aliases = capture(() => batch(`
    (import (only (core.msgpack) msgpack-null?))
    (import (rename (core.msgpack) (msgpack-null make-msgpack-null)))
    (require core.msgpack)
    (require core.msgpack)
    (display (if (msgpack-null? (make-msgpack-null)) "module-imports-ok" "module-imports-failed"))
    (newline)
  `));
  assert.equal(aliases.stdout.toString(), 'module-imports-ok\n');
  assert.deepEqual(aliases.stderr, []);

  const missing = capture(() => batch(`
    (display "visible-prefix")
    (require core.not-embedded)
    (display "must-not-run")
  `));
  assert.equal(missing.stdout.length, 0, 'batch missing-module refusal executed a visible prefix');
  assert.match(missing.stderr.join('\n'), /module source not found: core\.not-embedded/);

  function evaluate(source) { return capture(() => evalFn(source)); }
  evaluate('(define retained 17)');
  evaluate('(define retained-closure (lambda (x) (+ retained x)))');
  evaluate('(define (preserved-named x) (+ x 10))');

  const failedImport = evaluate(`
    (display "visible-prefix")
    (define (preserved-named x) (+ x 100))
    (define (new-procedure x) (+ x 1))
    (require core.not-embedded)
  `);
  assert.match(failedImport.stderr.join('\n'), /module source not found: core\.not-embedded/);
  assert.equal(failedImport.stdout.toString(), 'Error during evaluation\n', 'failed REPL import ran an earlier effect');
  assert.equal(evaluate('retained').stdout.toString(), '17\n');
  assert.equal(evaluate('(retained-closure 5)').stdout.toString(), '22\n');
  assert.equal(evaluate('(preserved-named 5)').stdout.toString(), '15\n', 'failed import committed a named-procedure redefinition');
  const absentNewProcedure = evaluate('new-procedure');
  assert.match(absentNewProcedure.stderr.join('\n'), /undefined variable 'new-procedure'/);
  const recoveredDefinition = evaluate('(define recovered-after-failure 41)');
  assert.deepEqual(recoveredDefinition.stderr, [], 'the session remained poisoned after rollback');
  assert.equal(evaluate('recovered-after-failure').stdout.toString(), '41\n');

  process.stdout.write('product WASM embedded modules: PASS (42 + 80 checks, reset, constructor, aliases, refusal, rollback)\n');
}

main().catch(error => {
  process.stderr.write(`${error.stack || String(error)}\n`);
  process.exitCode = 1;
});
