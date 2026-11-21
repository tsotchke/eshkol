'use strict';

const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const {createHash} = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const root = path.resolve(__dirname, '../..');
const kernel = path.join(root, 'verification/lean-kernel');
const helper = path.join(root, 'lib/types/dependent.cpp');
const vmDir = path.join(root, 'site/static');
const digest = data => createHash('sha256').update(data).digest('hex');
const guardError = 'ERROR: unhandled exception: error: tensor operation: invalid shape or allocation limit';

function run(tool, args, cwd, extra = {}) {
  return execFileSync(tool, args, {cwd, encoding: 'utf8', timeout: 60000, ...extra});
}

function reference(work) {
  fs.copyFileSync(path.join(kernel, 'Obligation.lean'), path.join(work, 'Obligation.lean'));
  fs.copyFileSync(path.join(__dirname, 'ShapeCases.lean'), path.join(work, 'ShapeCases.lean'));
  run('lean', ['-DwarningAsError=true', '-o', 'Obligation.olean', 'Obligation.lean'], work);
  const corpus = run('lean', ['-DwarningAsError=true', '--run', 'ShapeCases.lean'], work,
                     {env: {...process.env, LEAN_PATH: work}});
  const sha256 = digest(corpus);
  assert.equal(sha256, 'f1f4eca50cafe7b55e97b28edbdeccc9d5ee17c17ba46bc56b117dd34cd7c3f0', 'changed corpus');
  const cases = corpus.trimEnd().split('\n').map(row => {
    assert.match(row, /^[1-4] [1-4] [1-4] [1-4] (incompatible|compatible)$/);
    const [m, k, j, n, verdict] = row.split(' ');
    return {dims: [m, k, j, n].map(Number), compatible: verdict === 'compatible'};
  });
  assert.equal(new Set(cases.map(c => c.dims.join(' '))).size, 4 ** 4);
  return {cases, sha256};
}

function probe(source, work, cases) {
  const binary = path.join(work, 'dimension-probe');
  run('c++', ['-std=c++17', '-ffunction-sections', '-fdata-sections', '-Wl,--gc-sections',
              '-I' + path.join(root, 'inc'), source, path.join(__dirname, 'dimension-probe.cpp'),
              '-o', binary], work);
  const input = cases.map(c => c.dims.join(' ')).join('\n') + '\n';
  const lines = run(binary, [], work, {input}).trimEnd().split('\n');
  assert.equal(lines.length, cases.length + 1);
  return lines;
}

function checkHelper(work, cases) {
  const observed = probe(helper, work, cases);
  assert.equal(observed.pop(), 'Cannot statically verify dimension equality: 3 =? ? in matmul (proof required)');
  for (const [i, {dims: [, k, j], compatible}] of cases.entries()) {
    assert.equal(observed[i], compatible ? 'compatible' : `Dimension mismatch: ${k} != ${j} in matmul`);
  }

  const source = fs.readFileSync(helper, 'utf8');
  const operand = 'const CTValue& right_rows = right.value_indices[0]';
  assert.equal(source.split(operand).length, 2, 'mutation site must be unique');
  const mutant = path.join(work, 'weakened.cpp');
  fs.writeFileSync(mutant, source.replace(operand, 'const CTValue& right_rows = left.value_indices[1]'));
  const weakened = probe(mutant, work, cases);
  assert.ok(weakened.every(row => row === 'compatible'), 'control must execute the weakened rule');
  return cases.filter(c => !c.compatible).length;
}

async function checkVM(cases) {
  let lines = [];
  const createVM = require(path.join(vmDir, 'eshkol-vm.js'));
  const vm = await createVM({print: t => lines.push(t), printErr: t => lines.push(t)});
  vm.ccall('repl_init', null, [], []);
  const evaluate = vm.cwrap('repl_eval', 'string', ['string']);
  function expect(source, expected) {
    lines = [];
    evaluate(source);
    assert.deepEqual(lines, expected, source);
  }
  expect('(define (contract a b) (tensor-matmul a b))', []);
  for (let m = 1; m <= 4; m++) {
    for (let n = 1; n <= 4; n++) {
      const values = Array(m * n).fill('1.0').join(' ');
      expect(`(define t-${m}-${n} (reshape (vector ${values}) ${m} ${n}))`, []);
      expect(`(tensor-shape t-${m}-${n})`, [`(${m} ${n})`]);
    }
  }
  for (const {dims: [m, k, j, n], compatible} of cases) {
    const call = `(contract t-${m}-${k} t-${j}-${n})`;
    expect(`(tensor-shape ${call})`, [compatible ? `(${m} ${n})` : guardError]);
  }
  // Same source site and bindings, different current right operand.
  expect('(define lhs t-2-3)', []);
  expect('(define rhs t-3-4)', []);
  expect('(define (repeat-site) (contract lhs rhs))', []);
  expect('(tensor-shape (repeat-site))', ['(2 4)']);
  expect('(set! rhs t-4-2)', []);
  expect('(repeat-site)', [guardError]);
  expect('(set! rhs t-3-4)', []);
  expect('(tensor-shape (repeat-site))', ['(2 4)']);
}

async function main() {
  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'verificatum-matmul-'));
  try {
    const {cases, sha256} = reference(work);
    const falseAcceptances = checkHelper(work, cases);
    await checkVM(cases);
    console.log(JSON.stringify({
      r1_a: 'pass', r1_b: 'open', corpus_sha256: sha256,
      cases: cases.length, compatible: cases.length - falseAcceptances,
      incompatible: falseAcceptances, weakened_helper_false_acceptances: falseAcceptances,
      unknown_dimension: 'proof required', runtime_rebinding: 'accepted, guarded, accepted',
      helper_sha256: digest(fs.readFileSync(helper)),
      vm_sha256: digest(fs.readFileSync(path.join(vmDir, 'eshkol-vm.wasm'))),
      model_sha256: digest(fs.readFileSync(path.join(kernel, 'Obligation.lean'))),
    }, null, 2));
  } finally {
    fs.rmSync(work, {recursive: true, force: true});
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
