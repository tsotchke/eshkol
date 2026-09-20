// Execute the committed WASM VM with the exact output callbacks used by the page.
// No browser/network/dependency installation is needed for this deployment gate.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
const factory = require(path.join(root, 'site/static/eshkol-vm.js'));
const html = fs.readFileSync(path.join(root, 'site/static/index.html'), 'utf8');
const begin = html.indexOf('// ─── Bytecode VM for browser REPL');
const end = html.indexOf('// ─── REPL keyboard handler', begin);
assert(begin >= 0 && end > begin, 'page VM bootstrap must be present');
let loading;
const context = {window: {}, runtime: {}, TextDecoder, Uint8Array,
  console: {log() {}, warn() {}},
  EshkolVM(options) { loading = factory(options); return loading; }};
vm.createContext(context);
vm.runInContext(html.slice(begin, end), context);

function numeric(expected, tolerance = 1e-12) {
  return actual => assert(Math.abs(Number(actual) - expected) <= tolerance,
    `expected ${expected} ± ${tolerance}, got ${actual}`);
}
// Independent arithmetic/analytic oracles, not results blessed from the VM.
let euler = 1, time = 0;
while (time < 2) { euler += 0.001 * -euler; time += 0.001; }
const expected = [
  ['Hello, world!', ''], ['45.14', '14'], ['5', '3'], ['5'], ['15', '25'], ['8'],
  ['(1 4 9 16 25)', '(2 4 6)', '15'], ['(0 1 1 2 3 5 8 13 21 34 55)'],
  ['(1 1 2 3 3 4 5 5 5 6 9)'], ['18446744073709551616', '1/2', '1'],
  ['3+4i', '5', '4+3i'], ['ff', '11111111', '255', numeric(1 / 3)],
  ['12', '7'], [numeric(2 * Math.cos(1))], ['-4'], ['15', '3'],
  ['32', '(1 4 9 16)'], ['(1 2 3 4 5)', '(6 8 10)'], [numeric(euler)],
  ['3628800', '2432902008176640000'], [numeric(Math.sqrt(2), 0.00001), numeric(3, 0.00001)],
  ['({?food -> pizza} {?food -> sushi})'], ['({?x -> math} {?x -> physics})'],
  actual => {
    const m = actual.match(/^surprise with no evidence: (\S+)\nsurprise after symptom:    (\S+)$/);
    assert(m && Number.isFinite(+m[1]) && Number.isFinite(+m[2]) && +m[2] > +m[1],
      `the documented evidence must increase surprise: ${actual}`);
  }
];
const homeExpected = [
  ['12'], [numeric(2, 1e-8)], [numeric(Math.exp(-4), 1e-8)],
  ['({?child -> bob})'], ['-4'], ['#(17 39)'], ['(4 6 8)'], ['enter exit42', '']
];
let decay = 1, decayTime = 0;
while (decayTime < 1) { decay += 0.001 * (-2 * decay); decayTime += 0.001; }
const examplesExpected = [
  [numeric(Math.sqrt(2)), ''], ['832040'], [numeric(3, 1e-8)],
  ['18446744073709551616', '3/7', '1'], ['(1 4 9 16 25)', '(2 4 6)', '15'], ['1', '2', '3'],
  [line => { assert(line.startsWith('f(x): ')); numeric(Math.sin(2.25) + 4.5)(line.slice(6)); },
   line => { assert(line.startsWith("f'(x): ")); numeric(3 * Math.cos(2.25) + 3)(line.slice(7)); }],
  [numeric(decay)],
  ['({?d -> flu} {?d -> covid})', line => {
    assert(/^surprise after fever: -?\d+(\.\d+)?$/.test(line));
    assert(Number.isFinite(Number(line.slice('surprise after fever: '.length))));
  }]
];
async function main() {
  const module = await loading;
  function evaluate(source) {
    module.ccall('repl_reset', null, [], []);
    context.window._vmOutput = '';
    context.window._vmStderr = '';
    const result = module.cwrap('repl_eval', 'string', ['string'])(source);
    assert.equal(result, '', 'REPL initialization must succeed');
    return {out: context.window._vmOutput, err: context.window._vmStderr};
  }
  for (const text of ['tail without newline', 'λ café 日本語 🚀', 'next independent evaluation']) {
    const result = evaluate(`(display ${JSON.stringify(text)})`);
    assert.deepEqual(result, {out: text, err: ''});
  }
  const source = fs.readFileSync(path.join(root, 'site/src/main.esk'), 'utf8');
  const learn = source.slice(source.indexOf('(define (render-learn-page '), source.indexOf('(define (render-example-card '));
  const directPattern = /\(create-runnable-code-block\s+\w+\s+("(?:\\.|[^"\\])*")/g;
  const examples = [...learn.matchAll(directPattern)]
    .map(m => JSON.parse(m[1]));
  assert.equal(examples.length, expected.length, 'every Learn example needs a reviewed oracle');
  const home = [...source.slice(0, source.indexOf('(define (render-learn-page ')).matchAll(directPattern)]
    .map(m => JSON.parse(m[1]));
  assert.equal(home.length, homeExpected.length, 'every homepage example needs a reviewed oracle');
  const cards = [...source.matchAll(/\(render-example-card\s+\w+\s+"(?:\\.|[^"\\])*"\s+"(?:\\.|[^"\\])*"\s+("(?:\\.|[^"\\])*")/g)]
    .map(m => JSON.parse(m[1])).filter(Boolean);
  assert.equal(cards.length, examplesExpected.length, 'every Examples card needs a reviewed oracle');
  assert.equal([...source.matchAll(directPattern)].length, home.length + examples.length,
    'new runnable route needs test coverage');
  const cases = [...examples, ...home, ...cards];
  const oracles = [...expected, ...homeExpected, ...examplesExpected];
  for (let i = 0; i < cases.length; i++) {
    const {out, err} = evaluate(cases[i]);
    assert.equal(err, '', `Site example ${i + 1} stderr`);
    const oracle = oracles[i];
    if (typeof oracle === 'function') oracle(out);
    else {
      const lines = out.split('\n');
      assert.equal(lines.length, oracle.length, `Learn example ${i + 1}: missing/extra output: ${out}`);
      oracle.forEach((value, j) => typeof value === 'function' ? value(lines[j]) :
        assert.equal(lines[j], value, `Learn example ${i + 1}, line ${j + 1}`));
    }
  }
  console.log(`PASS: ${cases.length} runnable examples (24 Learn, 8 home, 9 Examples), with output/property oracles; 3 UTF-8/no-newline/isolation regressions`);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
