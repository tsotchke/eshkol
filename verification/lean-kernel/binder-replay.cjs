// The input is the finite corpus printed by the reviewed Lean specification.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const createVM = require('../../site/static/eshkol-vm.js');
async function replay() {
  let lines = [];
  const vm = await createVM({print:t => lines.push(t), printErr:t => lines.push(t)});
  vm.ccall('repl_init', null, [], []);
  const evaluate = vm.cwrap('repl_eval', 'string', ['string']);
  for (const file of ['../../lib/core/blc.esk', 'import.esk', 'kernel.esk']) {
    evaluate(fs.readFileSync(path.join(__dirname, file), 'utf8'));
  }
  const cases = fs.readFileSync(0, 'utf8').trimEnd().split('\n');
  if (!cases.length) throw Error('Empty binder corpus');
  for (const row of cases) {
    const fields = row.split('\t');
    if (fields.length !== 6 || fields.some(s => !/^[() a-z0-9]+$/.test(s))) throw Error('Bad corpus row');
    const [body,arg,expected,kbody,karg,kexpected] = fields;
    lines = [];
    evaluate("(begin (set! k-fuel 12000) (and " +
      "(equal? (blc-step (blc-app (blc-lam '" + body + ") '" + arg + ")) (cons #t '" + expected + ")) " +
      "(equal? (k-inst '" + kbody + " '" + karg + " 0) '" + kexpected + ")))");
    if (lines.join('').trim() !== '#t') throw Error('Binder disagreement: ' + row);
  }
  console.log('Lean erasure proof checked; both Eshkol reductions agreed on ' + cases.length + ' open redexes.');
}
replay().catch(e => { console.error(e.message); process.exitCode = 1; });
