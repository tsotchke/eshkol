// Replay the same open terms through the repository's bundled WASM VM.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const createVM = require('../site/static/eshkol-vm.js');

async function main() {
  let output = [];
  const vm = await createVM({print: s => output.push(s), printErr: s => output.push(s)});
  vm.ccall('repl_init', null, [], []);
  const evaluate = vm.cwrap('repl_eval', 'string', ['string']);
  evaluate(fs.readFileSync(path.join(__dirname, '../lib/core/blc.esk'), 'utf8'));
  evaluate(`(define (reference t count)
    (let ((r (blc-step t)))
      (if (car r) (reference (cdr r) (+ count 1)) (cons count t))))`);
  const rows = fs.readFileSync(process.argv[2], 'utf8').trimEnd().split('\n');
  if (rows.length !== 1560) throw Error('incomplete binder corpus');
  for (const [i, row] of rows.entries()) {
    const fields = row.split('\t');
    if (fields.length !== 6 || fields.some(s => !/^[() a-z0-9]+$/.test(s))) throw Error('bad fixture');
    const [body, arg, contractum] = fields;
    output = [];
    evaluate(`(let* ((t (blc-app (blc-lam '${body}) '${arg})) (r (reference t 0)))
      (set! blc-max-steps (car r))
      (and (equal? (blc-step t) (cons #t '${contractum}))
           (equal? (blc-eval t) (cdr r))
           (begin (set! blc-max-steps (- (car r) 1))
                  (guard (e (#t (equal? (error-object-message e)
                                         "blc-eval: no normal form within step bound")))
                    (blc-eval t) #f))))`);
    if (output.join('').trim() !== '#t') throw Error(`bound mismatch at row ${i + 1}: ${output}`);
  }
  evaluate(`(define (shared-right n x)
    (if (= n 0) x (shared-right (- n 1) (blc-app (blc-lam x) (blc-lam x)))))`);
  for (const bound of [0, 1]) {
    output = [];
    evaluate(`(let* ((i '(lam (var 1))) (right (shared-right 40 '(var 1)))
                     (t (blc-app (blc-app i (blc-app i '(var 1))) right)))
                (set! blc-max-steps ${bound})
                (guard (e (#t (equal? (error-object-message e)
                                      "blc-eval: no normal form within step bound")))
                  (blc-eval t) #f))`);
    if (output.join('').trim() !== '#t') throw Error(`shared argument at bound ${bound}: ${output}`);
  }
  console.log('1560 VM boundary pairs and shared argument bounds passed');
}

main().catch(e => { console.error(e); process.exitCode = 1; });
