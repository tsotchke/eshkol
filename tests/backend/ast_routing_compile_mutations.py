#!/usr/bin/env python3
"""Exercise real template instantiation and falsify the AST routing contract."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='c++')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    header = (root/'inc/eshkol/eshkol.h').read_text()
    enum = re.search(r'typedef\s+enum\s*\{([^{}]*)\}\s*eshkol_op_t\s*;', header, re.S)
    members = re.findall(r'\bESHKOL_\w+_OP\b', enum[1])
    # Two distinct routing actions; the runtime oracle is the enum's ordinal.
    source = '''#include <eshkol/core/ast_routing.h>
#include <cstdio>
enum class Route { Even, Odd };
Route classify(eshkol_op_t op) {
    return eshkol::routeAstOperation(op,
        eshkol::AstRouteGroup<Route::Even, EVEN>{},
        eshkol::AstRouteGroup<Route::Odd, ODD>{});
}
int main() {
    for (unsigned i = 0; i < COUNT; ++i) {
        if (classify(static_cast<eshkol_op_t>(i)) != (i % 2 ? Route::Odd : Route::Even)) return 1;
    }
    std::puts("PASS: every operation selected its declared policy");
}
'''.replace('EVEN', ', '.join(members[::2])).replace('ODD', ', '.join(members[1::2])).replace('COUNT', str(len(members)))
    with tempfile.TemporaryDirectory(prefix='ast-routing-mutations-') as directory:
        work = Path(directory)
        def compile_source(name, text, success, include=None):
            path = work/(name+'.cpp');path.write_text(text)
            msvc = Path(args.cxx).name.lower() in {'cl', 'cl.exe', 'clang-cl', 'clang-cl.exe'}
            executable = work/(name+'.exe' if msvc else name)
            if msvc:
                command = [args.cxx, '/nologo', '/std:c++17', '/W4', '/we4061', '/we4062']
                if include: command += ['/I'+str(include)]
                command += ['/I'+str(root/'inc'), str(path), '/Fe:'+str(executable), '/Fo:'+str(work/(name+'.obj'))]
            else:
                command = [args.cxx, '-std=c++17', '-Werror=switch', '-Werror=switch-enum']
                if include: command += ['-I', str(include)]
                command += ['-I', str(root/'inc'), str(path), '-o', str(executable)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            if (result.returncode == 0) != success:
                raise AssertionError(name+': unexpected compiler outcome\n'+result.stderr)
            diagnostics = result.stderr + result.stdout
            if not success and not any(token in diagnostics for token in ('AST operation', 'enumeration value', 'C4061', 'C4062')):
                raise AssertionError(name+': failed for an unrelated reason\n'+result.stderr)
            print('PASS:', name, 'accepted' if success else 'rejected', flush=True)
            return executable
        program = compile_source('complete-domain', source, True)
        subprocess.run([str(program)], check=True, timeout=10)
        compile_source('omitted-operation', source.replace(members[-1]+', ', '').replace(', '+members[-1], ''), False)
        compile_source('duplicate-operation', source.replace(members[-1], members[0]), False)
        # An enum extension must fail even if the central dispatch was not updated.
        mutant = work/'include/eshkol';mutant.mkdir(parents=True)
        (mutant/'eshkol.h').write_text(header.replace('} eshkol_op_t;', '    ESHKOL_MUTATION_OP,\n} eshkol_op_t;'))
        compile_source('new-enum-operation', source, False, mutant.parent)
        corrupted = source.replace('AstRouteGroup<Route::Even', 'AstRouteGroup<Route::Odd')
        wrong_program = compile_source('incorrect-route', corrupted, True)
        result = subprocess.run([str(wrong_program)], timeout=10)
        if result.returncode == 0: raise AssertionError('incorrect routing survived runtime oracle')
        print('PASS: incorrect-route rejected by runtime oracle')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
