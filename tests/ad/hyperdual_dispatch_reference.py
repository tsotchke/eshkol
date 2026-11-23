from pathlib import Path
import re
import sys


backend = Path(sys.argv[1])
reference = (backend / "vm_hyperdual.c").read_text()
dispatch = (backend / "vm_native.c").read_text().split("Hyper-Dual Numbers (", 1)[1]
dispatch = dispatch.split("AD Operations (", 1)[0]
cases = {int(n) for n in re.findall(r"case (\d+):", dispatch)}
low, high = map(int, re.search(r"Native call IDs: (\d+)-(\d+)", reference).groups())
assert cases == set(range(low, high + 1))
declarations = re.findall(
    r"@brief Native call (\d+):.*?\*/\s*VmHyperDual\* (vm_hd_\w+)\(", reference, re.S
)
for number, function in declarations:
    bodies = re.split(r"case (\d+):", dispatch)
    assert any(case == number and function + "(" in body
               for case, body in zip(bodies[1::2], bodies[2::2])), (number, function)
