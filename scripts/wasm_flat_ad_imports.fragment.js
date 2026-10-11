// Browser WASM has no Taylor tower lane. Keep the base lane's established
// flat behavior: extraction declines the tower and enter/leave do nothing.
eshkol_ad_tower_carry_result: () => 0,
eshkol_ad_jet_extract_tower: () => 0,
// Captured nested differentiation is explicitly unsupported in this lane.
// Throwing is required so unsupported semantics cannot silently look valid.
eshkol_ad_nested_capture_unsupported: () => {
    throw new Error('Nested autodiff through captured values is unsupported in the browser WASM runtime');
},
// A nested reverse pass cannot consume an enclosing forward jet. Match the
// native fatal path, including its operation name, instead of dropping the
// outer tangent while converting the operand to an AD constant.
eshkol_ad_refuse_perturbed_operand: (operationPtr) => {
    const operation = this.readString(operationPtr) || 'arithmetic';
    throw new Error(`${operation}: a nested reverse-mode derivative cannot consume an outer reverse-mode value; differentiate the outer level with forward mode or restructure the computation`);
},
eshkol_ad_tower_enter: () => {},
eshkol_ad_tower_leave: () => {},
// A jet pass's extraction guard (ADR-0027). The lite lane has no Taylor
// carrier, so no carrier can reach it.
eshkol_ad_jet_result_check: () => {},
