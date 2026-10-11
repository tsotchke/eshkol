/* Independent wasm32 negative control for a real forward-jet tensor operand. */
#include <stdint.h>
typedef struct { uint8_t type, flags; uint16_t reserved; uint64_t data; } tagged;
extern void eshkol_ad_refuse_perturbed_operand(const char *operation);

__attribute__((export_name("check_perturbed_operand")))
void check_perturbed_operand(const tagged *operand) {
    if (operand->type == 6) /* ESHKOL_VALUE_DUAL_NUMBER */
        eshkol_ad_refuse_perturbed_operand("arithmetic");
}
