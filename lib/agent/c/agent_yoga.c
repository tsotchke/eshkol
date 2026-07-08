/*******************************************************************************
 * Yoga Layout Engine Bindings for Eshkol Agent (B.21)
 *
 * Full flexbox layout via Facebook's Yoga (libyogacore).
 * Compile with -DHAS_YOGA and link against -lyogacore.
 *
 * Without HAS_YOGA, all functions return graceful errors (-1 / 0.0).
 *
 * Copyright (c) 2025 Eshkol Project — tsotchke
 ******************************************************************************/

#include <stdint.h>

#ifdef HAS_YOGA

#include <yoga/Yoga.h>

#define MAX_YOGA_NODES 512

static YGNodeRef g_nodes[MAX_YOGA_NODES] = {0};

/**
 * @brief Finds the first free slot in the @ref g_nodes handle table.
 *
 * Slot 0 is skipped (reserved so 0 is never a valid handle); the search
 * starts at index 1.
 *
 * @return The free slot index, or -1 if the table is full.
 */
static int alloc_node(void) {
    for (int i = 1; i < MAX_YOGA_NODES; i++) {
        if (!g_nodes[i]) return i;
    }
    return -1;
}

/**
 * @brief Allocates a new Yoga layout node and returns a handle for it.
 *
 * @return A handle (index into @ref g_nodes) for use with the other
 *         eshkol_yoga_node_* functions, or -1 if the node table is full or
 *         YGNodeNew() fails.
 */
int64_t eshkol_yoga_node_create(void) {
    int slot = alloc_node();
    if (slot < 0) return -1;
    g_nodes[slot] = YGNodeNew();
    return g_nodes[slot] ? (int64_t)slot : -1;
}

/**
 * @brief Frees the Yoga node for @p handle along with its entire child subtree.
 *
 * No-op if @p handle is out of range or already freed.
 */
void eshkol_yoga_node_free(int64_t handle) {
    if (handle < 1 || handle >= MAX_YOGA_NODES || !g_nodes[handle]) return;
    YGNodeFreeRecursive(g_nodes[handle]);
    g_nodes[handle] = NULL;
}

/**
 * @brief Sets a float-valued Yoga style property on the node for @p handle.
 *
 * @param prop Property selector: 0=width, 1=height, 2=min-width,
 *        3=min-height, 4=max-width, 5=max-height, 6=flex-grow,
 *        7=flex-shrink, 8=flex-basis, 9=gap, 10-13=padding
 *        (left/right/top/bottom), 14-17=margin (left/right/top/bottom),
 *        18-21=border (left/right/top/bottom). Unrecognized values are
 *        ignored.
 * @param value The property value, cast to float.
 *
 * No-op if @p handle is out of range or not currently allocated.
 */
/* Float properties: width, height, min/max, flex, padding, margin, border, gap */
void eshkol_yoga_node_set_float(int64_t handle, int32_t prop, double value) {
    if (handle < 1 || handle >= MAX_YOGA_NODES || !g_nodes[handle]) return;
    YGNodeRef node = g_nodes[handle];
    float v = (float)value;
    switch (prop) {
        case 0: YGNodeStyleSetWidth(node, v); break;
        case 1: YGNodeStyleSetHeight(node, v); break;
        case 2: YGNodeStyleSetMinWidth(node, v); break;
        case 3: YGNodeStyleSetMinHeight(node, v); break;
        case 4: YGNodeStyleSetMaxWidth(node, v); break;
        case 5: YGNodeStyleSetMaxHeight(node, v); break;
        case 6: YGNodeStyleSetFlexGrow(node, v); break;
        case 7: YGNodeStyleSetFlexShrink(node, v); break;
        case 8: YGNodeStyleSetFlexBasis(node, v); break;
        case 9: YGNodeStyleSetGap(node, YGGutterAll, v); break;
        case 10: YGNodeStyleSetPadding(node, YGEdgeLeft, v); break;
        case 11: YGNodeStyleSetPadding(node, YGEdgeRight, v); break;
        case 12: YGNodeStyleSetPadding(node, YGEdgeTop, v); break;
        case 13: YGNodeStyleSetPadding(node, YGEdgeBottom, v); break;
        case 14: YGNodeStyleSetMargin(node, YGEdgeLeft, v); break;
        case 15: YGNodeStyleSetMargin(node, YGEdgeRight, v); break;
        case 16: YGNodeStyleSetMargin(node, YGEdgeTop, v); break;
        case 17: YGNodeStyleSetMargin(node, YGEdgeBottom, v); break;
        case 18: YGNodeStyleSetBorder(node, YGEdgeLeft, v); break;
        case 19: YGNodeStyleSetBorder(node, YGEdgeRight, v); break;
        case 20: YGNodeStyleSetBorder(node, YGEdgeTop, v); break;
        case 21: YGNodeStyleSetBorder(node, YGEdgeBottom, v); break;
    }
}

/**
 * @brief Sets an enum-valued Yoga style property on the node for @p handle.
 *
 * @param prop Property selector: 0=flex-direction, 1=justify-content,
 *        2=align-items, 3=align-self, 4=align-content, 5=position-type,
 *        6=overflow, 7=display. Unrecognized values are ignored.
 * @param value The enum value, cast to the corresponding Yoga enum type.
 *
 * No-op if @p handle is out of range or not currently allocated.
 */
/* Integer properties: flex-direction, justify, align, position, overflow, display */
void eshkol_yoga_node_set_int(int64_t handle, int32_t prop, int32_t value) {
    if (handle < 1 || handle >= MAX_YOGA_NODES || !g_nodes[handle]) return;
    YGNodeRef node = g_nodes[handle];
    switch (prop) {
        case 0: YGNodeStyleSetFlexDirection(node, (YGFlexDirection)value); break;
        case 1: YGNodeStyleSetJustifyContent(node, (YGJustify)value); break;
        case 2: YGNodeStyleSetAlignItems(node, (YGAlign)value); break;
        case 3: YGNodeStyleSetAlignSelf(node, (YGAlign)value); break;
        case 4: YGNodeStyleSetAlignContent(node, (YGAlign)value); break;
        case 5: YGNodeStyleSetPositionType(node, (YGPositionType)value); break;
        case 6: YGNodeStyleSetOverflow(node, (YGOverflow)value); break;
        case 7: YGNodeStyleSetDisplay(node, (YGDisplay)value); break;
    }
}

/**
 * @brief Inserts the node for @p child into the node for @p parent's children at @p index.
 *
 * No-op if either @p parent or @p child is out of range or not currently
 * allocated.
 */
void eshkol_yoga_node_add_child(int64_t parent, int64_t child, int32_t index) {
    if (parent < 1 || parent >= MAX_YOGA_NODES || !g_nodes[parent]) return;
    if (child < 1 || child >= MAX_YOGA_NODES || !g_nodes[child]) return;
    YGNodeInsertChild(g_nodes[parent], g_nodes[child], (uint32_t)index);
}

/**
 * @brief Computes flexbox layout for the tree rooted at @p root, in left-to-right direction.
 *
 * After this call, computed values for @p root and its descendants can be
 * read via eshkol_yoga_node_get_computed().
 *
 * @param width Available width for the root node.
 * @param height Available height for the root node.
 *
 * No-op if @p root is out of range or not currently allocated.
 */
void eshkol_yoga_node_calculate(int64_t root, double width, double height) {
    if (root < 1 || root >= MAX_YOGA_NODES || !g_nodes[root]) return;
    YGNodeCalculateLayout(g_nodes[root], (float)width, (float)height, YGDirectionLTR);
}

/**
 * @brief Reads a computed layout value from the node for @p handle after eshkol_yoga_node_calculate() has run.
 *
 * @param prop Property selector: 0=left, 1=top, 2=width, 3=height,
 *        4-7=padding (left/top/right/bottom), 8-11=margin
 *        (left/top/right/bottom), 12-15=border (left/top/right/bottom).
 * @return The computed value, or 0.0 if @p handle is out of range, not
 *         currently allocated, or @p prop is unrecognized.
 */
double eshkol_yoga_node_get_computed(int64_t handle, int32_t prop) {
    if (handle < 1 || handle >= MAX_YOGA_NODES || !g_nodes[handle]) return 0.0;
    YGNodeRef node = g_nodes[handle];
    switch (prop) {
        case 0: return (double)YGNodeLayoutGetLeft(node);
        case 1: return (double)YGNodeLayoutGetTop(node);
        case 2: return (double)YGNodeLayoutGetWidth(node);
        case 3: return (double)YGNodeLayoutGetHeight(node);
        case 4: return (double)YGNodeLayoutGetPadding(node, YGEdgeLeft);
        case 5: return (double)YGNodeLayoutGetPadding(node, YGEdgeTop);
        case 6: return (double)YGNodeLayoutGetPadding(node, YGEdgeRight);
        case 7: return (double)YGNodeLayoutGetPadding(node, YGEdgeBottom);
        case 8: return (double)YGNodeLayoutGetMargin(node, YGEdgeLeft);
        case 9: return (double)YGNodeLayoutGetMargin(node, YGEdgeTop);
        case 10: return (double)YGNodeLayoutGetMargin(node, YGEdgeRight);
        case 11: return (double)YGNodeLayoutGetMargin(node, YGEdgeBottom);
        case 12: return (double)YGNodeLayoutGetBorder(node, YGEdgeLeft);
        case 13: return (double)YGNodeLayoutGetBorder(node, YGEdgeTop);
        case 14: return (double)YGNodeLayoutGetBorder(node, YGEdgeRight);
        case 15: return (double)YGNodeLayoutGetBorder(node, YGEdgeBottom);
        default: return 0.0;
    }
}

/** @brief Reports that Yoga layout support is compiled in. @return Always 1. */
int32_t eshkol_yoga_available(void) { return 1; }

#else /* !HAS_YOGA */

/** @brief Stub used when built without HAS_YOGA; no node is created. @return Always -1. */
int64_t eshkol_yoga_node_create(void) { return -1; }
/** @brief Stub used when built without HAS_YOGA; no-op. */
void    eshkol_yoga_node_free(int64_t h) { (void)h; }
/** @brief Stub used when built without HAS_YOGA; no-op. */
void    eshkol_yoga_node_set_float(int64_t h, int32_t p, double v) { (void)h;(void)p;(void)v; }
/** @brief Stub used when built without HAS_YOGA; no-op. */
void    eshkol_yoga_node_set_int(int64_t h, int32_t p, int32_t v) { (void)h;(void)p;(void)v; }
/** @brief Stub used when built without HAS_YOGA; no-op. */
void    eshkol_yoga_node_add_child(int64_t p, int64_t c, int32_t i) { (void)p;(void)c;(void)i; }
/** @brief Stub used when built without HAS_YOGA; no-op. */
void    eshkol_yoga_node_calculate(int64_t r, double w, double h) { (void)r;(void)w;(void)h; }
/** @brief Stub used when built without HAS_YOGA; layout is unavailable. @return Always 0.0. */
double  eshkol_yoga_node_get_computed(int64_t h, int32_t p) { (void)h;(void)p; return 0.0; }
/** @brief Reports that Yoga layout support is not compiled in. @return Always 0. */
int32_t eshkol_yoga_available(void) { return 0; }

#endif
