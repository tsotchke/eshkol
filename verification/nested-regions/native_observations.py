"""Check the sequence, region bindings and pair extents of each observed workload."""

from dataclasses import dataclass, replace


def require(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class Row:
    event: int
    fallback: int
    domain: int
    route: int
    depth: int
    obj: int
    base: int
    used: int
    capacity: int
    slots: tuple

    def binding(self):
        return (self.fallback, self.domain, self.route, self.depth, self.slots)


def check(rows, cases):
    require([r.event for r in rows] == list(range(10)), "incomplete region events")
    opening, parent, child, entry, returned, prior, fresh, closing, closed, final = rows
    arenas = (opening.fallback, parent.domain, child.domain)
    require(all(arenas) and len(set(arenas)) == 3, "three distinct arena identities")
    require(
        opening.binding() == (arenas[0], 0, arenas[0], 0, (0,) * 64), "opening state"
    )
    parent_id, child_id = parent.slots[0], child.slots[1]
    require(parent_id and child_id and parent_id != child_id, "distinct regions")
    require(
        parent.binding()
        == (arenas[0], arenas[1], arenas[1], 1, (parent_id,) + (0,) * 63),
        "parent entry",
    )
    require(
        child.binding()
        == (arenas[0], arenas[2], arenas[2], 2, (parent_id, child_id) + (0,) * 62),
        "child entry",
    )
    require(
        entry.obj == child.domain and closing.obj == parent.domain,
        "destructors bound to actual arenas",
    )
    require(
        all(r.binding() == parent.binding() for r in (entry, returned, prior, fresh)),
        "parent restoration",
    )
    require(
        all(r.binding() == opening.binding() for r in (closing, closed, final)),
        "outer restoration",
    )
    require(
        parent.base
        and entry.base
        and parent.base == prior.base == fresh.base == closing.base
        and parent.capacity == prior.capacity == fresh.capacity == closing.capacity,
        "parent backing",
    )
    require(
        0 < entry.used <= entry.capacity
        and 0
        < parent.used
        == prior.used
        < fresh.used
        <= closing.used
        <= closing.capacity,
        "live cursor order",
    )
    require(
        parent.base + parent.capacity <= entry.base
        or entry.base + entry.capacity <= parent.base,
        "shared backing",
    )
    require(
        parent.obj == prior.obj
        and parent.base <= parent.obj - 8
        and parent.obj + 32 <= parent.base + parent.used,
        "prior pair extent",
    )
    require(
        prior.base + prior.used <= fresh.obj - 8
        and fresh.obj + 32 <= fresh.base + fresh.used,
        "fresh pair extent",
    )
    for sample in (
        entry,
        replace(entry, domain=entry.fallback, route=entry.fallback),
        replace(entry, slots=(0,) + entry.slots[1:]),
        replace(entry, slots=entry.slots[:-1] + (child_id,)),
    ):
        cases.append(
            [
                opening.fallback,
                parent_id,
                parent.domain,
                child.fallback,
                child.domain,
                child.depth,
                sample.fallback,
                sample.domain,
                sample.depth,
                *child.slots,
                *sample.slots,
            ]
        )
    return dict(
        parent_cursor=[parent.used, prior.used, fresh.used],
        child_used=entry.used,
        survivor_offset=prior.obj - prior.base,
        fresh_offset=fresh.obj - fresh.base,
    )
