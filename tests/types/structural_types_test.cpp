#include "eshkol/types/structural_types.h"
#include <cstdio>

// Solver contract of the structural type layer (ADR-0046): rows, protocols,
// duality, levels, generalisation, usage restriction, predicativity.
using namespace eshkol::hott::st;
static int fails = 0;
#define EXPECT(c, msg) do { if (!(c)) { std::printf("FAIL %s\n", msg); ++fails; } else std::printf("ok   %s\n", msg); } while (0)
int main() {
  TypeStore s; Solver v(s);
  // row: {x: a | r} ~ {y: string, x: number}
  TyRef a = s.meta(Kind::Type, 1); TyRef r = s.meta(Kind::Row, 1); r->lacks.insert("x");
  TyRef r1 = s.record(s.rowExtend("x", a, r));
  TyRef r2 = s.record(s.rowExtend("y", s.string(), s.rowExtend("x", s.number(), s.rowEmpty())));
  EXPECT(v.unify(r1, r2), "row rewrite unify");
  std::printf("     %s\n", s.show(s.zonk(r1)).c_str());
  // duplicate label
  TyRef r3 = s.meta(Kind::Row, 1); r3->lacks.insert("x");
  v.clearError();
  EXPECT(!v.unify(s.record(r3), s.record(s.rowExtend("x", s.number(), s.rowEmpty()))), "lacks rejects dup");
  std::printf("     %s\n", v.error().c_str());
  // missing label
  v.clearError();
  EXPECT(!v.unify(s.record(s.rowExtend("y", s.number(), s.meta(Kind::Row,1))), s.record(s.rowExtend("x", s.number(), s.rowEmpty()))), "missing label");
  std::printf("     %s\n", v.error().c_str());
  // sessions: rec x. !int.x  ~  !int. rec y. !int. y
  int id1 = s.freshId(), id2 = s.freshId();
  TyRef m1 = s.mu(id1, "x", s.send(s.number(), s.recVar(id1, "x")));
  TyRef m2 = s.send(s.number(), s.mu(id2, "y", s.send(s.number(), s.recVar(id2, "y"))));
  v.clearError();
  EXPECT(v.unify(m1, m2), "equi-recursive protocols");
  // duality: dual(meta) solved later
  TyRef p = s.meta(Kind::Session, 1);
  TyRef dp = s.dual(p);
  v.clearError();
  EXPECT(v.unify(dp, s.recv(s.number(), s.end())), "dual of meta");
  std::printf("     p = %s\n", s.show(s.zonk(p)).c_str());
  // dual of recursive offer
  int id3 = s.freshId();
  TyRef off = s.mu(id3, "l", s.offer(s.rowExtend("more", s.recv(s.number(), s.recVar(id3, "l")), s.rowExtend("done", s.send(s.number(), s.end()), s.rowEmpty(Kind::Session)))));
  std::printf("     dual = %s\n", s.show(s.dual(off)).c_str());
  v.clearError();
  EXPECT(v.unify(s.dual(s.dual(off)), off), "dual involution");
  // mismatch
  v.clearError();
  EXPECT(!v.unify(s.send(s.number(), s.end()), s.recv(s.number(), s.end())), "send vs recv");
  std::printf("     %s\n", v.error().c_str());
  // skolem escape
  TyRef outer = s.meta(Kind::Type, 1); TyRef sk = s.skolem("s", Kind::Type, 2);
  v.clearError();
  EXPECT(!v.unify(outer, s.con("st-ref", {sk, s.number()})), "skolem escape");
  std::printf("     %s\n", v.error().c_str());
  // generalize
  TyRef g1 = s.meta(Kind::Type, 1); TyRef g2 = s.meta(Kind::Row, 1); g2->lacks.insert("x");
  TyRef fn = s.arrow({s.record(s.rowExtend("x", g1, g2))}, g1);
  std::printf("     gen = %s\n", s.show(s.generalize(fn, 0)).c_str());
  // instantiate + unrestricted
  TyRef sc = s.generalize(fn, 0);
  TyRef inst = s.instantiate(sc, 1);
  v.clearError();
  EXPECT(!v.unify(inst, s.arrow({s.record(s.rowExtend("x", s.chan(s.end()), s.rowEmpty()))}, s.chan(s.end()))), "unrestricted var rejects linear");
  std::printf("     %s\n", v.error().c_str());
  // predicativity
  TyRef pm = s.meta(Kind::Type, 1);
  Binder b; b.id = s.freshId(); b.name = "a"; b.kind = Kind::Type;
  TyRef poly = s.forall({b}, s.arrow({s.bound(b)}, s.bound(b)));
  v.clearError();
  EXPECT(!v.unify(pm, poly), "predicative");
  // rowInclusion
  TyRef sub = s.rowExtend("IO", s.null(), s.rowEmpty());
  TyRef sup = s.rowExtend("State", s.number(), s.rowExtend("IO", s.null(), s.rowEmpty()));
  v.clearError();
  EXPECT(v.rowInclusion(sub, sup), "row inclusion");
  std::printf("%d failures\n", fails);
  if (fails == 0) std::printf("PASS: structural type solver\n");
  return fails != 0;
}
