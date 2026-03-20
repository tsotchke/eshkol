#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

enum OpCode : uint8_t {
    OP_NOP=0, OP_CONST=1, OP_ADD=2, OP_SUB=3, OP_MUL=4, OP_DUP=5,
    OP_SWAP=6, OP_DROP=7, OP_LOAD=8, OP_STORE=9, OP_JUMP=10,
    OP_JUMP_IF=11, OP_OUTPUT=12, OP_HALT=13
};
struct I { OpCode op; int32_t operand; };

struct M { std::vector<int32_t> stk, mem, out; int pc; bool halt; };

void run(M& m, const I* p, int n, int mx=10000) {
    m.stk.clear(); m.mem.assign(256,0); m.out.clear(); m.pc=0; m.halt=false;
    for(int s=0;s<mx&&!m.halt&&m.pc>=0&&m.pc<n;s++){
        auto& i=p[m.pc]; int32_t a,b,addr;
        switch(i.op){
            case OP_NOP: m.pc++; break;
            case OP_CONST: m.stk.push_back(i.operand); m.pc++; break;
            case OP_ADD: b=m.stk.back();m.stk.pop_back();a=m.stk.back();m.stk.pop_back();m.stk.push_back(a+b);m.pc++; break;
            case OP_SUB: b=m.stk.back();m.stk.pop_back();a=m.stk.back();m.stk.pop_back();m.stk.push_back(a-b);m.pc++; break;
            case OP_MUL: b=m.stk.back();m.stk.pop_back();a=m.stk.back();m.stk.pop_back();m.stk.push_back(a*b);m.pc++; break;
            case OP_DUP: m.stk.push_back(m.stk.back()); m.pc++; break;
            case OP_SWAP: std::swap(m.stk[m.stk.size()-1],m.stk[m.stk.size()-2]); m.pc++; break;
            case OP_DROP: m.stk.pop_back(); m.pc++; break;
            case OP_LOAD: addr=m.stk.back();m.stk.pop_back();m.stk.push_back(m.mem[addr]);m.pc++; break;
            case OP_STORE: a=m.stk.back();m.stk.pop_back();addr=m.stk.back();m.stk.pop_back();m.mem[addr]=a;m.pc++; break;
            case OP_JUMP: m.pc=i.operand; break;
            case OP_JUMP_IF: a=m.stk.back();m.stk.pop_back();m.pc=(a!=0)?i.operand:m.pc+1; break;
            case OP_OUTPUT: a=m.stk.back();m.stk.pop_back();m.out.push_back(a);m.pc++; break;
            case OP_HALT: m.halt=true; break;
            default: m.pc++; break;
        }
    }
}

int main() {
    printf("=== Stack Machine Tests ===\n");
    M m;
    int pass=0, total=0;
    
    #define TEST(name, expected) { \
        run(m, prog, sizeof(prog)/sizeof(prog[0])); \
        bool ok = !m.out.empty() && m.out[0] == (expected); \
        printf("  %-20s = %4d (expected %4d) %s\n", name, m.out.empty()?-9999:m.out[0], expected, ok?"PASS":"FAIL"); \
        if(ok) pass++; total++; \
    }
    
    // 1. 3+5=8
    { I prog[]={{OP_CONST,3},{OP_CONST,5},{OP_ADD,0},{OP_OUTPUT,0},{OP_HALT,0}}; TEST("3+5", 8) }
    
    // 2. (3+5)*2=16
    { I prog[]={{OP_CONST,3},{OP_CONST,5},{OP_ADD,0},{OP_CONST,2},{OP_MUL,0},{OP_OUTPUT,0},{OP_HALT,0}}; TEST("(3+5)*2", 16) }
    
    // 3. 10-7=3
    { I prog[]={{OP_CONST,10},{OP_CONST,7},{OP_SUB,0},{OP_OUTPUT,0},{OP_HALT,0}}; TEST("10-7", 3) }
    
    // 4. mem store/load
    { I prog[]={{OP_CONST,0},{OP_CONST,42},{OP_STORE,0},{OP_CONST,0},{OP_LOAD,0},{OP_OUTPUT,0},{OP_HALT,0}}; TEST("mem[0]=42", 42) }
    
    // 5. sum(1..5)=15
    { I prog[]={
        {OP_CONST,0},{OP_CONST,0},{OP_STORE,0},     // 0-2: mem[0]=0
        {OP_CONST,1},{OP_CONST,5},{OP_STORE,0},     // 3-5: mem[1]=5
        {OP_CONST,1},{OP_LOAD,0},                    // 6-7: push i
        {OP_DUP,0},                                   // 8
        {OP_JUMP_IF,12},                              // 9
        {OP_DROP,0},                                   // 10
        {OP_JUMP,26},                                  // 11
        {OP_CONST,0},{OP_LOAD,0},                    // 12-13
        {OP_ADD,0},                                    // 14
        {OP_CONST,0},{OP_SWAP,0},{OP_STORE,0},      // 15-17
        {OP_CONST,1},{OP_LOAD,0},                    // 18-19
        {OP_CONST,1},{OP_SUB,0},                     // 20-21
        {OP_CONST,1},{OP_SWAP,0},{OP_STORE,0},      // 22-24
        {OP_JUMP,6},                                   // 25
        {OP_CONST,0},{OP_LOAD,0},                    // 26-27
        {OP_OUTPUT,0},{OP_HALT,0},                    // 28-29
    }; TEST("sum(1..5)", 15) }
    
    // 6. fib(7)=13
    { I prog[]={
        {OP_CONST,0},{OP_CONST,0},{OP_STORE,0},     // 0-2: mem[0]=0 (a)
        {OP_CONST,1},{OP_CONST,1},{OP_STORE,0},     // 3-5: mem[1]=1 (b)
        {OP_CONST,2},{OP_CONST,7},{OP_STORE,0},     // 6-8: mem[2]=7 (n)
        {OP_CONST,2},{OP_LOAD,0},                    // 9-10: push n
        {OP_DUP,0},                                   // 11
        {OP_JUMP_IF,15},                              // 12: if n!=0 goto body
        {OP_DROP,0},                                   // 13
        {OP_JUMP,33},                                  // 14: goto output
        // body: temp=a+b, a=b, b=temp, n--
        {OP_CONST,0},{OP_LOAD,0},                    // 15-16: push a
        {OP_CONST,1},{OP_LOAD,0},                    // 17-18: push b
        {OP_ADD,0},                                    // 19: a+b=temp
        // a=b: mem[0]=mem[1]
        {OP_CONST,0},{OP_CONST,1},{OP_LOAD,0},{OP_STORE,0}, // 20-23
        // b=temp: mem[1]=temp (temp is on stack)
        {OP_CONST,1},{OP_SWAP,0},{OP_STORE,0},      // 24-26
        // n--
        {OP_CONST,2},{OP_LOAD,0},                    // 27-28
        {OP_CONST,1},{OP_SUB,0},                     // 29-30
        {OP_CONST,2},{OP_SWAP,0},{OP_STORE,0},      // 31-33... wait that's 34
        // Recount: 31=CONST 2, 32=SWAP, 33=STORE, 34=JUMP 9
        {OP_JUMP,9},                                   // 34
        // output: but I said JUMP 33 above... need to fix
    }; 
    // Actually the JUMP target at PC 14 should go to the output section
    // which starts AFTER the JUMP at the end of the loop body
    // Let me count: body ends at PC 34 (JUMP 9)
    // So output starts at PC 35
    // Fix: change PC 14 from JUMP 33 to JUMP 35
    // But I can't easily modify the array in place...
    // Let me just reconstruct with correct indices
    }
    
    // 6. fib(7)=13 — corrected
    { I prog[]={
        {OP_CONST,0},{OP_CONST,0},{OP_STORE,0},     // 0-2
        {OP_CONST,1},{OP_CONST,1},{OP_STORE,0},     // 3-5
        {OP_CONST,2},{OP_CONST,7},{OP_STORE,0},     // 6-8
        {OP_CONST,2},{OP_LOAD,0},                    // 9-10
        {OP_DUP,0},                                   // 11
        {OP_JUMP_IF,15},                              // 12
        {OP_DROP,0},                                   // 13
        {OP_JUMP,35},                                  // 14: goto output at 35
        {OP_CONST,0},{OP_LOAD,0},                    // 15-16
        {OP_CONST,1},{OP_LOAD,0},                    // 17-18
        {OP_ADD,0},                                    // 19
        {OP_CONST,0},{OP_CONST,1},{OP_LOAD,0},{OP_STORE,0}, // 20-23: mem[0]=b
        {OP_CONST,1},{OP_SWAP,0},{OP_STORE,0},      // 24-26: mem[1]=temp
        {OP_CONST,2},{OP_LOAD,0},                    // 27-28
        {OP_CONST,1},{OP_SUB,0},                     // 29-30
        {OP_CONST,2},{OP_SWAP,0},{OP_STORE,0},      // 31-33
        {OP_JUMP,9},                                   // 34
        {OP_CONST,0},{OP_LOAD,0},                    // 35-36
        {OP_OUTPUT,0},{OP_HALT,0},                    // 37-38
    }; TEST("fib(7)", 13) }
    
    // 7. factorial(5) = 120
    { I prog[]={
        {OP_CONST,0},{OP_CONST,1},{OP_STORE,0},     // 0-2: mem[0]=1 (result)
        {OP_CONST,1},{OP_CONST,5},{OP_STORE,0},     // 3-5: mem[1]=5 (n)
        {OP_CONST,1},{OP_LOAD,0},                    // 6-7: push n
        {OP_DUP,0},                                   // 8
        {OP_JUMP_IF,12},                              // 9
        {OP_DROP,0},                                   // 10
        {OP_JUMP,26},                                  // 11: goto output
        // body: result *= n, n--
        {OP_CONST,0},{OP_LOAD,0},                    // 12-13: push result
        {OP_MUL,0},                                    // 14: n * result
        {OP_CONST,0},{OP_SWAP,0},{OP_STORE,0},      // 15-17: mem[0] = n*result
        {OP_CONST,1},{OP_LOAD,0},                    // 18-19: push n
        {OP_CONST,1},{OP_SUB,0},                     // 20-21: n-1
        {OP_CONST,1},{OP_SWAP,0},{OP_STORE,0},      // 22-24: mem[1] = n-1
        {OP_JUMP,6},                                   // 25
        {OP_CONST,0},{OP_LOAD,0},                    // 26-27
        {OP_OUTPUT,0},{OP_HALT,0},                    // 28-29
    }; TEST("5!", 120) }
    
    // 8. Multiple outputs: 1, 2, 3
    { I prog[]={
        {OP_CONST,1},{OP_OUTPUT,0},
        {OP_CONST,2},{OP_OUTPUT,0},
        {OP_CONST,3},{OP_OUTPUT,0},
        {OP_HALT,0},
    };
    run(m, prog, 7);
    bool ok = m.out.size()==3 && m.out[0]==1 && m.out[1]==2 && m.out[2]==3;
    printf("  %-20s = [%d,%d,%d] %s\n", "output 1,2,3",
           m.out.size()>0?m.out[0]:-1, m.out.size()>1?m.out[1]:-1, m.out.size()>2?m.out[2]:-1,
           ok?"PASS":"FAIL");
    if(ok) pass++; total++;
    }
    
    printf("\n%d/%d passed\n", pass, total);
    return pass == total ? 0 : 1;
}

extern "C" {
int eshkol_weight_compiler_test(void) { return main(); }
}
