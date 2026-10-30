/* Copyright (c) 2026 Richard Hoekstra. SPDX-License-Identifier: MIT
 * Consumer of the actual constructed interpreter; no replacement VM or AD.
 * This translation-unit include follows the upstream standalone tool pattern.
 */
#define main eshkol_weight_tool_main
#include "weight_matrices.c"
#undef main
#include <time.h>
static int checks=0, runs=0, native_steps=0;
static long matrix_steps=0;
static void need(int condition, const char *message) {
    if (!condition) {fprintf(stderr,"SDNC FAIL: %s\n",message);exit(1);} checks++;
}
static float observed(const InterpreterWeights *weights, const SdncInstr *program, int count) {
    float reference[4]={0}, matrix[4]={0};
    int nr=run_reference(program,count,reference,4);
    FILE *trace=tmpfile(); need(trace!=NULL,"trace allocation");
    g_trace_tf_fp=trace;g_trace_program_name="frozen-hole-consumer";
    int nm=run_with_weights(weights,program,count,matrix,4);
    g_trace_tf_fp=NULL;g_trace_program_name=NULL;
    rewind(trace);char line[32768];int lines=0;
    while (fgets(line,sizeof(line),trace)) {
        if (strstr(line,"\"is_native\":true")) native_steps++;
        lines++;
    }
    fclose(trace);
    need(nr==1&&nm==1,"one complete output");
    need(g_last_mat_steps==g_last_ref_steps,"halt at reference boundary");
    need(lines==g_last_mat_steps+1,"complete matrix trace");
    need(isfinite(matrix[0])&&fabsf(matrix[0]-reference[0])<0.03f,"matrix/reference result");
    need(native_steps==0,"unexpected native delegation");
    runs++; matrix_steps+=g_last_mat_steps;
    return matrix[0];
}
/* The same numerator as 72*x1' for w=k/4 in the admitted Hawkes model:
 * k*(18*x1+9) + 24*x2 + 16 - 144*x1. Only operand k is a trainable hole.
 * A/B are context slots, not learned semantic weights or inferred code.
 */
static float predict(const InterpreterWeights *w,int k,int x1,int x2) {
    SdncInstr p[]={{OP_CONST,k},{OP_CONST,18*x1+9},{OP_MUL,0},
       {OP_CONST,24*x2+16-144*x1},{OP_ADD,0},{OP_PRINT,0},{OP_HALT,0}};
    return observed(w,p,sizeof(p)/sizeof(p[0]));
}
static float derivative(const InterpreterWeights *w,int k,int x1,int x2) {
    SdncInstr p[]={{OP_AD_VAR,k},{OP_AD_CONST,18*x1+9},{OP_AD_MUL,0},
       {OP_AD_CONST,24*x2+16-144*x1},{OP_AD_ADD,0},{OP_AD_BACKWARD,0},
       {OP_CONST,0},{OP_AD_GRAD,0},{OP_PRINT,0},{OP_HALT,0}};
    return observed(w,p,sizeof(p)/sizeof(p[0]));
}
static void retention(const InterpreterWeights *w,float out[4]) {
    SdncInstr a[]={{OP_CONST,3},{OP_CONST,5},{OP_ADD,0},{OP_PRINT,0},{OP_HALT,0}};
    SdncInstr b[]={{OP_CONST,41},{OP_SET_LOCAL,0},{OP_GET_LOCAL,0},{OP_CONST,1},
                   {OP_ADD,0},{OP_PRINT,0},{OP_HALT,0}};
    SdncInstr c[]={{OP_CONST,10},{OP_CONST,7},{OP_SWAP,0},{OP_SUB,0},{OP_PRINT,0},{OP_HALT,0}};
    SdncInstr d[]={{OP_AD_VAR,2},{OP_DUP,0},{OP_AD_MUL,0},{OP_AD_BACKWARD,0},
                   {OP_CONST,0},{OP_AD_GRAD,0},{OP_PRINT,0},{OP_HALT,0}};
    out[0]=observed(w,a,5);out[1]=observed(w,b,7);out[2]=observed(w,c,6);out[3]=observed(w,d,8);
    need(fabsf(out[0]-8)<0.03f&&fabsf(out[1]-42)<0.03f&&fabsf(out[2]+3)<0.03f&&fabsf(out[3]-4)<0.03f,
         "fixed retention expectations");
}
int main(void) {
    clock_t started=clock();
    InterpreterWeights *weights=calloc(1,sizeof(*weights)),*frozen=malloc(sizeof(*weights));
    need(weights&&frozen,"weight allocation");
    generate_weights(weights);memcpy(frozen,weights,sizeof(*weights));
    float old_behavior[4],new_behavior[4];retention(weights,old_behavior);
    /* Freeze training context and all held-out contexts before any update.
     * Integer hole k belongs to [0,4], target k=3. Three clipped gradient
     * steps use one training context (1,2); no held-out signal selects k.
     */
    const int heldout[3][2]={{0,3},{3,1},{4,2}};
    double before=0,after=0;
    for(int j=0;j<3;j++) {
        int x=heldout[j][0],y=heldout[j][1];
        float target=3*(18*x+9)+24*y+16-144*x;
        float error=predict(weights,0,x,y)-target;before+=(double)error*error/3;
    }
    int hole=0;
    for(int tick=0;tick<4;tick++) {
        float prediction=predict(weights,hole,1,2);
        float gradient=derivative(weights,hole,1,2);
        need(fabsf(gradient-27)<0.03f,"object AD against analytic derivative");
        float target=3*27+24*2+16-144;
        float g=(prediction-target)*gradient;
        /* Projected, clipped gradient descent on an integer sketch hole. */
        int update=fabsf(g)<0.01f?0:(g>0?1:-1),next=hole-update;
        if(next<0)next=0;if(next>4)next=4;
        printf("TRAIN %d %d %.9g %.9g %d\n",tick,hole,prediction,gradient,next);
        hole=next;
        need(memcmp(weights,frozen,sizeof(*weights))==0,"all semantic weight bytes frozen");
    }
    need(hole==3,"trained hole reaches frozen target");
    for(int j=0;j<3;j++) {
        int x=heldout[j][0],y=heldout[j][1];
        float target=3*(18*x+9)+24*y+16-144*x;
        float prediction=predict(weights,hole,x,y),error=prediction-target;
        after+=(double)error*error/3;
        printf("HELDOUT %d %d %.9g %.9g\n",x,y,prediction,target);
    }
    retention(weights,new_behavior);
    need(memcmp(old_behavior,new_behavior,sizeof(old_behavior))==0,"retention outputs byte identical");
    need(memcmp(weights,frozen,sizeof(*weights))==0,"final frozen semantic weights");
    need(before>1000&&after<0.001,"held-out loss improves at real model kernel");
    printf("SDNC {\"heldout_mse_before\":%.17g,\"heldout_mse_after\":%.17g,\"learned_k\":%d,"
       "\"physical_w\":0.75,\"weight_bytes\":%zu,\"weight_bytes_unchanged\":true,"
       "\"retention_programs\":4,\"native_delegated_steps\":%d,\"matrix_runs\":%d,"
       "\"matrix_steps\":%ld,\"cpu_seconds\":%.9g,\"checks\":%d}\n",
       before,after,hole,sizeof(*weights),native_steps,runs,matrix_steps,
       (double)(clock()-started)/CLOCKS_PER_SEC,checks);
    free(weights);free(frozen);return 0;
}
