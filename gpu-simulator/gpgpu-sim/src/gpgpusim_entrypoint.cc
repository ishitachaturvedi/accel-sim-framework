// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ivan Sham,
// Andrew Turner, Ali Bakhoda, The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "gpgpusim_entrypoint.h"
#include <stdio.h>
#include <iostream>

#include "../libcuda/gpgpu_context.h"
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "cuda-sim/ptx_parser.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "option_parser.h"
#include "stream_manager.h"
#include "gpgpu-sim/fast.h"

using namespace std;

#include <vector>
#include <fstream>
using std::vector;

long long cycle_num;
long total_times_in_cycle = 0;
vector<vector<vector<int>>>stallData;
vector<vector<int>>act_warp;
vector<vector<int>>issued_warp;
vector<vector<vector<int>>> str_status; 
vector<int>warp_issue;
vector<int>icnt_pressure;
vector<int>warps_cannot_be_issued;
vector<unsigned>warp_inst_num;
vector<int> indep_pc_num_push_all_stalling_inst;

vector<int> stall_consolidated;

long long inst_counter = 0;
long long max_active;
long long actw;
long long max_warps_act;
long long cycles_passed = 0;
long long max_sid;
long long num_of_schedulers;
long long numstall = 12;
long long print_on = 0;
long long going_from_shader_to_mem = 0;
long long present_ongoing_cycle = 0;
long long stall_cycles = 0;
long long tot_icnt_buffer = 0;
long long tot_inst_exec = 0;
long long tot_cycles_exec_all_SM = 0;
long long tot_inst_ret = 0;
extern int total_warps = 0;

int tot_mem_dep_checks = 0;
int tot_mem_dep_true = 0;
int tot_mem_dep_false = 0;
int time_bw_inst = 0; 
int reached_barrier = 0;

int tot_issues_ILP = 0;
int tot_issues_OOO = 0;

// Stats collection
long long mem_data_stall = 0;
long long comp_data_stall = 0;
long long ibuffer_stall = 0;
long long comp_str_stall = 0;
long long mem_str_stall = 0;
long long mem_data_stall_kernel = 0;
long long tot_inst_OOO_because_dep = 0;
long long ibuffer_stall_kernel = 0;
long long comp_str_stall_kernel = 0;
long long mem_str_stall_kernel = 0;
long long waiting_warp = 0;
long long SM_num = 0;
long long idle = 0;
long long other_stall1_kernel = 0;
long long other_stall2_kernel= 0;
long long other_stall3_kernel = 0;
long long ooo_opp_kernel = 0;
long long mem_data_stall_issue_irr = 0;
long long comp_data_stall_issue_irr = 0;
long long ibuffer_stall_issue_irr = 0;
long long comp_str_stall_issue_irr = 0;
long long mem_str_stall_issue_irr = 0;
long long other_stall_issue_irr1 = 0;
long long other_stall_issue_irr2 = 0;
long long other_stall_issue_irr3 = 0;
long WAR_and_WAW_stalls = 0;
bool WAR_or_WAW_found = 0;
long long ICNT_TO_MEM_cycles = 0;
long long ICNT_TO_MEM_cycles_kernel = 0;
long long ICNT_TO_SHADER_count = 0;
long long ICNT_TO_SHADER_cycles = 0;
long long ICNT_TO_SHADER_count_kernel = 0;
long long ICNT_TO_SHADER_cycles_kernel = 0;
long long ROP_DELAY_count = 0;
long long ROP_DELAY_cycle = 0;
long long ROP_DELAY_count_kernel = 0;
long long ROP_DELAY_cycle_kernel = 0;
long long ICNT_TO_L2_QUEUE_count = 0;
long long ICNT_TO_L2_QUEUE_count_kernel = 0;
long long ICNT_TO_L2_QUEUE_cycles = 0;
long long ICNT_TO_L2_QUEUE_cycles_kernel = 0;
long long L2_TO_DRAM_QUEUE_count = 0;
long long L2_TO_DRAM_QUEUE_cycle = 0;
long long DRAM_LATENCY_QUEUE_count = 0;
long long DRAM_LATENCY_QUEUE_cycle = 0;
long long DRAM_LATENCY_QUEUE_count_kernel = 0;
long long DRAM_LATENCY_QUEUE_cycle_kernel = 0;
long long DRAM_TO_L2_QUEUE_count = 0;
long long DRAM_TO_L2_QUEUE_cycle = 0;
long long DRAM_TO_L2_QUEUE_count_kernel = 0;
long long DRAM_TO_L2_QUEUE_cycle_kernel = 0;
long long DRAM_L2_FILL_QUEUE_count = 0;
long long DRAM_L2_FILL_QUEUE_cycle = 0;
long long DRAM_L2_FILL_QUEUE_count_kernel = 0;
long long DRAM_L2_FILL_QUEUE_cycle_kernel = 0;
long long L2_TO_ICNT_count = 0;
long long L2_TO_ICNT_cycle = 0;
long long L2_TO_ICNT_count_kernel = 0;
long long L2_TO_ICNT_cycle_kernel = 0;
long long CLUSTER_TO_SHADER_QUEUE_count = 0;
long long CLUSTER_TO_SHADER_QUEUE_cycle = 0;
long long CLUSTER_TO_SHADER_QUEUE_count_kernel = 0;
long long CLUSTER_TO_SHADER_QUEUE_cycle_kernel                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 = 0;
long long CLUSTER_TO_SHADER_QUEUE_1_count = 0;
long long CLUSTER_TO_SHADER_QUEUE_1_cycle = 0;
long long mem_issues = 0;
long long mem_cycle_counter = 0;
long long l2_cache_bank_access = 0;
long long l2_cache_bank_miss = 0;
long long l2_cache_access = 0;
long long l2_cache_miss = 0;
long long l2_pending = 0;
long long l2_res_fail = 0;
long long c_mem_resource_stall = 0;
long long s_mem_bk_conf = 0;
long long gl_mem_resource_stall = 0;
long long gl_mem_coal_stall = 0;
long long gl_mem_data_port_stall = 0;
long long icnt_creat_inj = 0;
long long icnt_creat_arrival = 0;
long long icnt_inj_arrival = 0;
long long icnt_creat_inj_READ_REQUEST = 0;
long long icnt_creat_arrival_READ_REQUEST = 0;
long long icnt_inj_arrival_READ_REQUEST = 0;
long long icnt_creat_inj_WRITE_REQUEST = 0;
long long icnt_creat_arrival_WRITE_REQUEST = 0;
long long icnt_inj_arrival_WRITE_REQUEST = 0;
long long icnt_creat_inj_READ_REPLY = 0;
long long icnt_creat_arrival_READ_REPLY = 0;
long long icnt_inj_arrival_READ_REPLY = 0;
long long icnt_creat_inj_WRITE_REPLY = 0;
long long icnt_creat_arrival_WRITE_REPLY = 0;
long long icnt_inj_arrival_WRITE_REPLY = 0;
long long icnt_mem_total_time_spend_Ishita = 0;
long long L2_cache_access_total_Ishita = 0;
long long L2_cache_access_miss_Ishita = 0;
long long L2_cache_access_pending_Ishita = 0;
long long L2_cache_access_resfail_Ishita = 0;
long long simple_dram_count = 0;
long long delay_tot_sum = 0;
long long hits_num_total = 0;
long long access_num_total = 0;
long long hits_read_num_total = 0;
long long read_num_total = 0;
long long hits_write_num_total = 0;
long long write_num_total = 0;
long long banks_1time_total = 0;
long long banks_acess_total_Ishita = 0;
long long banks_time_rw_total = 0;
long long banks_access_rw_total_Ishita = 0;
long long banks_time_ready_total = 0;
long long banks_access_ready_total_Ishita = 0;
long long bwutil_total = 0;
long long checked_L2_DRAM_here = 0;
bool print_stall_data = false;
long SHADER_ICNT_PUSH = 0;
long long mem_inst_issue = 0;
long long comp_inst_issue = 0;
long long issued_inst_count = 0;
long long shared_cycle_count = 0;
long long constant_cycle_count = 0;
long long texture_cycle_count = 0;
long long memory_cycle_count = 0;
long long shared_cycle_cycle = 0;
long long constant_cycle_cycle = 0;
long long texture_cycle_cycle = 0;
long long memory_cycle_cycle = 0;
long long texture_issue_cycle = 0;
long long memory_issue_cycle = 0;
long long pushed_from_shader_icnt_l2_icnt = 0;
long long tex_icnt_l2_queue = 0;
long long icnt_ROP_queue = 0;
long long l2_queue_pop = 0;
long long l2_queue_reply = 0;
long long l2_dram_push = 0;
long long l2_dram_rop = 0;
long long l2_icnt_push = 0;
long long l2_dram_queue_pop = 0;
long long push_in_dram = 0;
long long push_from_dram = 0;
long long dram_l2_reached = 0;
long long icnt_back_to_shader = 0;
long long reached_shader_from_icnt = 0;
long long reach_tex_from_l2 = 0;
long long reach_glob_from_icnt = 0;
long long reach_L1_from_tex = 0;
long long reached_global_from_glob = 0;
long long finish_inst = 0;
long long ROP_no_push_l2_queue_push = 0;
long long ROP_extra_cycles = 0;
long long l2_dram_rop_count = 0;
long long NO_INST_ISSUE = 0;
long long opp_for_ooo = 0;
long long opp_for_mem = 0;
long long dram_access_total = 0;
long long dram_write_req_total = 0;
long long dram_read_req_total = 0;

// MEM TOTAL STATS COLLECTION
long long total_row_accesses_NET = 0;
long long total_num_activates_NET = 0;
long long tot_DRAM_reads = 0;
long long tot_DRAM_writes = 0;
long long gpu_stall_dramfull_total = 0;
long long gpu_stall_icnt2sh_total = 0;
long long mf_total_lat_tot = 0;
long long num_mfs_tot = 0;
long long icnt2mem_latency_tot = 0;
long long icnt2sh_latency_tot = 0;
long long n_act_tot = 0;
long long n_pre_tot = 0;
long long n_req_tot = 0;
long long n_rd_tot = 0;
long long n_wr_tot = 0;
long long total_dL1_misses = 0;
long long total_dL1_accesses = 0;
long long L2_total_cache_accesses = 0;
long long L2_total_cache_misses = 0;
long long L2_total_cache_pending_hits = 0;
long long L2_total_cache_reservation_fails = 0;
long long L1I_total_cache_accesses = 0;
long long L1I_total_cache_misses = 0;
long long L1I_total_cache_pending_hits = 0;
long long L1I_total_cache_reservation_fails = 0;
long long L1D_total_cache_accesses = 0;
long long L1D_total_cache_misses = 0;
long long L1D_total_cache_pending_hits = 0;
long long L1C_total_cache_accesses = 0;
long long L1C_total_cache_misses = 0;
long long L1C_total_cache_pending_hits = 0;
long long L1C_total_cache_reservation_fails = 0;
long long L1T_total_cache_accesses = 0;
long long L1T_total_cache_misses = 0;
long long L1T_total_cache_pending_hits = 0;
long long L1T_total_cache_reservation_fails = 0;
long long comp_inst_finish_time = 0;
long long mem_inst_finish_time = 0;
long long ibuffer_flush_count1 = 0;
long long ibuffer_flush_count2 = 0;
long long ibuffer_flush_count3 = 0;
long long replay_flush_count = 0;
long long L1D_total_cache_reservation_fails = 0;
long long opcode_tracer = -1;

long long ICACHE_EMPTY_Kernel = 0;
long long ICACHE_EMPTY_TOTAL = 0;
long long L1I_miss_kernel = 0;
long long L1I_hit_kernel = 0;
long long L1I_miss_TOTAL = 0;
long long L1I_hit_TOTAL = 0; 
long long ibuffer_empty_TOTAL = 0;
long long ibuffer_empty_kernel = 0; 
long long gen_mem_icache_kernel = 0;
bool stalled_on_DEB_dependence = 0;
long long cannot_issue_warp_DEB_dep = 0;
long long cannot_issue_warp_DEB_dep_kernel = 0;

long long tot_cycles_wb = 0;
long long tot_cycles_wb_done = 0;
long long inst_issued_sid_0 = 0;

int inst_counter_warp = 0;
int inst_counter_warp_dec = 0;

int num_mem_fetch_generated = 0;
int num_mem_fetch_generated_useful = 0;
int fetch_misses = 0;
int fetch_hits = 0;
int fetch_resfail = 0;
int control_hazard_count = 0;
int control_hazard_count_kernel = 0;

long long sync_inst_collision = 0;
long long sync_inst_ibuffer_coll = 0;
long long mem_inst_collision = 0;
long long mem_inst_ibuffer_coll = 0;

long long tot_in_order_stall = 0;
long long memory_inst_stuck = 0;
long long non_memory_inst_stuck = 0;
long long load_inst_stuck = 0;
long long store_inst_stuck = 0;
long inst_exec_total = 0;

long gpu_sim_insn_test = 0;
long gpu_sim_insn_test_run = 0;

bool replay_stall = 0;

long total_mem_inst = 0;

// writing the warp issued order to file
ofstream write_warps;
// read data from warp file to execute sched order
ifstream read_warps;
ofstream test_write_warps;
vector<int> warp_sched_order;
long long warp_issued_counter = 0;
long long DEB_BUFFER_SIZE = 1;

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

static int sg_argc = 3;
static const char *sg_argv[] = {"", "-config", "gpgpusim.config"};

void *gpgpu_sim_thread_sequential(void *ctx_ptr) {
  gpgpu_context *ctx = (gpgpu_context *)ctx_ptr;
  // at most one kernel running at a time
  bool done;
  do {
    sem_wait(&(ctx->the_gpgpusim->g_sim_signal_start));
    done = true;
    if (ctx->the_gpgpusim->g_the_gpu->get_more_cta_left()) {
      done = false;
      ctx->the_gpgpusim->g_the_gpu->init();
      while (ctx->the_gpgpusim->g_the_gpu->active()) {
        ctx->the_gpgpusim->g_the_gpu->cycle();
        ctx->the_gpgpusim->g_the_gpu->deadlock_check();
      }
      ctx->the_gpgpusim->g_the_gpu->print_stats();
      ctx->the_gpgpusim->g_the_gpu->update_stats();
      ctx->print_simulation_time();
    }
    sem_post(&(ctx->the_gpgpusim->g_sim_signal_finish));
  } while (!done);
  sem_post(&(ctx->the_gpgpusim->g_sim_signal_exit));
  return NULL;
}

static void termination_callback() {

  write_warps.close();
  std::cout <<"TOTAL CYCLES TAKEN "<<cycles_passed<<"\n";
  cout <<"gpu_sim_insn_test "<<gpu_sim_insn_test<<"\n";
  cout <<"total_mem_inst "<<total_mem_inst<<"\n";
  cout <<"tot_mem_dep_checks "<<tot_mem_dep_checks<<"\n";
  cout <<"tot_mem_dep_true "<<tot_mem_dep_true<<"\n";
  cout <<"tot_mem_dep_false "<<tot_mem_dep_false<<"\n";
  cout <<"tot_issues_ILP "<<tot_issues_ILP<<"\n";
  cout <<"total_times_in_cycle "<<total_times_in_cycle<<"\n";
  cout <<"tot_issues_OOO "<<tot_issues_OOO<<"\n";
  cout <<"tot_in_order_stall "<<tot_in_order_stall<<"\n";
  cout <<"tot_inst_OOO_because_dep "<<tot_inst_OOO_because_dep<<"\n";
  cout <<"WAR_and_WAW_stalls "<<WAR_and_WAW_stalls<<"\n";

  cout <<"STALL_STATS_PER_SCHED\n";

  cout <<"SM sched mem_data comp_data ibuffer comp_str mem_str waiting_warp_stall idle_stall total_cycles total_stalls\n";
  for(int i = 0; i<= SM_num; i++)
  {
    for(int j=0; j<4; j++)
    {
      cout <<i<<" "<<j<<" ";
      for(int k = 0;k<numstall; k++)
      {
        cout <<stallData[i][j][k]<<" ";
      }
      cout <<"\n";
    }
  }
  
  
  cout <<"non_memory_inst_stuck "<<non_memory_inst_stuck<<"\n";
  cout <<"load_inst_stuck "<<load_inst_stuck<<"\n";
  cout <<"store_inst_stuck "<<store_inst_stuck<<"\n";
  cout <<"tot_cycles_wb "<<tot_cycles_wb<<" tot_cycles_wb_done "<<tot_cycles_wb_done<<" inst_issued_sid_0 "<<inst_issued_sid_0<<"\n";
  cout <<"num_mem_fetch_generated "<<num_mem_fetch_generated<<"\n";
  cout <<"num_mem_fetch_generated_useful "<<num_mem_fetch_generated_useful<<"\n";
  cout <<"fetch_misses "<<fetch_misses<<"\n";
  cout <<"fetch_hits "<<fetch_hits<<"\n";
  cout <<"fetch_resfail "<<fetch_resfail<<"\n";
  cout <<"control_hazard_count "<<control_hazard_count<<"\n";
  cout <<"tot_inst_exec "<<tot_inst_exec<<"\n";
  cout <<"ICNT_TO_MEM_cycles "<<ICNT_TO_MEM_cycles<<"\n";
  cout <<"ROP_DELAY_count "<<ROP_DELAY_count<<" ROP_DELAY_cycle "<<ROP_DELAY_cycle<<"\n";
  cout <<"ICNT_TO_L2_QUEUE_count "<<ICNT_TO_L2_QUEUE_count<<" ICNT_TO_L2_QUEUE_cycles "<<ICNT_TO_L2_QUEUE_cycles<<"\n";
  cout <<"L2_TO_DRAM_QUEUE_count "<<L2_TO_DRAM_QUEUE_count<<" L2_TO_DRAM_QUEUE_cycle "<<L2_TO_DRAM_QUEUE_cycle<<"\n";
  cout <<"DRAM_LATENCY_QUEUE_count "<<DRAM_LATENCY_QUEUE_count<<" DRAM_LATENCY_QUEUE_cycle "<<DRAM_LATENCY_QUEUE_cycle<<"\n";
  cout <<"DRAM_TO_L2_QUEUE_count "<<DRAM_TO_L2_QUEUE_count<<" DRAM_TO_L2_QUEUE_cycle "<<DRAM_TO_L2_QUEUE_cycle<<"\n";
  cout <<"DRAM_L2_FILL_QUEUE_count "<<DRAM_L2_FILL_QUEUE_count<<" DRAM_L2_FILL_QUEUE_cycle "<<DRAM_L2_FILL_QUEUE_cycle<<"\n";
  cout <<"L2_TO_ICNT_count "<<L2_TO_ICNT_count<<" L2_TO_ICNT_cycle "<<L2_TO_ICNT_cycle<<"\n";
  cout <<"CLUSTER_TO_SHADER_QUEUE_count "<<CLUSTER_TO_SHADER_QUEUE_count<<" CLUSTER_TO_SHADER_QUEUE_cycle "<<CLUSTER_TO_SHADER_QUEUE_cycle<<"\n";
  cout <<"ICNT_TO_SHADER_count "<<ICNT_TO_SHADER_count<<" ICNT_TO_SHADER_cycles "<<ICNT_TO_SHADER_cycles<<"\n";
  cout <<"CLUSTER_TO_SHADER_QUEUE_1_count "<<CLUSTER_TO_SHADER_QUEUE_1_count<<" CLUSTER_TO_SHADER_QUEUE_1_cycle "<<CLUSTER_TO_SHADER_QUEUE_1_cycle<<"\n";
  cout <<"issued_inst_count "<<issued_inst_count<<"\n";
  cout <<"mem_inst_issue "<<mem_inst_issue<<"\n";
  cout <<"comp_inst_issue "<<comp_inst_issue<<"\n";
  cout <<"tot_cycles_exec_all_SM "<<tot_cycles_exec_all_SM<<"\n";
  printf("final_stall_cycle = %d\n",stall_cycles);
  cout <<"mem_data_stall "<<mem_data_stall<<"\n";
  cout <<"comp_data_stall "<<comp_data_stall<<"\n";
  cout <<"ibuffer_stall "<<ibuffer_stall<<"\n";
  cout <<"comp_str_stall "<<comp_str_stall<<"\n";
  cout <<"mem_str_stall "<<mem_str_stall<<"\n";
  cout <<"waiting_warp "<<waiting_warp<<"\n";
  
  cout <<"sync_inst_collision "<<sync_inst_collision<<'\n';
  cout <<"sync_inst_ibuffer_coll "<<sync_inst_ibuffer_coll<<"\n";
  cout <<"mem_inst_collision "<<mem_inst_collision<<'\n';
  cout <<"mem_inst_ibuffer_coll "<<mem_inst_ibuffer_coll<<"\n";

  cout <<"mem_data_stall_issue_irr "<<mem_data_stall_issue_irr<<"\n";
  cout <<"comp_data_stall_issue_irr "<<comp_data_stall_issue_irr<<"\n";
  cout <<"ibuffer_stall_issue_irr "<<ibuffer_stall_issue_irr<<"\n";
  cout <<"comp_str_stall_issue_irr "<<comp_str_stall_issue_irr<<"\n";
  cout <<"mem_str_stall_issue_irr "<<mem_str_stall_issue_irr<<"\n";
  cout <<"other_stall_issue_irr1 "<<other_stall_issue_irr1<<"\n";
  cout <<"other_stall_issue_irr2 "<<other_stall_issue_irr2<<"\n";
  cout <<"other_stall_issue_irr3 "<<other_stall_issue_irr3<<"\n";

  cout <<"shared_cycle_count "<<shared_cycle_count<<"\n";
  cout <<"constant_cycle_count "<<constant_cycle_count<<"\n";
  cout <<"texture_cycle_count "<<texture_cycle_count<<"\n";
  cout <<"memory_cycle_count "<<memory_cycle_count<<"\n";
  cout <<"shared_cycle_cycle "<<shared_cycle_cycle<<"\n";
  cout <<"constant_cycle_cycle "<<constant_cycle_cycle<<"\n";
  cout <<"texture_cycle_cycle "<<texture_cycle_cycle<<"\n";
  cout <<"memory_cycle_cycle "<<memory_cycle_cycle<<"\n";
  cout <<"texture_issue_cycle "<<texture_issue_cycle<<"\n";
  cout <<"memory_issue_cycle "<<memory_issue_cycle<<"\n";
  cout <<"pushed_from_shader_icnt_l2_icnt "<<pushed_from_shader_icnt_l2_icnt<<"\n";
  cout <<"tex_icnt_l2_queue "<<tex_icnt_l2_queue<<"\n";
  cout <<"icnt_ROP_queue "<<icnt_ROP_queue<<"\n";
  cout <<"l2_queue_pop "<<l2_queue_pop<<"\n";
  cout <<"l2_queue_reply "<<l2_queue_reply<<"\n";
  cout <<"l2_dram_push "<<l2_dram_push<<"\n";
  cout <<"l2_dram_rop "<<l2_dram_rop<<"\n";
  cout <<"l2_icnt_push "<<l2_icnt_push<<"\n";
  cout <<"l2_dram_queue_pop "<<l2_dram_queue_pop<<"\n";
  cout <<"push_in_dram "<<push_in_dram<<"\n";
  cout <<"push_from_dram "<<push_from_dram<<"\n";
  cout <<"dram_l2_reached "<<dram_l2_reached<<"\n";
  cout <<"icnt_back_to_shader "<<icnt_back_to_shader<<"\n";
  cout <<"reached_shader_from_icnt "<<reached_shader_from_icnt<<"\n";
  cout <<"reach_tex_from_l2 "<<reach_tex_from_l2<<"\n";
  cout <<"reach_glob_from_icnt "<<reach_glob_from_icnt<<"\n";
  cout <<"reach_L1_from_tex "<<reach_L1_from_tex<<"\n";
  cout <<"reached_global_from_glob "<<reached_global_from_glob<<"\n";
  cout <<"finish_inst "<<finish_inst<<"\n";
  cout <<"ROP_no_push_l2_queue_push "<<ROP_no_push_l2_queue_push<<"\n";
  cout <<"ROP_extra_cycles "<<ROP_extra_cycles<<"\n";
  cout <<"l2_dram_rop_count "<<l2_dram_rop_count<<"\n";
  cout <<"NO_INST_ISSUE "<<NO_INST_ISSUE<<"\n";
  cout <<"opp_for_ooo "<<opp_for_ooo<<"\n";
  cout <<"opp_for_mem "<<opp_for_mem<<"\n";
  cout <<"dram_access_total "<<dram_access_total<<"\n";
  cout <<"dram_write_req_total "<<dram_write_req_total<<"\n";
  cout <<"dram_read_req_total "<<dram_read_req_total<<"\n";

  std::cout <<"NUMBER OF MEM ISSUES "<<mem_issues<<"\n";
  std::cout <<"MEMORY STALLS SUM "<<mem_cycle_counter<<"\n";
  std::cout <<"l2_cache_bank_access "<<l2_cache_bank_access<<" l2_cache_bank_miss "<<l2_cache_bank_miss<<"\n";
  std::cout <<"l2_cache_access "<<l2_cache_access<<" l2_cache_miss "<<l2_cache_miss<<"\n";
  std::cout <<"l2_pending "<<l2_pending<<" l2_res_fail "<<l2_res_fail<<"\n";
  std::cout <<"c_mem_resource_stall "<<c_mem_resource_stall<<" s_mem_bk_conf "<<s_mem_bk_conf<<" gl_mem_resource_stall "<<gl_mem_resource_stall<<" gl_mem_coal_stall "<<gl_mem_coal_stall<<" gl_mem_data_port_stall "<<gl_mem_data_port_stall<<"\n";


  cout <<"icnt_creat_inj "<<icnt_creat_inj<<"\n";
  cout <<"icnt_creat_arrival "<<icnt_creat_arrival<<"\n";
  cout <<"icnt_inj_arrival "<<icnt_inj_arrival<<"\n";

  cout <<"icnt_creat_inj_READ_REQUEST "<<icnt_creat_inj_READ_REQUEST<<"\n";
  cout <<"icnt_creat_arrival_READ_REQUEST "<<icnt_creat_arrival_READ_REQUEST<<"\n";
  cout <<"icnt_inj_arrival_READ_REQUEST "<<icnt_inj_arrival_READ_REQUEST<<"\n";

  cout <<"icnt_creat_inj_WRITE_REQUEST "<<icnt_creat_inj_WRITE_REQUEST<<"\n";
  cout <<"icnt_creat_arrival_WRITE_REQUEST "<<icnt_creat_arrival_WRITE_REQUEST<<"\n";
  cout <<"icnt_inj_arrival_WRITE_REQUEST "<<icnt_inj_arrival_WRITE_REQUEST<<"\n";

  cout <<"icnt_creat_inj_READ_REPLY "<<icnt_creat_inj_READ_REPLY<<"\n";
  cout <<"icnt_creat_arrival_READ_REPLY "<<icnt_creat_arrival_READ_REPLY<<"\n";
  cout <<"icnt_inj_arrival_READ_REPLY "<<icnt_inj_arrival_READ_REPLY<<"\n";

  cout <<"icnt_creat_inj_WRITE_REPLY "<<icnt_creat_inj_WRITE_REPLY<<"\n";
  cout <<"icnt_creat_arrival_WRITE_REPLY "<<icnt_creat_arrival_WRITE_REPLY<<"\n";
  cout <<"icnt_inj_arrival_WRITE_REPLY "<<icnt_inj_arrival_WRITE_REPLY<<"\n";

  cout <<"icnt_mem_total_time_spend_Ishita "<<icnt_mem_total_time_spend_Ishita<<"\n";

  cout<<"L2_FINAL_STATS_HERE\n";
  cout << "L2_cache_access_total_Ishita "<<L2_cache_access_total_Ishita<<"\n";
  cout << "L2_cache_access_miss_Ishita "<<L2_cache_access_miss_Ishita<<"\n";
  cout << "L2_cache_access_pending_Ishita "<<L2_cache_access_pending_Ishita<<"\n";
  cout << "L2_cache_access_resfail_Ishita "<<L2_cache_access_resfail_Ishita<<"\n";

  std::cout <<"DRAM MEM_STATS_HERE\n";
  cout <<"simple_dram_count "<<simple_dram_count<<"\n";
  cout <<"delay_tot_sum "<<delay_tot_sum<<"\n";
  cout <<"hits_num_total "<<hits_num_total<<"\n";
  cout <<"access_num_total "<<access_num_total<<"\n";
  cout <<"hits_read_num_total "<<hits_read_num_total<<"\n";
  cout <<"read_num_total "<<read_num_total<<"\n";
  cout <<"hits_write_num_total "<<hits_write_num_total<<"\n";
  cout <<"write_num_total "<<write_num_total<<"\n";
  cout <<"banks_1time_total "<<banks_1time_total<<"\n";
  cout <<"banks_acess_total_Ishita "<<banks_acess_total_Ishita<<"\n";
  cout <<"banks_time_rw_total "<<banks_time_rw_total<<"\n";
  cout <<"banks_access_rw_total_Ishita "<<banks_access_rw_total_Ishita<<"\n";
  cout <<"banks_time_ready_total "<<banks_time_ready_total<<"\n";
  cout <<"banks_access_ready_total_Ishita "<<banks_access_ready_total_Ishita<<"\n";

  if(access_num_total)
    printf("\nRow_Buffer_Locality = %.6f", (float)hits_num_total / access_num_total);
  if(read_num_total)
    printf("\nRow_Buffer_Locality_read = %.6f", (float)hits_read_num_total / read_num_total);
  if(write_num_total)
    printf("\nRow_Buffer_Locality_write = %.6f",
         (float)hits_write_num_total / write_num_total);
  if(banks_acess_total_Ishita)
    printf("\nBank_Level_Parallism = %.6f\n",
         (float)banks_1time_total / banks_acess_total_Ishita);
  if(banks_access_rw_total_Ishita)
    printf("\nBank_Level_Parallism_Col = %.6f\n",
         (float)banks_time_rw_total / banks_access_rw_total_Ishita);
  if(banks_access_ready_total_Ishita)
    printf("\nBank_Level_Parallism_Ready = %.6f",
         (float)banks_time_ready_total / banks_access_ready_total_Ishita);

  cout <<"average row locality "<<total_row_accesses_NET <<" "<<total_num_activates_NET<<" ";
  if(total_num_activates_NET>0)
    cout<<float(float(total_row_accesses_NET)/float(total_num_activates_NET));
  cout<<"\n";
  cout <<"tot_DRAM_reads "<<tot_DRAM_reads<<"\n";
  cout <<"tot_DRAM_writes "<<tot_DRAM_writes<<"\n";
  cout <<"bwutil_total "<<bwutil_total<<"\n";
  cout <<"gpu_stall_dramfull_total "<<gpu_stall_dramfull_total<<"\n";
  cout <<"gpu_stall_icnt2sh_total "<<gpu_stall_icnt2sh_total<<"\n";
  if(num_mfs_tot>0)
    printf("FINAL_averagemflatency = %lld \n", mf_total_lat_tot / num_mfs_tot);
  cout <<"icnt2mem_latency_tot "<<icnt2mem_latency_tot<<"\n";
  cout <<"icnt2sh_latency_tot "<<icnt2sh_latency_tot<<"\n";
  cout <<"n_act_tot "<<n_act_tot<<"\n";
  cout <<"n_pre_tot "<<n_pre_tot<<"\n";
  cout <<"n_req_tot "<<n_req_tot<<"\n";
  cout <<"n_wr_tot "<<n_wr_tot<<"\n";
  cout <<"total_dL1_misses "<<total_dL1_misses<<"\n";
  cout <<"total_dL1_accesses "<<total_dL1_accesses<<"\n";
  if(total_dL1_accesses > 0)
  {
    cout <<"total_dL1_miss_rate "<<float(float(total_dL1_misses)/float(total_dL1_accesses)) <<"\n";
  }
  else{
    cout <<"total_dL1_miss_rate 0\n";
  }
  cout <<"L2_total_cache_accesses "<<L2_total_cache_accesses<<"\n";
  cout <<"L2_total_cache_misses "<<L2_total_cache_misses<<"\n";
  if(L2_total_cache_accesses > 0) {
    cout <<"L2_total_cache_miss_rate "<<float(float(L2_total_cache_misses) / float(L2_total_cache_accesses)) <<"\n";
  }
  else {
    cout <<"NONE L2_total_cache_miss_rate 0 \n";
  }
  cout <<"L2_total_cache_reservation_fails "<<L2_total_cache_reservation_fails<<"\n";
  cout <<"L1I_total_cache_accesses "<<L1I_total_cache_accesses<<"\n";
  cout <<"L1I_total_cache_misses "<<L1I_total_cache_misses<<"\n";
  if(L1I_total_cache_accesses > 0) {
    cout <<"L1I_total_cache_miss_rate "<<float(float(L1I_total_cache_misses)/float(L1I_total_cache_accesses)) <<"\n";
  }
  else {
    cout <<"NONE L1I_total_cache_miss_rate 0\n";
  }
  cout <<"L1I_total_cache_pending_hits "<<L1I_total_cache_pending_hits<<"\n";
  cout <<"L1I_total_cache_reservation_fails "<<L1I_total_cache_reservation_fails<<"\n";
  cout <<"L1D_total_cache_accesses "<<L1D_total_cache_accesses<<"\n";
  cout <<"L1D_total_cache_misses "<<L1D_total_cache_misses<<"\n";
  if(L1D_total_cache_accesses> 0) {
    cout <<"L1D_total_cache_miss_rate "<<float(float(L1D_total_cache_misses)/float(L1D_total_cache_accesses)) <<'\n';
  }
  else {
    cout <<"NONE L1D_total_cache_miss_rate 0\n";
  }
  cout <<"L1D_total_cache_pending_hits "<<L1D_total_cache_pending_hits<<"\n";
  cout <<"L1C_total_cache_accesses "<<L1C_total_cache_accesses<<"\n";
  cout <<"L1C_total_cache_misses "<<L1C_total_cache_misses<<"\n";
  if(L1C_total_cache_accesses> 0) {
    cout <<"L1C_total_cache_miss_rate "<<float(float(L1C_total_cache_misses)/float(L1C_total_cache_accesses)) <<'\n';
  }
  else {
    cout <<"NONE L1C_total_cache_miss_rate 0\n";
  }
  cout <<"L1D_total_cache_reservation_fails "<<L1D_total_cache_reservation_fails<<"\n";
  cout <<"L1C_total_cache_pending_hits "<<L1C_total_cache_pending_hits<<"\n";
  cout <<"L1C_total_cache_reservation_fails "<<L1C_total_cache_reservation_fails<<"\n";
  cout <<"L1T_total_cache_accesses "<<L1T_total_cache_accesses<<"\n";
  cout <<"L1T_total_cache_misses "<<L1T_total_cache_misses<<"\n";
  if(L1T_total_cache_accesses> 0) {
    cout <<"L1T_total_cache_miss_rate "<<float(float(L1T_total_cache_misses)/float(L1T_total_cache_accesses)) <<'\n';
  }
  else {
    cout <<"NONE L1T_total_cache_miss_rate 0\n";
  }
  cout <<"L1T_total_cache_pending_hits "<<L1T_total_cache_pending_hits<<"\n";
  cout <<"L1T_total_cache_reservation_fails "<<L1T_total_cache_reservation_fails<<"\n";
  cout <<"comp_inst_finish_time "<<comp_inst_finish_time<<"\n";
  cout <<"mem_inst_finish_time "<<mem_inst_finish_time<<"\n";
  cout <<"ibuffer_flush_count1 "<<ibuffer_flush_count1<<"\n";
  cout <<"ibuffer_flush_count2 "<<ibuffer_flush_count2<<"\n";
  cout <<"ibuffer_flush_count3 "<<ibuffer_flush_count3<<"\n";
  cout <<"replay_flush_count "<<replay_flush_count<<"\n";

  printf("\nGPGPU-Sim: *** exit detected ***\n");
  fflush(stdout);
}

void *gpgpu_sim_thread_concurrent(void *ctx_ptr) {
  gpgpu_context *ctx = (gpgpu_context *)ctx_ptr;
  atexit(termination_callback);

  // Per Shader
  stallData.resize(500,
    // Per scheduler
    vector<vector<int>>(4,
      // Per Stall
      vector<int>(numstall,0)));

    // Per Shader
  str_status.resize(500,
    // Per Sched
    vector<vector<int>>(4,
      // Per str
      vector<int>(8,0)));

  warp_inst_num.resize(500,0);

  act_warp.resize(500, vector<int>(300,0));
  issued_warp.resize(500, vector<int>(4,0));
  warp_issue.resize(64,0);
  icnt_pressure.resize(500,0);
  warps_cannot_be_issued.resize(100,0);
  indep_pc_num_push_all_stalling_inst.resize(100,0);
  stall_consolidated.resize(20,0);

  // concurrent kernel execution simulation thread
  do {
    if (g_debug_execution >= 3) {
      printf(
          "GPGPU-Sim: *** simulation thread starting and spinning waiting for "
          "work ***\n");
      fflush(stdout);
    }
    while (ctx->the_gpgpusim->g_stream_manager->empty_protected() &&
           !ctx->the_gpgpusim->g_sim_done)
      ;
    if (g_debug_execution >= 3) {
      printf("GPGPU-Sim: ** START simulation thread (detected work) **\n");
      ctx->the_gpgpusim->g_stream_manager->print(stdout);
      fflush(stdout);
    }
    pthread_mutex_lock(&(ctx->the_gpgpusim->g_sim_lock));
    ctx->the_gpgpusim->g_sim_active = true;
    pthread_mutex_unlock(&(ctx->the_gpgpusim->g_sim_lock));
    bool active = false;
    bool sim_cycles = false;
    ctx->the_gpgpusim->g_the_gpu->init();
    do {
      // check if a kernel has completed
      // launch operation on device if one is pending and can be run

      // Need to break this loop when a kernel completes. This was a
      // source of non-deterministic behaviour in GPGPU-Sim (bug 147).
      // If another stream operation is available, g_the_gpu remains active,
      // causing this loop to not break. If the next operation happens to be
      // another kernel, the gpu is not re-initialized and the inter-kernel
      // behaviour may be incorrect. Check that a kernel has finished and
      // no other kernel is currently running.
      if (ctx->the_gpgpusim->g_stream_manager->operation(&sim_cycles) &&
          !ctx->the_gpgpusim->g_the_gpu->active())
        break;
      // functional simulation
      if (ctx->the_gpgpusim->g_the_gpu->is_functional_sim()) {
        kernel_info_t *kernel =
            ctx->the_gpgpusim->g_the_gpu->get_functional_kernel();
        assert(kernel);
        ctx->the_gpgpusim->gpgpu_ctx->func_sim->gpgpu_cuda_ptx_sim_main_func(
            *kernel);
        ctx->the_gpgpusim->g_the_gpu->finish_functional_sim(kernel);
      }

      // performance simulation
      if (ctx->the_gpgpusim->g_the_gpu->active()) {
        ctx->the_gpgpusim->g_the_gpu->cycle();
        sim_cycles = true;
        ctx->the_gpgpusim->g_the_gpu->deadlock_check();
      } else {
        if (ctx->the_gpgpusim->g_the_gpu->cycle_insn_cta_max_hit()) {
          ctx->the_gpgpusim->g_stream_manager->stop_all_running_kernels();
          ctx->the_gpgpusim->g_sim_done = true;
          ctx->the_gpgpusim->break_limit = true;
        }
      }
      active = ctx->the_gpgpusim->g_the_gpu->active() ||
               !(ctx->the_gpgpusim->g_stream_manager->empty_protected());

    }
    while (active && !ctx->the_gpgpusim->g_sim_done);
    if (g_debug_execution >= 3) {
      printf("GPGPU-Sim: ** STOP simulation thread (no work) **\n");
      fflush(stdout);
    }
    if (sim_cycles) {
      ctx->the_gpgpusim->g_the_gpu->print_stats();
      ctx->the_gpgpusim->g_the_gpu->update_stats();
      ctx->print_simulation_time();
    }
    pthread_mutex_lock(&(ctx->the_gpgpusim->g_sim_lock));
    ctx->the_gpgpusim->g_sim_active = false;
    pthread_mutex_unlock(&(ctx->the_gpgpusim->g_sim_lock));
  } while (!ctx->the_gpgpusim->g_sim_done);

  printf("GPGPU-Sim: *** simulation thread exiting ***\n");
  fflush(stdout);

  if (ctx->the_gpgpusim->break_limit) {
    printf(
        "GPGPU-Sim: ** break due to reaching the maximum cycles (or "
        "instructions) **\n");
    exit(1);
  }

  sem_post(&(ctx->the_gpgpusim->g_sim_signal_exit));
  return NULL;
}

void gpgpu_context::synchronize() {
  printf("GPGPU-Sim: synchronize waiting for inactive GPU simulation\n");
  the_gpgpusim->g_stream_manager->print(stdout);
  fflush(stdout);
  //    sem_wait(&g_sim_signal_finish);
  bool done = false;
  do {
    pthread_mutex_lock(&(the_gpgpusim->g_sim_lock));
    done = (the_gpgpusim->g_stream_manager->empty() &&
            !the_gpgpusim->g_sim_active) ||
           the_gpgpusim->g_sim_done;
    pthread_mutex_unlock(&(the_gpgpusim->g_sim_lock));
  } while (!done);
  printf("GPGPU-Sim: detected inactive GPU simulation thread\n");
  fflush(stdout);
  //    sem_post(&g_sim_signal_start);
}

void gpgpu_context::exit_simulation() {
  the_gpgpusim->g_sim_done = true;
  printf("GPGPU-Sim: exit_simulation called\n");
  fflush(stdout);
  sem_wait(&(the_gpgpusim->g_sim_signal_exit));
  printf("GPGPU-Sim: simulation thread signaled exit\n");
  fflush(stdout);
}

gpgpu_sim *gpgpu_context::gpgpu_ptx_sim_init_perf() {
  srand(1);
  print_splash();
  func_sim->read_sim_environment_variables();
  ptx_parser->read_parser_environment_variables();
  option_parser_t opp = option_parser_create();

  ptx_reg_options(opp);
  func_sim->ptx_opcocde_latency_options(opp);

  icnt_reg_options(opp);
  the_gpgpusim->g_the_gpu_config = new gpgpu_sim_config(this);
  the_gpgpusim->g_the_gpu_config->reg_options(
      opp);  // register GPU microrachitecture options

  option_parser_cmdline(opp, sg_argc, sg_argv);  // parse configuration options
  fprintf(stdout, "GPGPU-Sim: Configuration options:\n\n");
  option_parser_print(opp, stdout);
  // Set the Numeric locale to a standard locale where a decimal point is a
  // "dot" not a "comma" so it does the parsing correctly independent of the
  // system environment variables
  assert(setlocale(LC_NUMERIC, "C"));
  the_gpgpusim->g_the_gpu_config->init();

  the_gpgpusim->g_the_gpu =
      new exec_gpgpu_sim(*(the_gpgpusim->g_the_gpu_config), this);
  the_gpgpusim->g_stream_manager = new stream_manager(
      (the_gpgpusim->g_the_gpu), func_sim->g_cuda_launch_blocking);

  the_gpgpusim->g_simulation_starttime = time((time_t *)NULL);

  sem_init(&(the_gpgpusim->g_sim_signal_start), 0, 0);
  sem_init(&(the_gpgpusim->g_sim_signal_finish), 0, 0);
  sem_init(&(the_gpgpusim->g_sim_signal_exit), 0, 0);

  return the_gpgpusim->g_the_gpu;
}

void gpgpu_context::start_sim_thread(int api) {
  if (the_gpgpusim->g_sim_done) {
    the_gpgpusim->g_sim_done = false;
    if (api == 1) {
      pthread_create(&(the_gpgpusim->g_simulation_thread), NULL,
                     gpgpu_sim_thread_concurrent, (void *)this);
    } else {
      pthread_create(&(the_gpgpusim->g_simulation_thread), NULL,
                     gpgpu_sim_thread_sequential, (void *)this);
    }
  }
}

void gpgpu_context::print_simulation_time() {
  time_t current_time, difference, d, h, m, s;
  current_time = time((time_t *)NULL);
  difference = MAX(current_time - the_gpgpusim->g_simulation_starttime, 1);

  d = difference / (3600 * 24);
  h = difference / 3600 - 24 * d;
  m = difference / 60 - 60 * (h + 24 * d);
  s = difference - 60 * (m + 60 * (h + 24 * d));

  fflush(stderr);
  printf(
      "\n\ngpgpu_simulation_time = %u days, %u hrs, %u min, %u sec (%u sec)\n",
      (unsigned)d, (unsigned)h, (unsigned)m, (unsigned)s, (unsigned)difference);
  printf("gpgpu_simulation_rate = %u (inst/sec)\n",
         (unsigned)(the_gpgpusim->g_the_gpu->gpu_tot_sim_insn / difference));
  const unsigned cycles_per_sec =
      (unsigned)(the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle / difference);
  printf("gpgpu_simulation_rate = %u (cycle/sec)\n", cycles_per_sec);
  printf("gpgpu_silicon_slowdown = %ux\n",
         the_gpgpusim->g_the_gpu->shader_clock() * 1000 / cycles_per_sec);
  fflush(stdout);
}

int gpgpu_context::gpgpu_opencl_ptx_sim_main_perf(kernel_info_t *grid) {
  the_gpgpusim->g_the_gpu->launch(grid);
  sem_post(&(the_gpgpusim->g_sim_signal_start));
  sem_wait(&(the_gpgpusim->g_sim_signal_finish));
  return 0;
}

//! Functional simulation of OpenCL
/*!
 * This function call the CUDA PTX functional simulator
 */
int cuda_sim::gpgpu_opencl_ptx_sim_main_func(kernel_info_t *grid) {
  // calling the CUDA PTX simulator, sending the kernel by reference and a flag
  // set to true, the flag used by the function to distinguish OpenCL calls from
  // the CUDA simulation calls which it is needed by the called function to not
  // register the exit the exit of OpenCL kernel as it doesn't register entering
  // in the first place as the CUDA kernels does
  gpgpu_cuda_ptx_sim_main_func(*grid, true);
  return 0;
}
