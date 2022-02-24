// developed by Mahmoud Khairy, Purdue Univ
// abdallm@purdue.edu

#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <time.h>
#include <vector>

#include "gpgpu_context.h"
#include "abstract_hardware_model.h"
#include "cuda-sim/cuda-sim.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "gpgpusim_entrypoint.h"
#include "option_parser.h"
#include "../ISA_Def/trace_opcode.h"
#include "trace_driven.h"
#include "../trace-parser/trace_parser.h"
#include "accelsim_version.h"

/* TO DO:
 * NOTE: the current version of trace-driven is functionally working fine,
 * but we still need to improve traces compression and simulation speed.
 * This includes:
 *
 * 1- Prefetch concurrent thread that prefetches traces from disk (to not be
 * limited by disk speed)
 *
 * 2- traces compression format a) cfg format and remove
 * thread/block Id from the head and b) using zlib library to save in binary format
 *
 * 3- Efficient memory improvement (save string not objects - parse only 10 in
 * the buffer)
 *
 * 4- Seeking capability - thread scheduler (save tb index and warp
 * index info in the traces header)
 *
 * 5- Get rid off traces intermediate files -
 * change the tracer
 */

#include "gpgpu-sim/src/gpgpu-sim/fast.h"

using namespace std;

#include <vector>
#include <fstream>
using std::vector;

int cycle_num;
vector<vector<vector<int>>>stallData;
vector<vector<int>>act_warp;
vector<vector<vector<int>>> str_status; 
vector<int>warp_issue;
vector<int>icnt_pressure;
int inst_counter = 0;
int max_active;
int actw;
int max_warps_act;
int cycles_passed = 0;
int max_sid;
int num_of_schedulers;
int numstall = 19;
int print_on = 0;
int going_from_shader_to_mem = 0;
int present_ongoing_cycle = 0;
int stall_cycles = 0;
int tot_icnt_buffer = 0;
int tot_inst_exec = 0;
int tot_cycles_exec_all_SM = 0;
int tot_inst_ret = 0;

// Stats collection
int mem_data_stall = 0;
int comp_data_stall = 0;
int ibuffer_stall = 0;
int comp_str_stall = 0;
int mem_str_stall = 0;
int other_stall1 = 0;
int other_stall2 = 0;
int other_stall3 = 0;
int mem_data_stall_issue_irr = 0;
int comp_data_stall_issue_irr = 0;
int ibuffer_stall_issue_irr = 0;
int comp_str_stall_issue_irr = 0;
int mem_str_stall_issue_irr = 0;
int other_stall_issue_irr1 = 0;
int other_stall_issue_irr2 = 0;
int other_stall_issue_irr3 = 0;
int ICNT_TO_MEM_count = 0;
int ICNT_TO_MEM_cycles = 0;
int ICNT_TO_SHADER_count = 0;
int ICNT_TO_SHADER_cycles = 0;
int ROP_DELAY_count = 0;
int ROP_DELAY_cycle = 0;
int ICNT_TO_L2_QUEUE_count = 0;
int ICNT_TO_L2_QUEUE_cycles = 0;
int L2_TO_DRAM_QUEUE_count = 0;
int L2_TO_DRAM_QUEUE_cycle = 0;
int DRAM_LATENCY_QUEUE_count = 0;
int DRAM_LATENCY_QUEUE_cycle = 0;
int DRAM_TO_L2_QUEUE_count = 0;
int DRAM_TO_L2_QUEUE_cycle = 0;
int DRAM_L2_FILL_QUEUE_count = 0;
int DRAM_L2_FILL_QUEUE_cycle = 0;
int L2_TO_ICNT_count = 0;
int L2_TO_ICNT_cycle = 0;
int CLUSTER_TO_SHADER_QUEUE_count = 0;
int CLUSTER_TO_SHADER_QUEUE_cycle = 0;
int CLUSTER_TO_SHADER_QUEUE_1_count = 0;
int CLUSTER_TO_SHADER_QUEUE_1_cycle = 0;
int mem_issues = 0;
int mem_cycle_counter = 0;
int l2_cache_bank_access = 0;
int l2_cache_bank_miss = 0;
int l2_cache_access = 0;
int l2_cache_miss = 0;
int l2_pending = 0;
int l2_res_fail = 0;
int c_mem_resource_stall = 0;
int s_mem_bk_conf = 0;
int gl_mem_resource_stall = 0;
int gl_mem_coal_stall = 0;
int gl_mem_data_port_stall = 0;
int icnt_creat_inj = 0;
int icnt_creat_arrival = 0;
int icnt_inj_arrival = 0;
int icnt_creat_inj_READ_REQUEST = 0;
int icnt_creat_arrival_READ_REQUEST = 0;
int icnt_inj_arrival_READ_REQUEST = 0;
int icnt_creat_inj_WRITE_REQUEST = 0;
int icnt_creat_arrival_WRITE_REQUEST = 0;
int icnt_inj_arrival_WRITE_REQUEST = 0;
int icnt_creat_inj_READ_REPLY = 0;
int icnt_creat_arrival_READ_REPLY = 0;
int icnt_inj_arrival_READ_REPLY = 0;
int icnt_creat_inj_WRITE_REPLY = 0;
int icnt_creat_arrival_WRITE_REPLY = 0;
int icnt_inj_arrival_WRITE_REPLY = 0;
int icnt_mem_total_time_spend_Ishita = 0;
int L2_cache_access_total_Ishita = 0;
int L2_cache_access_miss_Ishita = 0;
int L2_cache_access_pending_Ishita = 0;
int L2_cache_access_resfail_Ishita = 0;
int simple_dram_count = 0;
int delay_tot_sum = 0;
int hits_num_total = 0;
int access_num_total = 0;
int hits_read_num_total = 0;
int read_num_total = 0;
int hits_write_num_total = 0;
int write_num_total = 0;
int banks_1time_total = 0;
int banks_acess_total_Ishita = 0;
int banks_time_rw_total = 0;
int banks_access_rw_total_Ishita = 0;
int banks_time_ready_total = 0;
int banks_access_ready_total_Ishita = 0;
int bwutil_total = 0;
int checked_L2_DRAM_here = 0;
bool print_stall_data = false;
int SHADER_ICNT_PUSH = 0;
int mem_inst_issue = 0;
int comp_inst_issue = 0;
int issued_inst_count = 0;
int shared_cycle_count = 0;
int constant_cycle_count = 0;
int texture_cycle_count = 0;
int memory_cycle_count = 0;
int shared_cycle_cycle = 0;
int constant_cycle_cycle = 0;
int texture_cycle_cycle = 0;
int memory_cycle_cycle = 0;
int texture_issue_cycle = 0;
int memory_issue_cycle = 0;
int pushed_from_shader_icnt_l2_icnt = 0;
int tex_icnt_l2_queue = 0;
int icnt_ROP_queue = 0;
int l2_queue_pop = 0;
int l2_queue_reply = 0;
int l2_dram_push = 0;
int l2_dram_rop = 0;
int l2_icnt_push = 0;
int l2_dram_queue_pop = 0;
int push_in_dram = 0;
int push_from_dram = 0;
int dram_l2_reached = 0;
int icnt_back_to_shader = 0;
int reached_shader_from_icnt = 0;
int reach_tex_from_l2 = 0;
int reach_glob_from_icnt = 0;
int reach_L1_from_tex = 0;
int reached_global_from_glob = 0;
int finish_inst = 0;
int ROP_no_push_l2_queue_push = 0;
int ROP_extra_cycles = 0;
int l2_dram_rop_count = 0;
int NO_INST_ISSUE = 0;
int opp_for_ooo = 0;
int opp_for_mem = 0;
int dram_access_total = 0;
int dram_write_req_total = 0;
int dram_read_req_total = 0;

// MEM TOTAL STATS COLLECTION
int total_row_accesses_NET = 0;
int total_num_activates_NET = 0;
int tot_DRAM_reads = 0;
int tot_DRAM_writes = 0;
int gpu_stall_dramfull_total = 0;
int gpu_stall_icnt2sh_total = 0;
int mf_total_lat_tot = 0;
int num_mfs_tot = 0;
int icnt2mem_latency_tot = 0;
int icnt2sh_latency_tot = 0;
int n_act_tot = 0;
int n_pre_tot = 0;
int n_req_tot = 0;
int n_rd_tot = 0;
int n_wr_tot = 0;
int total_dL1_misses = 0;
int total_dL1_accesses = 0;
int L2_total_cache_accesses = 0;
int L2_total_cache_misses = 0;
int L2_total_cache_pending_hits = 0;
int L2_total_cache_reservation_fails = 0;
int L1I_total_cache_accesses = 0;
int L1I_total_cache_misses = 0;
int L1I_total_cache_pending_hits = 0;
int L1I_total_cache_reservation_fails = 0;
int L1D_total_cache_accesses = 0;
int L1D_total_cache_misses = 0;
int L1D_total_cache_pending_hits = 0;
int L1C_total_cache_accesses = 0;
int L1C_total_cache_misses = 0;
int L1C_total_cache_pending_hits = 0;
int L1C_total_cache_reservation_fails = 0;
int L1T_total_cache_accesses = 0;
int L1T_total_cache_misses = 0;
int L1T_total_cache_pending_hits = 0;
int L1T_total_cache_reservation_fails = 0;
int comp_inst_finish_time = 0;
int mem_inst_finish_time = 0;
int ibuffer_flush_count1 = 0;
int ibuffer_flush_count2 = 0;
int ibuffer_flush_count3 = 0;
int replay_flush_count = 0;
int L1D_total_cache_reservation_fails = 0;
int DEB_BUFFER_SIZE = 1;

// writing the warp issued order to file
ofstream write_warps;
// read data from warp file to execute sched order
ifstream read_warps;
ofstream test_write_warps;
vector<int> warp_sched_order;
int warp_issued_counter = 0;
int opcode_tracer = -1;

gpgpu_sim *gpgpu_trace_sim_init_perf_model(int argc, const char *argv[],
                                           gpgpu_context *m_gpgpu_context,
                                           class trace_config *m_config);

trace_kernel_info_t *create_kernel_info( kernel_trace_t* kernel_trace_info,
		                      gpgpu_context *m_gpgpu_context, class trace_config *config,
							  trace_parser *parser);


int main(int argc, const char **argv) {

  // Per Shader
  stallData.resize(500,
    // Per Warp
    vector<vector<int>>(300,
      // Per Stall
      vector<int>(numstall,0)));

    // Per Shader
  str_status.resize(500,
    // Per Sched
    vector<vector<int>>(4,
      // Per str
      vector<int>(8,0)));

  act_warp.resize(500, vector<int>(300,0));
  warp_issue.resize(64,0);
  icnt_pressure.resize(500,0);


  gpgpu_context *m_gpgpu_context = new gpgpu_context();
  trace_config tconfig;

  gpgpu_sim *m_gpgpu_sim =
      gpgpu_trace_sim_init_perf_model(argc, argv, m_gpgpu_context, &tconfig);
  m_gpgpu_sim->init();

  trace_parser tracer(tconfig.get_traces_filename());

  tconfig.parse_config();

  // for each kernel
  // load file
  // parse and create kernel info
  // launch
  // while loop till the end of the end kernel execution
  // prints stats

  std::vector<trace_command> commandlist = tracer.parse_commandlist_file();

  for (unsigned i = 0; i < commandlist.size(); ++i) {
    trace_kernel_info_t *kernel_info = NULL;
    if (commandlist[i].m_type == command_type::cpu_gpu_mem_copy) {
      size_t addre, Bcount;
      tracer.parse_memcpy_info(commandlist[i].command_string, addre, Bcount);
      std::cout << "launching memcpy command : " << commandlist[i].command_string << std::endl;
      m_gpgpu_sim->perf_memcpy_to_gpu(addre, Bcount);
      continue;
    } else if (commandlist[i].m_type == command_type::kernel_launch) {
      kernel_trace_t kernel_trace_info = tracer.parse_kernel_info(commandlist[i].command_string);
      kernel_info = create_kernel_info(&kernel_trace_info, m_gpgpu_context, &tconfig, &tracer);
      std::cout << "launching kernel command : " << commandlist[i].command_string << std::endl;
      m_gpgpu_sim->launch(kernel_info);
    }
    else
    	assert(0 && "Undefined Command");

    bool active = false;
    bool sim_cycles = false;

    do {
      if (!m_gpgpu_sim->active())
        break;

      // performance simulation
      if (m_gpgpu_sim->active()) {
        m_gpgpu_sim->cycle();
        sim_cycles = true;
        m_gpgpu_sim->deadlock_check();
      } else {
        if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
          m_gpgpu_context->the_gpgpusim->g_stream_manager
              ->stop_all_running_kernels();
          break;
        }
      }

      active = m_gpgpu_sim->active();

    } while (active);

    if (kernel_info) {
      delete kernel_info->entry();
      delete kernel_info;
      tracer.kernel_finalizer();
      m_gpgpu_sim->print_stats();
    }

    if (sim_cycles) {
      m_gpgpu_sim->update_stats();
      m_gpgpu_context->print_simulation_time();
    }

    if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
      printf("GPGPU-Sim: ** break due to reaching the maximum cycles (or "
             "instructions) **\n");
      fflush(stdout);
      break;
    }
  }

  //write_warps.close();
  std::cout <<"TOTAL CYCLES TAKEN "<<cycles_passed<<"\n";
  cout <<"tot_inst_exec "<<tot_inst_exec<<"\n";
  cout <<"ICNT_TO_MEM_count "<<ICNT_TO_MEM_count<<" ICNT_TO_MEM_cycles "<<ICNT_TO_MEM_cycles<<"\n";
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
  cout <<"SHADER_ICNT_PUSH "<<SHADER_ICNT_PUSH<<"\n";
  cout <<"mem_inst_issue "<<mem_inst_issue<<"\n";
  cout <<"comp_inst_issue "<<comp_inst_issue<<"\n";
  cout <<"mem_data_stall "<<mem_data_stall<<"\n";
  cout <<"comp_data_stall "<<comp_data_stall<<"\n";
  cout <<"ibuffer_stall "<<ibuffer_stall<<"\n";
  cout <<"comp_str_stall "<<comp_str_stall<<"\n";
  cout <<"mem_str_stall "<<mem_str_stall<<"\n";
  cout <<"other_stall1 "<<other_stall1<<"\n";
  cout <<"other_stall2 "<<other_stall2<<"\n";
  cout <<"other_stall3 "<<other_stall3<<"\n";
  cout <<"tot_cycles_exec_all_SM "<<tot_cycles_exec_all_SM<<"\n";

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

  // we print this message to inform the gpgpu-simulation stats_collect script
  // that we are done
  printf("GPGPU-Sim: *** simulation thread exiting ***\n");
  printf("GPGPU-Sim: *** exit detected ***\n");
  fflush(stdout);

  return 1;
}


trace_kernel_info_t *create_kernel_info( kernel_trace_t* kernel_trace_info,
		                      gpgpu_context *m_gpgpu_context, class trace_config *config,
							  trace_parser *parser){

  gpgpu_ptx_sim_info info;
  info.smem = kernel_trace_info->shmem;
  info.regs = kernel_trace_info->nregs;
  dim3 gridDim(kernel_trace_info->grid_dim_x, kernel_trace_info->grid_dim_y, kernel_trace_info->grid_dim_z);
  dim3 blockDim(kernel_trace_info->tb_dim_x, kernel_trace_info->tb_dim_y, kernel_trace_info->tb_dim_z);
  trace_function_info *function_info =
      new trace_function_info(info, m_gpgpu_context);
  function_info->set_name(kernel_trace_info->kernel_name.c_str());
  trace_kernel_info_t *kernel_info =
      new trace_kernel_info_t(gridDim, blockDim, function_info,
    		  parser, config, kernel_trace_info);

  return kernel_info;
}

gpgpu_sim *gpgpu_trace_sim_init_perf_model(int argc, const char *argv[],
                                           gpgpu_context *m_gpgpu_context,
                                           trace_config *m_config) {
  srand(1);
  print_splash();

  option_parser_t opp = option_parser_create();

  m_gpgpu_context->ptx_reg_options(opp);
  m_gpgpu_context->func_sim->ptx_opcocde_latency_options(opp);

  icnt_reg_options(opp);

  m_gpgpu_context->the_gpgpusim->g_the_gpu_config =
      new gpgpu_sim_config(m_gpgpu_context);
  m_gpgpu_context->the_gpgpusim->g_the_gpu_config->reg_options(
      opp); // register GPU microrachitecture options
  m_config->reg_options(opp);

  option_parser_cmdline(opp, argc, argv); // parse configuration options
  fprintf(stdout, "GPGPU-Sim: Configuration options:\n\n");
  option_parser_print(opp, stdout);
  // Set the Numeric locale to a standard locale where a decimal point is a
  // "dot" not a "comma" so it does the parsing correctly independent of the
  // system environment variables
  assert(setlocale(LC_NUMERIC, "C"));
  m_gpgpu_context->the_gpgpusim->g_the_gpu_config->init();

  m_gpgpu_context->the_gpgpusim->g_the_gpu = new trace_gpgpu_sim(
      *(m_gpgpu_context->the_gpgpusim->g_the_gpu_config), m_gpgpu_context);

  m_gpgpu_context->the_gpgpusim->g_stream_manager =
      new stream_manager((m_gpgpu_context->the_gpgpusim->g_the_gpu),
                         m_gpgpu_context->func_sim->g_cuda_launch_blocking);

  m_gpgpu_context->the_gpgpusim->g_simulation_starttime = time((time_t *)NULL);

  return m_gpgpu_context->the_gpgpusim->g_the_gpu;
}

