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


//   long long cycle_num;
// long total_times_in_cycle = 0;
// vector<vector<vector<int>>>stallData;
// vector<vector<int>>act_warp;
// vector<vector<int>>issued_warp;
// vector<vector<vector<int>>> str_status; 
// vector<int>warp_issue;
// vector<int>icnt_pressure;
// vector<int>warps_cannot_be_issued;
// vector<unsigned>warp_inst_num;
// vector<int> indep_pc_num_push_all_stalling_inst;

// vector<int> stall_consolidated;

// long long inst_counter = 0;
// long long max_active;
// long long actw;
// long long max_warps_act;
// long long cycles_passed = 0;
// long long max_sid;
// long long num_of_schedulers;
// long long numstall = 12;
// long long print_on = 0;
// long long going_from_shader_to_mem = 0;
// long long present_ongoing_cycle = 0;
// long long stall_cycles = 0;
// long long tot_icnt_buffer = 0;
// long long tot_inst_exec = 0;
// long long tot_cycles_exec_all_SM = 0;
// long long tot_inst_ret = 0;
// extern int total_warps = 0;

// int tot_mem_dep_checks = 0;
// int tot_mem_dep_true = 0;
// int tot_mem_dep_false = 0;
// int time_bw_inst = 0; 
// int reached_barrier = 0;

// int tot_issues_ILP = 0;
// int tot_issues_OOO = 0;

// // Stats collection
// long long mem_data_stall = 0;
// long long comp_data_stall = 0;
// long long ibuffer_stall = 0;
// long long comp_str_stall = 0;
// long long mem_str_stall = 0;
// long long mem_data_stall_kernel = 0;
// long long tot_inst_OOO_because_dep = 0;
// long long ibuffer_stall_kernel = 0;
// long long comp_str_stall_kernel = 0;
// long long mem_str_stall_kernel = 0;
// long long waiting_warp = 0;
// long long SM_num = 0;
// long long idle = 0;
// long long other_stall1_kernel = 0;
// long long other_stall2_kernel= 0;
// long long other_stall3_kernel = 0;
// long long ooo_opp_kernel = 0;
// long long mem_data_stall_issue_irr = 0;
// long long comp_data_stall_issue_irr = 0;
// long long ibuffer_stall_issue_irr = 0;
// long long comp_str_stall_issue_irr = 0;
// long long mem_str_stall_issue_irr = 0;
// long long other_stall_issue_irr1 = 0;
// long long other_stall_issue_irr2 = 0;
// long long other_stall_issue_irr3 = 0;
// long WAR_and_WAW_stalls = 0;
// bool WAR_or_WAW_found = 0;
// long long ICNT_TO_MEM_cycles = 0;
// long long ICNT_TO_MEM_cycles_kernel = 0;
// long long ICNT_TO_SHADER_count = 0;
// long long ICNT_TO_SHADER_cycles = 0;
// long long ICNT_TO_SHADER_count_kernel = 0;
// long long ICNT_TO_SHADER_cycles_kernel = 0;
// long long ROP_DELAY_count = 0;
// long long ROP_DELAY_cycle = 0;
// long long ROP_DELAY_count_kernel = 0;
// long long ROP_DELAY_cycle_kernel = 0;
// long long ICNT_TO_L2_QUEUE_count = 0;
// long long ICNT_TO_L2_QUEUE_count_kernel = 0;
// long long ICNT_TO_L2_QUEUE_cycles = 0;
// long long ICNT_TO_L2_QUEUE_cycles_kernel = 0;
// long long L2_TO_DRAM_QUEUE_count = 0;
// long long L2_TO_DRAM_QUEUE_cycle = 0;
// long long DRAM_LATENCY_QUEUE_count = 0;
// long long DRAM_LATENCY_QUEUE_cycle = 0;
// long long DRAM_LATENCY_QUEUE_count_kernel = 0;
// long long DRAM_LATENCY_QUEUE_cycle_kernel = 0;
// long long DRAM_TO_L2_QUEUE_count = 0;
// long long DRAM_TO_L2_QUEUE_cycle = 0;
// long long DRAM_TO_L2_QUEUE_count_kernel = 0;
// long long DRAM_TO_L2_QUEUE_cycle_kernel = 0;
// long long DRAM_L2_FILL_QUEUE_count = 0;
// long long DRAM_L2_FILL_QUEUE_cycle = 0;
// long long DRAM_L2_FILL_QUEUE_count_kernel = 0;
// long long DRAM_L2_FILL_QUEUE_cycle_kernel = 0;
// long long L2_TO_ICNT_count = 0;
// long long L2_TO_ICNT_cycle = 0;
// long long L2_TO_ICNT_count_kernel = 0;
// long long L2_TO_ICNT_cycle_kernel = 0;
// long long CLUSTER_TO_SHADER_QUEUE_count = 0;
// long long CLUSTER_TO_SHADER_QUEUE_cycle = 0;
// long long CLUSTER_TO_SHADER_QUEUE_count_kernel = 0;
// long long CLUSTER_TO_SHADER_QUEUE_cycle_kernel                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 = 0;
// long long CLUSTER_TO_SHADER_QUEUE_1_count = 0;
// long long CLUSTER_TO_SHADER_QUEUE_1_cycle = 0;
// long long mem_issues = 0;
// long long mem_cycle_counter = 0;
// long long l2_cache_bank_access = 0;
// long long l2_cache_bank_miss = 0;
// long long l2_cache_access = 0;
// long long l2_cache_miss = 0;
// long long l2_pending = 0;
// long long l2_res_fail = 0;
// long long c_mem_resource_stall = 0;
// long long s_mem_bk_conf = 0;
// long long gl_mem_resource_stall = 0;
// long long gl_mem_coal_stall = 0;
// long long gl_mem_data_port_stall = 0;
// long long icnt_creat_inj = 0;
// long long icnt_creat_arrival = 0;
// long long icnt_inj_arrival = 0;
// long long icnt_creat_inj_READ_REQUEST = 0;
// long long icnt_creat_arrival_READ_REQUEST = 0;
// long long icnt_inj_arrival_READ_REQUEST = 0;
// long long icnt_creat_inj_WRITE_REQUEST = 0;
// long long icnt_creat_arrival_WRITE_REQUEST = 0;
// long long icnt_inj_arrival_WRITE_REQUEST = 0;
// long long icnt_creat_inj_READ_REPLY = 0;
// long long icnt_creat_arrival_READ_REPLY = 0;
// long long icnt_inj_arrival_READ_REPLY = 0;
// long long icnt_creat_inj_WRITE_REPLY = 0;
// long long icnt_creat_arrival_WRITE_REPLY = 0;
// long long icnt_inj_arrival_WRITE_REPLY = 0;
// long long icnt_mem_total_time_spend_Ishita = 0;
// long long L2_cache_access_total_Ishita = 0;
// long long L2_cache_access_miss_Ishita = 0;
// long long L2_cache_access_pending_Ishita = 0;
// long long L2_cache_access_resfail_Ishita = 0;
// long long simple_dram_count = 0;
// long long delay_tot_sum = 0;
// long long hits_num_total = 0;
// long long access_num_total = 0;
// long long hits_read_num_total = 0;
// long long read_num_total = 0;
// long long hits_write_num_total = 0;
// long long write_num_total = 0;
// long long banks_1time_total = 0;
// long long banks_acess_total_Ishita = 0;
// long long banks_time_rw_total = 0;
// long long banks_access_rw_total_Ishita = 0;
// long long banks_time_ready_total = 0;
// long long banks_access_ready_total_Ishita = 0;
// long long bwutil_total = 0;
// long long checked_L2_DRAM_here = 0;
// bool print_stall_data = false;
// long SHADER_ICNT_PUSH = 0;
// long long mem_inst_issue = 0;
// long long comp_inst_issue = 0;
// long long issued_inst_count = 0;
// long long shared_cycle_count = 0;
// long long constant_cycle_count = 0;
// long long texture_cycle_count = 0;
// long long memory_cycle_count = 0;
// long long shared_cycle_cycle = 0;
// long long constant_cycle_cycle = 0;
// long long texture_cycle_cycle = 0;
// long long memory_cycle_cycle = 0;
// long long texture_issue_cycle = 0;
// long long memory_issue_cycle = 0;
// long long pushed_from_shader_icnt_l2_icnt = 0;
// long long tex_icnt_l2_queue = 0;
// long long icnt_ROP_queue = 0;
// long long l2_queue_pop = 0;
// long long l2_queue_reply = 0;
// long long l2_dram_push = 0;
// long long l2_dram_rop = 0;
// long long l2_icnt_push = 0;
// long long l2_dram_queue_pop = 0;
// long long push_in_dram = 0;
// long long push_from_dram = 0;
// long long dram_l2_reached = 0;
// long long icnt_back_to_shader = 0;
// long long reached_shader_from_icnt = 0;
// long long reach_tex_from_l2 = 0;
// long long reach_glob_from_icnt = 0;
// long long reach_L1_from_tex = 0;
// long long reached_global_from_glob = 0;
// long long finish_inst = 0;
// long long ROP_no_push_l2_queue_push = 0;
// long long ROP_extra_cycles = 0;
// long long l2_dram_rop_count = 0;
// long long NO_INST_ISSUE = 0;
// long long opp_for_ooo = 0;
// long long opp_for_mem = 0;
// long long dram_access_total = 0;
// long long dram_write_req_total = 0;
// long long dram_read_req_total = 0;

// // MEM TOTAL STATS COLLECTION
// long long total_row_accesses_NET = 0;
// long long total_num_activates_NET = 0;
// long long tot_DRAM_reads = 0;
// long long tot_DRAM_writes = 0;
// long long gpu_stall_dramfull_total = 0;
// long long gpu_stall_icnt2sh_total = 0;
// long long mf_total_lat_tot = 0;
// long long num_mfs_tot = 0;
// long long icnt2mem_latency_tot = 0;
// long long icnt2sh_latency_tot = 0;
// long long n_act_tot = 0;
// long long n_pre_tot = 0;
// long long n_req_tot = 0;
// long long n_rd_tot = 0;
// long long n_wr_tot = 0;
// long long total_dL1_misses = 0;
// long long total_dL1_accesses = 0;
// long long L2_total_cache_accesses = 0;
// long long L2_total_cache_misses = 0;
// long long L2_total_cache_pending_hits = 0;
// long long L2_total_cache_reservation_fails = 0;
// long long L1I_total_cache_accesses = 0;
// long long L1I_total_cache_misses = 0;
// long long L1I_total_cache_pending_hits = 0;
// long long L1I_total_cache_reservation_fails = 0;
// long long L1D_total_cache_accesses = 0;
// long long L1D_total_cache_misses = 0;
// long long L1D_total_cache_pending_hits = 0;
// long long L1C_total_cache_accesses = 0;
// long long L1C_total_cache_misses = 0;
// long long L1C_total_cache_pending_hits = 0;
// long long L1C_total_cache_reservation_fails = 0;
// long long L1T_total_cache_accesses = 0;
// long long L1T_total_cache_misses = 0;
// long long L1T_total_cache_pending_hits = 0;
// long long L1T_total_cache_reservation_fails = 0;
// long long comp_inst_finish_time = 0;
// long long mem_inst_finish_time = 0;
// long long ibuffer_flush_count1 = 0;
// long long ibuffer_flush_count2 = 0;
// long long ibuffer_flush_count3 = 0;
// long long replay_flush_count = 0;
// long long L1D_total_cache_reservation_fails = 0;
// long long opcode_tracer = -1;

// long long ICACHE_EMPTY_Kernel = 0;
// long long ICACHE_EMPTY_TOTAL = 0;
// long long L1I_miss_kernel = 0;
// long long L1I_hit_kernel = 0;
// long long L1I_miss_TOTAL = 0;
// long long L1I_hit_TOTAL = 0; 
// long long ibuffer_empty_TOTAL = 0;
// long long ibuffer_empty_kernel = 0; 
// long long gen_mem_icache_kernel = 0;
// bool stalled_on_DEB_dependence = 0;
// long long cannot_issue_warp_DEB_dep = 0;
// long long cannot_issue_warp_DEB_dep_kernel = 0;

// long long tot_cycles_wb = 0;
// long long tot_cycles_wb_done = 0;
// long long inst_issued_sid_0 = 0;

// int inst_counter_warp = 0;
// int inst_counter_warp_dec = 0;

// int num_mem_fetch_generated = 0;
// int num_mem_fetch_generated_useful = 0;
// int fetch_misses = 0;
// int fetch_hits = 0;
// int fetch_resfail = 0;
// int control_hazard_count = 0;
// int control_hazard_count_kernel = 0;

// long long sync_inst_collision = 0;
// long long sync_inst_ibuffer_coll = 0;
// long long mem_inst_collision = 0;
// long long mem_inst_ibuffer_coll = 0;

// long long tot_in_order_stall = 0;
// long long memory_inst_stuck = 0;
// long long non_memory_inst_stuck = 0;
// long long load_inst_stuck = 0;
// long long store_inst_stuck = 0;
// long inst_exec_total = 0;

// long gpu_sim_insn_test = 0;
// long gpu_sim_insn_test_run = 0;

// bool replay_stall = 0;

// long total_mem_inst = 0;

// writing the warp issued order to file
// read data from warp file to execute sched order
vector<int> warp_sched_order;
long long warp_issued_counter = 0;
long long DEB_BUFFER_SIZE = 1;


gpgpu_sim *gpgpu_trace_sim_init_perf_model(int argc, const char *argv[],
                                           gpgpu_context *m_gpgpu_context,
                                           class trace_config *m_config);

trace_kernel_info_t *create_kernel_info( kernel_trace_t* kernel_trace_info,
		                      gpgpu_context *m_gpgpu_context, class trace_config *config,
							  trace_parser *parser);


int main(int argc, const char **argv) {
  std::cout << "Accel-Sim [build " << g_accelsim_version << "]";
  gpgpu_context *m_gpgpu_context = new gpgpu_context();
  trace_config tconfig;

  gpgpu_sim *m_gpgpu_sim =
      gpgpu_trace_sim_init_perf_model(argc, argv, m_gpgpu_context, &tconfig);
  m_gpgpu_sim->init();

  trace_parser tracer(tconfig.get_traces_filename());

  tconfig.parse_config();

  // Per Shader
  // stallData.resize(500,
  //   // Per scheduler
  //   vector<vector<int>>(4,
  //     // Per Stall
  //     vector<int>(numstall,0)));

  //   // Per Shader
  // str_status.resize(500,
  //   // Per Sched
  //   vector<vector<int>>(4,
  //     // Per str
  //     vector<int>(8,0)));

  // warp_inst_num.resize(500,0);

  // act_warp.resize(500, vector<int>(300,0));
  // issued_warp.resize(500, vector<int>(4,0));
  // warp_issue.resize(64,0);
  // icnt_pressure.resize(500,0);
  // warps_cannot_be_issued.resize(100,0);
  // indep_pc_num_push_all_stalling_inst.resize(100,0);
  // stall_consolidated.resize(20,0);

  // for each kernel
  // load file
  // parse and create kernel info
  // launch
  // while loop till the end of the end kernel execution
  // prints stats
  bool concurrent_kernel_sm =  m_gpgpu_sim->getShaderCoreConfig()->gpgpu_concurrent_kernel_sm;
  unsigned window_size = concurrent_kernel_sm ? m_gpgpu_sim->get_config().get_max_concurrent_kernel() : 1;
  assert(window_size > 0);
  std::vector<trace_command> commandlist = tracer.parse_commandlist_file();
  std::vector<unsigned long> busy_streams;
  std::vector<trace_kernel_info_t*> kernels_info;
  kernels_info.reserve(window_size);

  unsigned i = 0;
  while (i < commandlist.size() || !kernels_info.empty()) {
    trace_kernel_info_t *kernel_info = NULL;
    if (commandlist[i].m_type == command_type::cpu_gpu_mem_copy) {
      size_t addre, Bcount;
      tracer.parse_memcpy_info(commandlist[i].command_string, addre, Bcount);
      std::cout << "launching memcpy command : " << commandlist[i].command_string << std::endl;
      m_gpgpu_sim->perf_memcpy_to_gpu(addre, Bcount);
      i++;
      continue;
    } else if (commandlist[i].m_type == command_type::kernel_launch) {
      // Read trace header info for window_size number of kernels
      while (kernels_info.size() < window_size && i < commandlist.size()) {
        kernel_trace_t* kernel_trace_info = tracer.parse_kernel_info(commandlist[i].command_string);
        kernel_info = create_kernel_info(kernel_trace_info, m_gpgpu_context, &tconfig, &tracer);
        kernels_info.push_back(kernel_info);
        std::cout << "Header info loaded for kernel command : " << commandlist[i].command_string << std::endl;
        i++;
      }
      
      // Launch all kernels within window that are on a stream that isn't already running
      for (auto k : kernels_info) {
        bool stream_busy = false;
        for (auto s: busy_streams) {
          if (s == k->get_cuda_stream_id())
            stream_busy = true;
        }
        if (!stream_busy && m_gpgpu_sim->can_start_kernel() && !k->was_launched()) {
          std::cout << "launching kernel name: " << k->name() << " uid: " << k->get_uid() << std::endl;
          m_gpgpu_sim->launch(k);
          k->set_launched();
          busy_streams.push_back(k->get_cuda_stream_id());
        }
      }
    }
    else if (kernels_info.empty())
    	assert(0 && "Undefined Command");

    bool active = false;
    bool sim_cycles = false;
    unsigned finished_kernel_uid = 0;

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
      finished_kernel_uid = m_gpgpu_sim->finished_kernel();
    } while (active && !finished_kernel_uid);

    // cleanup finished kernel
    if (finished_kernel_uid) {
      trace_kernel_info_t* k = NULL;
      for (unsigned j = 0; j < kernels_info.size(); j++) {
        k = kernels_info.at(j);
        if (k->get_uid() == finished_kernel_uid) {
          for (int l = 0; l < busy_streams.size(); l++) {
            if (busy_streams.at(l) == k->get_cuda_stream_id()) {
              busy_streams.erase(busy_streams.begin()+l);
              break;
            }
          }
          tracer.kernel_finalizer(k->get_trace_info());
          delete k->entry();
          delete k;
          kernels_info.erase(kernels_info.begin()+j);
          break;
        }
      }
      assert(k);
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

  // we print this message to inform the gpgpu-simulation stats_collect script
  // that we are done
  printf("GPGPU-Sim: *** simulation thread exiting ***\n");
  printf("GPGPU-Sim: *** exit detected ***\n");
  fflush(stdout);

  return 0;
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