// Copyright (c) 2009-2011, Tor M. Aamodt, Inderpreet Singh
// The University of British Columbia
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

#include "scoreboard.h"
#include "../cuda-sim/ptx_sim.h"
#include "shader.h"
#include "shader_trace.h"

// Constructor
Scoreboard::Scoreboard(unsigned sid, unsigned n_warps, class gpgpu_t* gpu)
    : longopregs() {
  m_sid = sid;
  // Initialize size of table
  reg_table.resize(n_warps);
  longopregs.resize(n_warps);

  /* New variables for FAST */
  reg_table_mem.resize(n_warps);
  reg_table_comp.resize(n_warps);

  reg_reserved_mem.resize(n_warps);
  reg_type_mem.resize(n_warps);
  reg_load_type.resize(n_warps);
  reg_released_mem.resize(n_warps);
  reg_reserved_comp.resize(n_warps);
  reg_released_comp.resize(n_warps);
  reg_reserved.resize(n_warps);
  reg_released.resize(n_warps);

  reg_reserved_type.resize(n_warps);

  m_gpu = gpu;
}

// Print scoreboard contents
void Scoreboard::printContents() const {
  printf("scoreboard contents (sid=%d): \n", m_sid);
  for (unsigned i = 0; i < reg_table.size(); i++) {
    if (reg_table[i].size() == 0) continue;
    printf("  wid = %2d: ", i);
    std::set<unsigned>::const_iterator it;
    for (it = reg_table[i].begin(); it != reg_table[i].end(); it++)
      printf("%u ", *it);
    printf("\n");
  }
}

void Scoreboard::reserveRegister(unsigned wid, unsigned regnum, bool gpgpu_perfect_mem_data) {
  if (!(reg_table[wid].find(regnum) == reg_table[wid].end()) && !gpgpu_perfect_mem_data) {
    printf(
        "Error: trying to reserve an already reserved register (sid=%d, "
        "wid=%d, regnum=%d).",
        m_sid, wid, regnum);
    abort();
  }
  reg_reserved[wid][regnum] = cycles_passed;
  SHADER_DPRINTF(SCOREBOARD, "Reserved Register - warp:%d, reg: %d\n", wid,
                 regnum);
  reg_table[wid].insert(regnum);
}

// Unmark register as write-pending
void Scoreboard::releaseRegister(unsigned wid, unsigned regnum) {
  if (!(reg_table[wid].find(regnum) != reg_table[wid].end())) return;
  SHADER_DPRINTF(SCOREBOARD, "Release register - warp:%d, reg: %d\n", wid,
                 regnum);
  reg_table[wid].erase(regnum);
  mem_issues = mem_issues + 1;
  mem_cycle_counter = mem_cycle_counter + (cycles_passed - reg_reserved[wid].find(regnum)->second);
}

const bool Scoreboard::islongop(unsigned warp_id, unsigned regnum) {
  return longopregs[warp_id].find(regnum) != longopregs[warp_id].end();
}

void Scoreboard::reserveRegisters(const class warp_inst_t* inst, bool gpgpu_perfect_mem_data, int status) {
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      reserveRegister(inst->warp_id(), inst->out[r],gpgpu_perfect_mem_data);
      SHADER_DPRINTF(SCOREBOARD, "Reserved register - warp:%d, reg: %d\n",
                     inst->warp_id(), inst->out[r]);
    }
  }
  // Keep track of long operations
  if (inst->is_load() && (inst->space.get_type() == global_space ||
                          inst->space.get_type() == local_space ||
                          inst->space.get_type() == param_space_kernel ||
                          inst->space.get_type() == param_space_local ||
                          inst->space.get_type() == param_space_unclassified ||
                          inst->space.get_type() == tex_space)) {
    for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
      if (inst->out[r] > 0) {
        SHADER_DPRINTF(SCOREBOARD, "New longopreg marked - warp:%d, reg: %d\n",
                       inst->warp_id(), inst->out[r]);
        longopregs[inst->warp_id()].insert(inst->out[r]);
      }
    }
  }
}

// Release registers for an instruction
void Scoreboard::releaseRegisters(const class warp_inst_t* inst) {
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      SHADER_DPRINTF(SCOREBOARD, "Register Released - warp:%d, reg: %d\n",
                     inst->warp_id(), inst->out[r]);
      releaseRegister(inst->warp_id(), inst->out[r]);
      longopregs[inst->warp_id()].erase(inst->out[r]);
    }
  }
}

/**
 * Checks to see if registers used by an instruction are reserved in the
 *scoreboard
 *
 * @return
 * true if WAW or RAW hazard (no WAR since in-order issue)
 **/
bool Scoreboard::checkCollision(unsigned wid, const class inst_t* inst, bool print) const {
  // Get list of all input and output registers
  std::set<int> inst_regs;

  for (unsigned iii = 0; iii < inst->outcount; iii++)
    inst_regs.insert(inst->out[iii]);

  for (unsigned jjj = 0; jjj < inst->incount; jjj++)
    inst_regs.insert(inst->in[jjj]);

  if (inst->pred > 0) inst_regs.insert(inst->pred);
  if (inst->ar1 > 0) inst_regs.insert(inst->ar1);
  if (inst->ar2 > 0) inst_regs.insert(inst->ar2);

  // Check for collision, get the intersection of reserved registers and
  // instruction registers
  std::set<int>::iterator it2;
  std::set<unsigned>::const_iterator it;
  
  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    if (reg_table[wid].find(*it2) != reg_table[wid].end()) {
      return true;
    }
  }

  return false;
}

bool Scoreboard::pendingWrites(unsigned wid) const {
  return !reg_table[wid].empty();
}

bool Scoreboard::pendingWrites(unsigned wid, bool ignore) const {
  if (ignore) return false;
  return !reg_table[wid].empty();
}

/* Added Functions */
/* Check if instructions collide with an instruction in the replay queue  
*  @return
* true if WAR or WAW hazard
*/
bool Scoreboard::checkReplayCollision(unsigned wid, const class inst_t* inst, std::vector<const warp_inst_t *> replayInst) const {
  std::set<int> inst_regs;
  std::set<int> inst_replay_regs;

  for (unsigned iii = 0; iii < inst->outcount; iii++)
    inst_regs.insert(inst->out[iii]);

  for (unsigned jjj = 0; jjj < inst->incount; jjj++)
    inst_regs.insert(inst->in[jjj]);

  if (inst->pred > 0) inst_regs.insert(inst->pred);
  if (inst->ar1 > 0) inst_regs.insert(inst->ar1);
  if (inst->ar2 > 0) inst_regs.insert(inst->ar2);

  // make list of all regs in replay list
  for(const class inst_t* ins : replayInst) 
  {
    for (unsigned iii = 0; iii < ins->outcount; iii++)
      inst_replay_regs.insert(ins->out[iii]);

    for (unsigned jjj = 0; jjj < ins->incount; jjj++)
      inst_replay_regs.insert(ins->in[jjj]);
  }

  // check for collision against replay queue instructions
  std::set<int>::const_iterator it2;
  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    if (inst_replay_regs.find(*it2) != inst_replay_regs.end()) {
      return true;
    }
  }
  return false;

}

//unmark registers as write Pending for mem operations
void Scoreboard::releaseRegisterMem(unsigned wid, unsigned regnum,int val,int op, int type) {
  if (!(reg_table_mem[wid].find(regnum) != reg_table_mem[wid].end())) return;
  reg_table_mem[wid].erase(regnum);
  reg_released_mem[wid][regnum] = cycles_passed;
}

//unmark registers as write pending for comp operations
void Scoreboard::releaseRegisterComp(unsigned wid, unsigned regnum) {
  if (!(reg_table_comp[wid].find(regnum) != reg_table_comp[wid].end())) return;
  reg_table_comp[wid].erase(regnum);

  reg_released_comp[wid][regnum] = cycles_passed;
  //comp_reg_reserve_cycle = comp_reg_reserve_cycle + (cycles_passed - reg_reserved_comp[wid][regnum]);
}

void Scoreboard::reserveRegisterMem(unsigned wid, unsigned regnum, bool is_load) {
  reg_table_mem[wid].insert(regnum);
  reg_reserved_mem[wid][regnum] = cycles_passed;
  reg_reserved_type[wid][regnum] = 1;
}

void Scoreboard::reserveRegisterComp(unsigned wid, unsigned regnum) {
  reg_table_comp[wid].insert(regnum);
  reg_reserved_comp[wid][regnum] = cycles_passed;
  reg_reserved_type[wid][regnum] = 2;
}

void Scoreboard::reserveRegistersMem(const class warp_inst_t* inst) {
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      reserveRegisterMem(inst->warp_id(), inst->out[r], inst->is_load());
      SHADER_DPRINTF(SCOREBOARD, "Reserved register - warp:%d, reg: %d\n",
                     inst->warp_id(), inst->out[r]);
    }
  }
}

void Scoreboard::reserveRegistersComp(const class warp_inst_t* inst) {
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      reserveRegisterComp(inst->warp_id(), inst->out[r]);
      SHADER_DPRINTF(SCOREBOARD, "Reserved register - warp:%d, reg: %d\n",
                     inst->warp_id(), inst->out[r]);
    }
  }
}

// Release registers for a mem instruction
void Scoreboard::releaseRegistersMem(const class warp_inst_t* inst,int val) {
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      releaseRegisterMem(inst->warp_id(), inst->out[r],val,inst->op,inst->space.get_type());
    }
  }
}

// Release registers for a comp instruction
void Scoreboard::releaseRegistersComp(const class warp_inst_t* inst) {
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      releaseRegisterComp(inst->warp_id(), inst->out[r]);
    }
  }
}

void Scoreboard::appendMemStatus(warp_inst_t &inst,int type)
{
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst.out[r] > 0) {
      reg_type_mem[inst.warp_id()][inst.out[r]] = type;
      reg_load_type[inst.warp_id()][inst.out[r]] = inst.is_load();
    }
  }
}

bool Scoreboard::pendingWritesMem(unsigned wid) const {
  return !reg_table_mem[wid].empty();
}

bool Scoreboard::pendingWritesComp(unsigned wid) const {
  return !reg_table_comp[wid].empty();
}

std::vector<int> Scoreboard::checkCollisionMem(unsigned wid, const class inst_t* inst) const {
  // Get list of all input and output registers
  std::set<int> inst_regs;
  std::vector<int> result;
  result.resize(3);

  for (unsigned iii = 0; iii < inst->outcount; iii++)
    inst_regs.insert(inst->out[iii]);

  for (unsigned jjj = 0; jjj < inst->incount; jjj++)
    inst_regs.insert(inst->in[jjj]);

  if (inst->pred > 0) inst_regs.insert(inst->pred);
  if (inst->ar1 > 0) inst_regs.insert(inst->ar1);
  if (inst->ar2 > 0) inst_regs.insert(inst->ar2);

  // Check for collision, get the intersection of reserved registers and
  // instruction registers
  std::set<int>::const_iterator it2;

  // check for longest taking release Reg and print the corresponding reserve and release cycle with the register number
  int reserve_c = -1;
  int release_c = -1;
  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    unsigned regnum = *it2;
    if (reg_reserved_mem[wid].count(regnum) == 0) continue;
    int reserve = reg_reserved_mem[wid].find(regnum)->second;

    if (reg_released_mem[wid].count(regnum) == 0) continue;
    int release = reg_released_mem[wid].find(regnum)->second;

    // Take latest reservation that was released
    if (reserve <= release && release > release_c)
    {
      reserve_c = reserve;
      release_c = release;
    }
  }
  result[1] = reserve_c;
  result[2] = release_c;

  std::set<unsigned>::const_iterator it;

    bool mem_data_col = false;
  bool comp_data_col = false;

  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    unsigned regnum = *it2;
    if (reg_table[wid].find(*it2) != reg_table[wid].end()) {
      if (reg_reserved_type[wid].find(regnum)->second == 1)
        mem_data_col = true;
      if (reg_reserved_type[wid].find(regnum)->second == 2)
        comp_data_col = true;
    }
  }

  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    if (reg_table_mem[wid].find(*it2) != reg_table_mem[wid].end()) {
	    result[0] = 1;
      return result;
    }
  }
  result[0] = 0;
  return result;
} 

std::vector<int> Scoreboard::checkCollisionComp(unsigned wid, const class inst_t* inst) const {
  // Get list of all input and output registers
  std::set<int> inst_regs;
  std::vector<int> result;
  result.resize(3);

  for (unsigned iii = 0; iii < inst->outcount; iii++)
    inst_regs.insert(inst->out[iii]);

  for (unsigned jjj = 0; jjj < inst->incount; jjj++)
    inst_regs.insert(inst->in[jjj]);

  if (inst->pred > 0) inst_regs.insert(inst->pred);
  if (inst->ar1 > 0) inst_regs.insert(inst->ar1);
  if (inst->ar2 > 0) inst_regs.insert(inst->ar2);

  // Check for collision, get the intersection of reserved registers and
  // instruction registers
  std::set<int>::const_iterator it2;

  // check for longest taking release Reg and print the corresponding reserve and release cycle with the register number
  int reserve_c = -1;
  int release_c = -1;
  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    unsigned regnum = *it2;
    if (reg_reserved_comp[wid].count(regnum) == 0) continue;
    int reserve = reg_reserved_comp[wid].find(regnum)->second;

    if (reg_released_comp[wid].count(regnum) == 0) continue;
    int release = reg_released_comp[wid].find(regnum)->second;

    // Take latest reservation that was released
    if (reserve <= release && release > release_c)
    {
      reserve_c = reserve;
      release_c = release;
    }
  }
  result[1] = reserve_c;
  result[2] = release_c;
  
    bool mem_data_col = false;
  bool comp_data_col = false;

  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    unsigned regnum = *it2;
    if (reg_table[wid].find(*it2) != reg_table[wid].end()) {
      if (reg_reserved_type[wid].find(regnum)->second == 1)
        mem_data_col = true;
      if (reg_reserved_type[wid].find(regnum)->second == 2)
        comp_data_col = true;
    }
  }

  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    if (reg_table_comp[wid].find(*it2) != reg_table_comp[wid].end()) {
      result[0] = 1;
      return result;
    }
  }
  result[0] = 0;
  return result;
}

bool Scoreboard::checkConsecutiveInstIndep(const class inst_t* inst, const class inst_t *inst1) const{
  
  std::set<int> inst_regs;
  std::set<int> inst_regs1;
  int a = 0;

  for (unsigned iii = 0; iii < inst->outcount; iii++)
    inst_regs.insert(inst->out[iii]);

  for (unsigned jjj = 0; jjj < inst->incount; jjj++)
    inst_regs.insert(inst->in[jjj]);

  for (unsigned iii = 0; iii < inst1->outcount; iii++)
    inst_regs1.insert(inst1->out[iii]);

  for (unsigned jjj = 0; jjj < inst1->incount; jjj++)
    inst_regs1.insert(inst1->in[jjj]);  

  std::set<int>::const_iterator it2;
  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
  {
    if (inst_regs1.find(*it2) != inst_regs1.end()) {
      return false;
    }
  }
  return true;
}


