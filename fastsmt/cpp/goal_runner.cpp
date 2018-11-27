#include "z3++.h"
#include <map>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <ctime>

using namespace z3;

std::vector<std::string> split(std::string s, std::string delim) {
  std::vector<std::string> ret;
  while (s.find(delim) != std::string::npos)  {
    int pos = s.find(delim);
    ret.push_back(s.substr(0, pos));
    s = s.substr(pos+(int)delim.size());
  }
  if (s.size() > 0) {
    ret.push_back(s);
  }
  return ret;
}

tactic fromString(context& ctx, std::string s) {
  if (s.find("Tactic(") == 0) {
    return tactic(ctx, s.substr(7, (int)s.size() - 8).c_str());
  } else if (s.find("AndThen(") == 0) {
    auto tokens = split(s.substr(8, (int)s.size() - 9), ",");
    
    tactic ret = fromString(ctx, tokens[0]);
    
    for (size_t i = 1; i < tokens.size(); ++i) {
      auto tactic = fromString(ctx, tokens[i]);
      ret = ret & tactic;
    }

    return ret;
  } else if (s.find("With(") == 0) {
    auto tokens = split(s.substr(5, (int)s.size() - 6), ";");
    auto tactic_name = tokens[0];

    params p(ctx);

    for (size_t i = 1; i < tokens.size(); ++i) {
      auto tmp_tokens = split(tokens[i], "=");
      std::string x = tmp_tokens[0];
      std::string val = tmp_tokens[1];

      if (val == "True") {
	p.set(x.c_str(), true);
      } else if (val == "False") {
	p.set(x.c_str(), false);
      } else {
	p.set(x.c_str(), (unsigned int)std::stoi(val));
      }
    }

    return with(tactic(ctx, tactic_name.c_str()), p);
  }
  return tactic(ctx, s.c_str());
}

int get_rlimit(solver s) {
  auto stats = s.statistics();
  auto sz = stats.size();
  
  for (size_t i = 0; i < sz; ++i) {
    if (stats.key(i) == "rlimit count") {
      return stats.uint_value(i);
    }
  }

  return 0;
}

int main(int argc, char* argv[]) {
  context ctx;
  
  char* strategy = argv[1];
  char* smt_input_file = argv[2];
  char* smt_out_file = argv[3];
  
  Z3_ast a = Z3_parse_smtlib2_file(ctx, smt_input_file, 0, 0, 0, 0, 0, 0);    
  expr f(ctx, a);

  tactic t = fromString(ctx, strategy);

  auto rlimit_before = get_rlimit(t.mk_solver());

  goal g(ctx);
  g.add(f);
  auto old_hash = g.as_expr().hash();

  clock_t begin = clock();

  try {
    apply_result r = t(g);
    assert(r.size() == 1); // assert that there is only 1 resulting goal
    goal new_goal = r[0];
    auto new_hash = new_goal.as_expr().hash();

    clock_t end = clock();

    auto rlimit_after = get_rlimit(t.mk_solver());
    auto rlimit = rlimit_after - rlimit_before;

    Z3_ast *assumptionsArray = NULL;
    auto out_string = Z3_benchmark_to_smtlib_string(ctx,
						    "benchmark",
						    "",
						    "unknown",
						    "",
						    0,
						    assumptionsArray,
						    new_goal.as_expr()
						    );

    std::string res = "unknown";
    if (new_goal.is_decided_sat()) {
      res = "sat";
    } else if (new_goal.is_decided_unsat()) {
      res = "unsat";
    }

    FILE *smt_out = fopen(smt_out_file, "w");
    fprintf(smt_out, "%s\n", out_string);
    fclose(smt_out);

    double elapsed_sec = (double)(end - begin) / CLOCKS_PER_SEC;

    std::cout << res << " " << rlimit << " " << old_hash << " " << new_hash << " " << elapsed_sec << std::endl;
    
    std::vector<probe> probes;
    probes.push_back(probe(ctx, "num-consts"));
    probes.push_back(probe(ctx, "num-exprs"));
    probes.push_back(probe(ctx, "size"));
    probes.push_back(probe(ctx, "depth"));
    probes.push_back(probe(ctx, "ackr-bound-probe"));
    probes.push_back(probe(ctx, "is-qfbv-eq"));
    probes.push_back(probe(ctx, "arith-max-deg"));
    probes.push_back(probe(ctx, "arith-avg-deg"));
    probes.push_back(probe(ctx, "arith-max-bw"));
    probes.push_back(probe(ctx, "arith-avg-bw"));
    probes.push_back(probe(ctx, "is-unbounded"));
    probes.push_back(probe(ctx, "is-pb"));
    probes.push_back(probe(ctx, "num-bv-consts"));
    probes.push_back(probe(ctx, "num-arith-consts"));
    probes.push_back(probe(ctx, "is-qfbv-eq"));
  
    for (size_t i = 0; i < probes.size(); ++i) {
      if (i != 0) {
	std::cout << " ";
      }
      std::cout << probes[i](new_goal);
    }
    std::cout << std::endl;
  } catch (z3::exception) {
    std::cout << -1 << std::endl;
    return 0;
  }
  
  return 0;
}
