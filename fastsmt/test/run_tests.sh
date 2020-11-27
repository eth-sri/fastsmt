#!/bin/sh
rm -f test/test_outcome.txt
./cpp/goal_runner "AndThen(With(simplify;blast_distinct=True;elim_and=False;flat=False;hoist_mul=False;local_ctx=True;pull_cheap_ite=True;push_ite_bv=True;som=True),Tactic(qfnra-nlsat),Tactic(sat),With(propagate-values;push_ite_bv=True),Tactic(max-bv-sharing),With(aig;aig_per_assertion=False),Tactic(smt))" test/bench_7150.smt2 test/out.smt2 | head -1 | awk '{ print $1 }' >> test/test_outcome.txt
./cpp/goal_runner "AndThen(With(simplify;blast_distinct=True;elim_and=False;flat=False;hoist_mul=True;local_ctx=False;pull_cheap_ite=True;push_ite_bv=True;som=True),Tactic(purify-arith),With(aig;aig_per_assertion=False),Tactic(elim-uncnstr),Tactic(smt))" test/bench_9849.smt2 test/out.smt2 | head -1 | awk '{ print $1 }' >> test/test_outcome.txt
./cpp/goal_runner "AndThen(With(simplify;blast_distinct=True;elim_and=False;flat=False;hoist_mul=True;local_ctx=False;pull_cheap_ite=True;push_ite_bv=False;som=True),Tactic(purify-arith),With(aig;aig_per_assertion=False),Tactic(bit-blast),Tactic(smt))" test/bench_9582.smt2 test/out.smt2 | head -1 | awk '{ print $1 }' >> test/test_outcome.txt
./cpp/goal_runner "AndThen(With(simplify;blast_distinct=True;elim_and=False;flat=False;hoist_mul=True;local_ctx=False;pull_cheap_ite=True;push_ite_bv=True;som=True),With(aig;aig_per_assertion=False),Tactic(qfnra-nlsat),Tactic(smt))" test/bench_1015.smt2 test/out.smt2 | head -1 | awk '{ print $1 }' >> test/test_outcome.txt
./cpp/goal_runner "AndThen(With(simplify;blast_distinct=True;elim_and=False;flat=False;hoist_mul=True;local_ctx=False;pull_cheap_ite=True;push_ite_bv=True;som=True),Tactic(elim-uncnstr),Tactic(solve-eqs),With(aig;aig_per_assertion=False),Tactic(smt))" test/bench_1504.smt2 test/out.smt2 | head -1 | awk '{ print $1 }' >> test/test_outcome.txt

if diff test/test_outcome.txt test/target_outcome.txt; then
    echo "OK!"
else
    echo "ERROR!"
    echo "-> Test outputs: "
    cat test/test_outcome.txt
    echo "-> Target outputs: "
    cat test/target_outcome.txt
fi
