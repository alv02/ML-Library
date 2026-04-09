#!/bin/bash
set -e

mkdir -p build

# ── Compile library objects ───────────────────────────────────────────────────
g++ -I include -c src/tensor.cpp    -o build/tensor.o
g++ -I include -c src/autograd.cpp  -o build/autograd.o
g++ -I include -c src/ops.cpp       -o build/ops.o
g++ -I include -c src/tensor_iterator.cpp       -o build/tensorIterator.o

LIB_OBJS="build/tensor.o build/autograd.o build/ops.o build/tensorIterator.o"

# ── Generate test data ────────────────────────────────────────────────────────
echo "=== Generating test data ==="
(cd test && python3 gen_test_data.py)

# ── Run tests ─────────────────────────────────────────────────────────────────
TOTAL_PASS=0
TOTAL_FAIL=0

run_test() {
    local name=$1
    local src=$2

    echo ""
    echo "=== $name ==="
    g++ -I include "$src" $LIB_OBJS -o "build/test_$name"

    # run from test/ so ../data/ paths resolve correctly
    if (cd test && "../build/test_$name"); then
        TOTAL_PASS=$((TOTAL_PASS + 1))
    else
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
}

run_test broadcast test/broadcast.cpp

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================"
echo "$TOTAL_PASS test file(s) passed, $TOTAL_FAIL failed"
echo "================================"

[ $TOTAL_FAIL -eq 0 ]
