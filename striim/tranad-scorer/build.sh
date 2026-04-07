#!/bin/bash
# =============================================================
# TranAD Scorer - Build Script (WAEvent pass-through, no types JAR)
# =============================================================
# Run this from the tranad-scorer/ directory.
#
# Prerequisites:
#   - Maven installed (brew install maven)
#   - Striim installed at /opt/Striim
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

STRIIM_HOME="${STRIIM_HOME:-/opt/Striim}"

echo "=== Step 1: Install Striim SDK into local Maven repo ==="

SDK_JAR="$STRIIM_HOME/StriimSDK/StriimOpenProcessor-SDK.jar"

if [ ! -f "$SDK_JAR" ]; then
    echo "ERROR: SDK jar not found at $SDK_JAR"
    exit 1
fi

mvn install:install-file \
    -DgroupId=com.striim \
    -DartifactId=OpenProcessorSDK \
    -Dversion=1.0.0-SNAPSHOT \
    -Dpackaging=jar \
    -Dfile="$SDK_JAR" \
    -DgeneratePom=true

echo ""
echo "=== Step 1b: Install Striim Common (runtime WAEvent class) ==="

COMMON_JAR="$STRIIM_HOME/lib/Common-5.2.0.4.jar"

if [ ! -f "$COMMON_JAR" ]; then
    echo "ERROR: Common jar not found at $COMMON_JAR"
    exit 1
fi

mvn install:install-file \
    -DgroupId=com.striim \
    -DartifactId=Common \
    -Dversion=5.2.0.4 \
    -Dpackaging=jar \
    -Dfile="$COMMON_JAR" \
    -DgeneratePom=true

echo ""
echo "=== Step 2: Build the Open Processor ==="

mvn clean package

echo ""
echo "=== Step 3: Copy .scm to Striim modules ==="

cp target/TranADScorer.jar target/TranADScorer.scm
cp target/TranADScorer.scm "$STRIIM_HOME/modules/TranADScorer.scm"
echo "Copied to $STRIIM_HOME/modules/TranADScorer.scm"

echo ""
echo "=== Done ==="
echo ""
echo "Next steps:"
echo "  1. Restart Striim (full restart required for .scm updates)"
echo "  2. Clear OP cache: rm -rf \$STRIIM_HOME/.striim/OpenProcessor/"
echo "  3. In the Striim console:"
echo '     LOAD OPEN PROCESSOR "/opt/Striim/modules/TranADScorer.scm";'
echo "  4. Paste the TQL from striim/TRANAD.tql"
echo "  5. Wire the OP in Flow Designer, then deploy and start"
