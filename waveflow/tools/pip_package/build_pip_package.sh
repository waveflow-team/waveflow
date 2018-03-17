#!/usr/bin/env bash

set -e

if [ $# -lt 1 ] ; then
    echo "No destination dir provided"
    exit 1
fi

CURR_DIR=$(cd $(dirname "$0") && pwd)
TMP_DIR=$(mktemp -d -t tmp.waveflow.XXX)
WF_DIR="${TMP_DIR}/waveflow"
DEST="$1"

mkdir ${WF_DIR}
cp ${CURR_DIR}/build_pip_package.runfiles/waveflow/waveflow/__init__.py ${WF_DIR}
cp -r ${CURR_DIR}/build_pip_package.runfiles/waveflow/waveflow/core ${WF_DIR}
cp -r ${CURR_DIR}/build_pip_package.runfiles/waveflow/waveflow/python ${WF_DIR}
cp ${CURR_DIR}/build_pip_package.runfiles/waveflow/waveflow/tools/pip_package/* ${TMP_DIR}

cd ${TMP_DIR}
python setup.py bdist_wheel

mkdir -p ${DEST}
cp dist/* ${DEST}
cd ${DEST}
rm -rf ${TMP_DIR}

echo "You can find your wheel file in directory: ${DEST}"



