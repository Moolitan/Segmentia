# 创建测试目录
mkdir -p /mnt/Large_Language_Model_Lab_1/wsh/fio_benchmark

# 在终端1执行：
while true; do
    printf '%s speed=' "$(date +%H:%M:%S.%3N)"
    cat /sys/bus/pci/devices/0000:99:00.0/current_link_speed
    printf ' width='
    cat /sys/bus/pci/devices/0000:99:00.0/current_link_width
    sleep 0.2
done | tee /tmp/nvme2_pcie_link.log


# 在终端2执行：
# 顺序写入测试
fio \
  --name=nvme-seq-write \
  --filename=/mnt/Large_Language_Model_Lab_1/wsh/fio_benchmark/fio_test_32g.bin \
  --size=32G \
  --rw=write \
  --bs=1M \
  --direct=1 \
  --ioengine=io_uring \
  --iodepth=32 \
  --numjobs=1 \
  --end_fsync=1 \
  --group_reporting

# 顺序读取测试
fio --name=nvme-seq-read \
    --filename=/mnt/Large_Language_Model_Lab_1/wsh/fio_benchmark/fio_test_32g.bin \
    --size=32G \
    --rw=read \
    --bs=1M \
    --direct=1 \
    --ioengine=io_uring \
    --iodepth=32 \
    --numjobs=1 \
    --readonly \
    --group_reporting \
    --output-format=json \
    --output=/mnt/Large_Language_Model_Lab_1/wsh/fio_benchmark/fio_seq_read.json


# 确认路径无误后：
rm -f /mnt/Large_Language_Model_Lab_1/wsh/fio_benchmark/fio_test_32g.bin
rmdir /mnt/Large_Language_Model_Lab_1/wsh/fio_benchmark


# 换不同参数测试
fio --name=nvme-seq-read-bs2m-psync \
    --filename=/mnt/Large_Language_Model_Lab_1/wsh/fio_benchmark/fio_test_32g.bin \
    --size=32G \
    --rw=read \
    --bs=2M \
    --direct=1 \
    --ioengine=psync \
    --iodepth=1 \
    --numjobs=8 \
    --readonly \
    --group_reporting \
    --output-format=json \
    --output=/mnt/Large_Language_Model_Lab_1/wsh/fio_benchmark/fio_seq_read_bs2m_psync.json